#!/usr/bin/env python3
"""
ecg_spectrum.py

Compute and visualize ECG spectrum (FFT + Welch PSD) with:
- optional baseline removal (high-pass Butterworth, zero-phase filtfilt)
- optional sliding-window motion index (baseline/motion artifact proxy)

Examples:
  python ecg_spectrum.py /path/to/1025_0.npy --fs 200
  python ecg_spectrum.py /path/to/1025_0.zip --fs 200 --member 1025_0.npy

path="/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_ORIG/ecg_zive_npy/1025_0.npy"
python ecg_spectrum.py "$path" --fs 200 --tsec 15 --fmax 60 --save out.png
Baseline removal:
python ecg_spectrum.py "$path" --fs 200 --baseline highpass --hp-cutoff 0.5
Motion index:
python ecg_spectrum.py "$path" --fs 200 --motion-index --mi-win 8 --mi-hop 1 --mi-csv mi.csv
Save figure:
python ecg_spectrum.py "$path" --fs 200 --save out.png --no-show

Examples:

Baseline removal:
  python ecg_spectrum.py 1025_0.npy --fs 200 --baseline highpass --hp-cutoff 0.5

Motion index:
  python ecg_spectrum.py 1025_0.npy --fs 200 --motion-index --mi-win 8 --mi-hop 1 --mi-csv mi.csv

Save figure:
  python ecg_spectrum.py 1025_0.npy --fs 200 --save out.png --no-show

Raw binary (.bin):
  python ecg_spectrum.py ecg.bin --fs 200 --bin-dtype i4 --endian big --scale 1.0
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import signal as sp_signal
except ImportError:
    sp_signal = None


# ----------------------------
# Loading helpers
# ----------------------------

_SIGNAL_EXTS = (".npy", ".npz", ".csv", ".txt", ".bin")


def _pick_member(names, member: Optional[str]) -> str:
    if member:
        if member not in names:
            raise FileNotFoundError(f"--member '{member}' not found in zip. Available: {names[:50]}")
        return member

    for n in names:
        if n.lower().endswith(_SIGNAL_EXTS) and not n.endswith("/"):
            return n

    raise FileNotFoundError(f"No supported signal files found in zip. Supported: {_SIGNAL_EXTS}")


def _load_from_bytes(
    name: str,
    b: bytes,
    *,
    npz_key: Optional[str],
    bin_dtype: str,
    endian: str,
    scale: float,
) -> np.ndarray:
    lower = name.lower()

    if lower.endswith(".npy"):
        x = np.load(io.BytesIO(b), allow_pickle=False)

    elif lower.endswith(".npz"):
        z = np.load(io.BytesIO(b), allow_pickle=False)
        keys = list(z.keys())
        if not keys:
            raise ValueError("NPZ has no arrays.")
        key = keys[0] if npz_key is None else npz_key
        if key not in z:
            raise KeyError(f"--npz-key '{key}' not found. Keys: {keys}")
        x = z[key]

    elif lower.endswith((".csv", ".txt")):
        s = b.decode("utf-8", errors="ignore")
        # try comma first, then whitespace
        try:
            arr = np.genfromtxt(io.StringIO(s), delimiter=",")
            x = arr[:, 0] if arr.ndim == 2 else arr
        except Exception:
            arr = np.genfromtxt(io.StringIO(s))
            x = arr[:, 0] if arr.ndim == 2 else arr

    elif lower.endswith(".bin"):
        dt = np.dtype(bin_dtype)
        e = endian.lower()
        if e in ("big", "be", ">"):
            dt = dt.newbyteorder(">")
        elif e in ("little", "le", "<"):
            dt = dt.newbyteorder("<")
        else:
            raise ValueError("--endian must be one of: big, little")

        x = np.frombuffer(b, dtype=dt).astype(np.float64) * float(scale)

    else:
        raise ValueError(f"Unsupported file type: {name}")

    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Signal must be 1D after squeeze; got shape {x.shape}")

    x = x.astype(np.float64, copy=False)
    good = np.isfinite(x)
    if not np.all(good):
        x = x[good]

    if x.size < 4:
        raise ValueError(f"Signal too short: {x.size} samples")

    return x


def load_signal(
    path: Path,
    *,
    member: Optional[str],
    npz_key: Optional[str],
    bin_dtype: str,
    endian: str,
    scale: float,
) -> Tuple[np.ndarray, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            chosen = _pick_member(names, member)
            with zf.open(chosen) as f:
                b = f.read()
        x = _load_from_bytes(chosen, b, npz_key=npz_key, bin_dtype=bin_dtype, endian=endian, scale=scale)
        label = f"{path.name}:{chosen}"
        return x, label

    b = path.read_bytes()
    x = _load_from_bytes(path.name, b, npz_key=npz_key, bin_dtype=bin_dtype, endian=endian, scale=scale)
    return x, path.name


# ----------------------------
# Baseline removal
# ----------------------------

def remove_baseline_highpass(x: np.ndarray, fs: float, cutoff_hz: float = 0.5, order: int = 4) -> np.ndarray:
    """Zero-phase high-pass Butterworth baseline removal (requires scipy)."""
    if sp_signal is None:
        raise RuntimeError("scipy is required for high-pass baseline removal. Install: pip install scipy")

    nyq = 0.5 * fs
    wn = float(cutoff_hz) / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"Invalid hp cutoff: {cutoff_hz} Hz for fs={fs}")

    b, a = sp_signal.butter(int(order), wn, btype="highpass")
    return sp_signal.filtfilt(b, a, x)


# ----------------------------
# Spectrum computation
# ----------------------------

def compute_fft_spectrum(x: np.ndarray, fs: float, nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-sided amplitude spectrum via rFFT.
    Output amplitude is normalized for window sum (relative comparisons OK).
    Returns (freqs_hz, amplitude).
    """
    x0 = x - np.mean(x)
    n = x0.size

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(n)))
        nfft = min(nfft, 262144)  # cap for very long signals

    w = np.hanning(n)
    xw = x0 * w
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    amp = np.abs(X) / (np.sum(w) / 2.0)
    return freqs, amp


def compute_welch_psd(x: np.ndarray, fs: float, nperseg: int = 4096, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD (requires scipy). Returns (freqs_hz, psd_density)."""
    if sp_signal is None:
        raise RuntimeError("scipy is required for Welch PSD. Install: pip install scipy")

    x0 = x - np.mean(x)
    nperseg = int(min(nperseg, x0.size))
    if nperseg < 256:
        nperseg = int(x0.size)

    if noverlap is None:
        noverlap = int(0.5 * nperseg)

    f, pxx = sp_signal.welch(
        x0,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        average="mean",
    )
    return f, pxx


# ----------------------------
# Motion index (sliding window)
# ----------------------------

def _bandpower_from_psd(f: np.ndarray, pxx: np.ndarray, f1: float, f2: float) -> float:
    """Integrate PSD over [f1, f2] using trapezoidal rule."""
    f1, f2 = float(f1), float(f2)
    if f2 <= f1:
        return 0.0
    m = (f >= f1) & (f <= f2)
    if not np.any(m):
        return 0.0
    return float(np.trapz(pxx[m], f[m]))


def compute_motion_index_series(
    x: np.ndarray,
    fs: float,
    win_s: float = 8.0,
    hop_s: float = 1.0,
    low_band: Tuple[float, float] = (0.1, 0.7),
    qrs_band: Tuple[float, float] = (5.0, 15.0),
    welch_nperseg: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window motion index:
      MI = P(low_band) / (P(qrs_band) + eps)

    low_band captures baseline wander / electrode motion (slow).
    qrs_band normalizes by QRS energy.

    Returns (t_center_s, mi).
    """
    if sp_signal is None:
        raise RuntimeError("scipy is required for motion index. Install: pip install scipy")

    win = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    win = max(256, win)
    hop = max(1, hop)
    win = min(win, x.size)

    eps = 1e-30
    t_list = []
    mi_list = []

    nperseg = int(min(welch_nperseg, win))
    nperseg = max(128, nperseg)
    noverlap = int(0.5 * nperseg)

    for start in range(0, x.size - win + 1, hop):
        seg = x[start:start + win]
        seg = seg - np.mean(seg)

        f, pxx = sp_signal.welch(
            seg,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            average="mean",
        )

        p_low = _bandpower_from_psd(f, pxx, low_band[0], low_band[1])
        p_qrs = _bandpower_from_psd(f, pxx, qrs_band[0], qrs_band[1])
        mi = p_low / (p_qrs + eps)

        t_center = (start + win / 2) / fs
        t_list.append(t_center)
        mi_list.append(mi)

    return np.asarray(t_list), np.asarray(mi_list)


# ----------------------------
# Plotting
# ----------------------------

def plot_ecg_and_spectrum(
    x_proc: np.ndarray,
    fs: float,
    label: str,
    *,
    x_raw: Optional[np.ndarray] = None,
    tsec: float = 10.0,
    fmax: float = 60.0,
    nfft: Optional[int] = None,
    welch_nperseg: int = 4096,
    motion: Optional[Dict[str, Any]] = None,
    save: Optional[Path] = None,
    show: bool = True,
) -> None:
    t = np.arange(x_proc.size) / fs

    # spectra on processed signal
    f_fft, a_fft = compute_fft_spectrum(x_proc, fs, nfft=nfft)

    have_welch = sp_signal is not None
    if have_welch:
        f_psd, pxx = compute_welch_psd(x_proc, fs, nperseg=welch_nperseg)
        pxx_db = 10.0 * np.log10(np.maximum(pxx, 1e-30))
    else:
        f_psd, pxx_db = None, None

    fmax = min(float(fmax), fs / 2.0)

    # time window
    n_show = int(min(x_proc.size, max(1, round(float(tsec) * fs))))

    nrows = 4 if motion is not None else 3
    fig = plt.figure(figsize=(12, 9 if nrows == 4 else 8))

    # Time plot
    ax1 = plt.subplot(nrows, 1, 1)
    if x_raw is not None:
        ax1.plot(t[:n_show], x_raw[:n_show], linewidth=0.8, alpha=0.6, label="raw")
    ax1.plot(t[:n_show], x_proc[:n_show], linewidth=1.0, label="processed")
    ax1.set_title(f"ECG (first {t[:n_show][-1]:.2f} s) — {label}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    if x_raw is not None:
        ax1.legend(loc="upper right")

    # FFT
    ax2 = plt.subplot(nrows, 1, 2)
    m = f_fft <= fmax
    ax2.plot(f_fft[m], 20.0 * np.log10(np.maximum(a_fft[m], 1e-30)), linewidth=1.0)
    ax2.set_title("FFT magnitude spectrum (dB, single-sided)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, alpha=0.3)

    # Welch PSD
    ax3 = plt.subplot(nrows, 1, 3)
    if have_welch:
        m2 = f_psd <= fmax
        ax3.plot(f_psd[m2], pxx_db[m2], linewidth=1.0)
        ax3.set_title("Welch PSD (dB/Hz)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("PSD (dB/Hz)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.01, 0.5, "scipy not installed → Welch PSD disabled", transform=ax3.transAxes)
        ax3.axis("off")

    # Motion index
    if motion is not None:
        ax4 = plt.subplot(nrows, 1, 4)
        ax4.plot(motion["t"], motion["mi"], linewidth=1.0)
        ax4.set_title(
            f"Motion index = P({motion['low'][0]}–{motion['low'][1]} Hz) / "
            f"P({motion['qrs'][0]}–{motion['qrs'][1]} Hz)"
        )
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("MI (ratio)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Saved figure to: {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute & visualize ECG spectrum from zip or file.")
    ap.add_argument("path", type=Path, help="Path to .zip or signal file (.npy/.npz/.csv/.txt/.bin)")
    ap.add_argument("--fs", type=float, default=200.0, help="Sampling frequency (Hz). Default: 200")
    ap.add_argument("--member", type=str, default=None, help="Zip member path to load (if input is .zip)")
    ap.add_argument("--npz-key", type=str, default=None, help="Array key for .npz (default: first key)")
    ap.add_argument("--tsec", type=float, default=10.0, help="Seconds to show in time plot (default: 10)")
    ap.add_argument("--fmax", type=float, default=60.0, help="Max frequency to display (Hz). Default: 60")
    ap.add_argument("--nfft", type=int, default=None, help="FFT length (default: auto, capped)")
    ap.add_argument("--welch-nperseg", type=int, default=4096, help="Welch nperseg (default: 4096)")

    # binary options
    ap.add_argument("--bin-dtype", type=str, default="i4", help="Raw .bin dtype (e.g., i2, i4, f4). Default: i4")
    ap.add_argument("--endian", type=str, default="big", help="Endianness for .bin: big or little. Default: big")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply raw .bin samples by this scale. Default: 1.0")

    # output
    ap.add_argument("--save", type=Path, default=None, help="Save figure to PNG path")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactive window")

    # baseline removal (default = highpass, per your preference)
    ap.add_argument("--baseline", choices=["none", "highpass"], default="highpass",
                    help="Baseline removal method before spectrum: none|highpass (default: highpass)")
    ap.add_argument("--hp-cutoff", type=float, default=0.5,
                    help="High-pass cutoff Hz (for --baseline highpass). Default: 0.5")
    ap.add_argument("--hp-order", type=int, default=4,
                    help="High-pass Butterworth order (for --baseline highpass). Default: 4")

    # motion index
    ap.add_argument("--motion-index", action="store_true",
                    help="Compute and plot sliding-window motion index (computed on RAW by default)")
    ap.add_argument("--mi-win", type=float, default=8.0,
                    help="Motion-index window length in seconds (default: 8)")
    ap.add_argument("--mi-hop", type=float, default=1.0,
                    help="Motion-index hop in seconds (default: 1)")
    ap.add_argument("--mi-low", type=float, default=0.1,
                    help="Motion-index low band start Hz (default: 0.1)")
    ap.add_argument("--mi-high", type=float, default=0.7,
                    help="Motion-index low band end Hz (default: 0.7)")
    ap.add_argument("--mi-qrs-low", type=float, default=5.0,
                    help="Motion-index QRS band start Hz (default: 5)")
    ap.add_argument("--mi-qrs-high", type=float, default=15.0,
                    help="Motion-index QRS band end Hz (default: 15)")
    ap.add_argument("--mi-csv", type=Path, default=None,
                    help="Optional CSV output for motion index (time_s, motion_index)")
    ap.add_argument("--mi-on-processed", action="store_true",
                    help="Compute motion index on processed (baseline-filtered) signal instead of raw")

    args = ap.parse_args()

    x_raw, label = load_signal(
        args.path,
        member=args.member,
        npz_key=args.npz_key,
        bin_dtype=args.bin_dtype,
        endian=args.endian,
        scale=args.scale,
    )

    print(f"Loaded: {label}")
    print(f"Samples: {x_raw.size}, fs={args.fs} Hz, duration={x_raw.size/args.fs:.2f} s")
    print(f"Mean={np.mean(x_raw):.6g}, std={np.std(x_raw):.6g}, min={np.min(x_raw):.6g}, max={np.max(x_raw):.6g}")

    # baseline removal for spectrum/PSD
    x_proc = x_raw
    if args.baseline == "highpass":
        x_proc = remove_baseline_highpass(x_raw, fs=args.fs, cutoff_hz=args.hp_cutoff, order=args.hp_order)
        print(f"Applied baseline removal: highpass {args.hp_cutoff} Hz (order {args.hp_order})")
    else:
        print("Baseline removal: none")

    # motion index (default on RAW; can be forced to processed)
    motion = None
    if args.motion_index:
        x_mi = x_proc if args.mi_on_processed else x_raw
        t_mi, mi = compute_motion_index_series(
            x_mi,
            fs=args.fs,
            win_s=args.mi_win,
            hop_s=args.mi_hop,
            low_band=(args.mi_low, args.mi_high),
            qrs_band=(args.mi_qrs_low, args.mi_qrs_high),
            welch_nperseg=min(1024, int(args.mi_win * args.fs)),
        )
        motion = {
            "t": t_mi,
            "mi": mi,
            "low": (args.mi_low, args.mi_high),
            "qrs": (args.mi_qrs_low, args.mi_qrs_high),
        }

        if args.mi_csv is not None:
            args.mi_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.mi_csv.open("w", encoding="utf-8") as f:
                f.write("time_s,motion_index\n")
                for tt, vv in zip(t_mi, mi):
                    f.write(f"{tt:.3f},{vv:.10g}\n")
            print(f"Saved motion index CSV to: {args.mi_csv}")

        src = "processed" if args.mi_on_processed else "raw"
        print(f"Motion index computed on: {src}")

    plot_ecg_and_spectrum(
        x_proc=x_proc,
        fs=args.fs,
        label=label,
        x_raw=x_raw if args.baseline != "none" else None,  # overlay raw only when processing is applied
        tsec=args.tsec,
        fmax=args.fmax,
        nfft=args.nfft,
        welch_nperseg=args.welch_nperseg,
        motion=motion,
        save=args.save,
        show=(not args.no_show),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
path="/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_ORIG/ecg_zive_npy/1009_0.npy"
python ecg_spectrum.py "$path" --fs 200 --baseline highpass --hp-cutoff 0.5


"""