#!/usr/bin/env python3
"""
ecg_spectrum_of_list.py

https://chatgpt.com/c/6985de00-5238-838e-a9c7-7b780fa852d5

Batch ECG spectrum analyzer (FFT + Welch PSD) with:
- optional baseline removal (high-pass Butterworth, zero-phase filtfilt)
- optional sliding-window motion index (baseline/motion artifact proxy)

It can analyze:
1) A folder with signal files, using a list of filenames
2) A .zip containing signal members, using a list of member names
3) A single file (backward-compatible), if you pass a file and no list

Examples (FOLDER):
  python ecg_spectrum_of_list.py --source /data/ecg_npy --list records.txt --fs 200 --no-show --save-dir out_plots
  python ecg_spectrum_of_list.py --source /data/ecg_npy --files 1025_0.npy 1025_1.npy --fs 200 --save-dir out_plots

Examples (ZIP):
  python ecg_spectrum_of_list.py --source patient.zip --list members.txt --fs 200 --save-dir out_plots --no-show

Single file (legacy):
  python ecg_spectrum_of_list.py --source 1025_0.npy --fs 200
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

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


def load_signal_from_source(
    source: Path,
    name: Optional[str],
    *,
    npz_key: Optional[str],
    bin_dtype: str,
    endian: str,
    scale: float,
) -> Tuple[np.ndarray, str]:
    """
    source can be:
      - a directory: then name must be a filename inside it
      - a .zip: then name must be a member inside it
      - a file: then name must be None (single-file mode)
    Returns (x, label)
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(str(source))

    # single-file mode
    if source.is_file() and source.suffix.lower() != ".zip":
        if name is not None:
            raise ValueError("When --source is a single file, do not pass --files/--list.")
        b = source.read_bytes()
        x = _load_from_bytes(source.name, b, npz_key=npz_key, bin_dtype=bin_dtype, endian=endian, scale=scale)
        return x, source.name

    # zip mode
    if source.is_file() and source.suffix.lower() == ".zip":
        if not name:
            raise ValueError("When --source is a .zip, you must provide members via --files or --list.")
        with zipfile.ZipFile(source, "r") as zf:
            names = set(zf.namelist())
            if name not in names:
                # try a lenient match by basename
                bn = Path(name).name
                candidates = [n for n in names if Path(n).name == bn]
                if len(candidates) == 1:
                    name = candidates[0]
                else:
                    raise FileNotFoundError(f"Member '{name}' not found in zip.")
            with zf.open(name) as f:
                b = f.read()
        x = _load_from_bytes(name, b, npz_key=npz_key, bin_dtype=bin_dtype, endian=endian, scale=scale)
        return x, f"{source.name}:{name}"

    # directory mode
    if source.is_dir():
        if not name:
            raise ValueError("When --source is a folder, you must provide filenames via --files or --list.")
        p = source / name
        if not p.exists():
            # also try basename match inside folder
            bn = Path(name).name
            matches = list(source.glob(bn))
            if len(matches) == 1:
                p = matches[0]
            else:
                raise FileNotFoundError(str(p))
        b = p.read_bytes()
        x = _load_from_bytes(p.name, b, npz_key=npz_key, bin_dtype=bin_dtype, endian=endian, scale=scale)
        return x, p.name

    raise ValueError(f"Unsupported --source: {source}")


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
    if sp_signal is None:
        raise RuntimeError("scipy is required for motion index. Install: pip install scipy")

    win = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    win = max(256, win)
    hop = max(1, hop)
    win = min(win, x.size)

    eps = 1e-30
    t_list: List[float] = []
    mi_list: List[float] = []

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

    f_fft, a_fft = compute_fft_spectrum(x_proc, fs, nfft=nfft)

    have_welch = sp_signal is not None
    if have_welch:
        f_psd, pxx = compute_welch_psd(x_proc, fs, nperseg=welch_nperseg)
        pxx_db = 10.0 * np.log10(np.maximum(pxx, 1e-30))
    else:
        f_psd, pxx_db = None, None

    fmax = min(float(fmax), fs / 2.0)
    n_show = int(min(x_proc.size, max(1, round(float(tsec) * fs))))

    nrows = 4 if motion is not None else 3
    fig = plt.figure(figsize=(12, 9 if nrows == 4 else 8))

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

    ax2 = plt.subplot(nrows, 1, 2)
    m = f_fft <= fmax
    ax2.plot(f_fft[m], 20.0 * np.log10(np.maximum(a_fft[m], 1e-30)), linewidth=1.0)
    ax2.set_title("FFT magnitude spectrum (dB, single-sided)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(True, alpha=0.3)

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
# File list parsing
# ----------------------------

def read_list_file(p: Path) -> List[str]:
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def sanitize_stem(s: str) -> str:
    # safe-ish filename for outputs
    stem = Path(s).name
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        stem = stem.replace(ch, "_")
    return stem


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute & visualize ECG spectrum for one file or a list of records.")
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Folder with records OR a .zip OR a single signal file (.npy/.npz/.csv/.txt/.bin)",
    )

    # list of records (filenames inside folder OR member names inside zip)
    ap.add_argument("--files", nargs="*", default=None, help="Record filenames/members to analyze (space-separated).")
    ap.add_argument("--list", type=Path, default=None, help="Text file with record filenames/members (one per line).")

    ap.add_argument("--fs", type=float, default=200.0, help="Sampling frequency (Hz). Default: 200")
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
    ap.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, saves one PNG per record into this folder (recommended for batch).",
    )
    ap.add_argument("--no-show", action="store_true", help="Do not display interactive window(s).")

    # baseline removal (default highpass, per your preference)
    ap.add_argument(
        "--baseline",
        choices=["none", "highpass"],
        default="highpass",
        help="Baseline removal method before spectrum: none|highpass (default: highpass)",
    )
    ap.add_argument("--hp-cutoff", type=float, default=0.5, help="High-pass cutoff Hz. Default: 0.5")
    ap.add_argument("--hp-order", type=int, default=4, help="High-pass Butterworth order. Default: 4")

    # motion index
    ap.add_argument("--motion-index", action="store_true", help="Compute and plot sliding-window motion index")
    ap.add_argument("--mi-win", type=float, default=8.0, help="Motion-index window length in seconds (default: 8)")
    ap.add_argument("--mi-hop", type=float, default=1.0, help="Motion-index hop in seconds (default: 1)")
    ap.add_argument("--mi-low", type=float, default=0.1, help="Motion-index low band start Hz (default: 0.1)")
    ap.add_argument("--mi-high", type=float, default=0.7, help="Motion-index low band end Hz (default: 0.7)")
    ap.add_argument("--mi-qrs-low", type=float, default=5.0, help="Motion-index QRS band start Hz (default: 5)")
    ap.add_argument("--mi-qrs-high", type=float, default=15.0, help="Motion-index QRS band end Hz (default: 15)")
    ap.add_argument("--mi-on-processed", action="store_true", help="Compute motion index on processed instead of raw")

    args = ap.parse_args()

    # build the record list
    recs: List[str] = []
    if args.files:
        recs.extend(args.files)
    if args.list is not None:
        recs.extend(read_list_file(args.list))

    # If no list given: allow single-file mode
    if not recs:
        if args.source.is_file() and args.source.suffix.lower() != ".zip":
            recs = [None]  # marker for single-file mode
        else:
            raise SystemExit("No records specified. Use --files and/or --list (or pass a single file as --source).")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    total = len(recs)
    print(f"Source: {args.source}")
    print(f"Records to analyze: {total}")

    for idx, rec in enumerate(recs, start=1):
        print("\n" + "=" * 80)
        tag = rec if rec is not None else str(args.source)
        print(f"[{idx}/{total}] {tag}")

        try:
            x_raw, label = load_signal_from_source(
                args.source,
                rec,
                npz_key=args.npz_key,
                bin_dtype=args.bin_dtype,
                endian=args.endian,
                scale=args.scale,
            )
        except Exception as e:
            print(f"ERROR loading '{tag}': {e}")
            continue

        print(f"Loaded: {label}")
        print(f"Samples: {x_raw.size}, fs={args.fs} Hz, duration={x_raw.size/args.fs:.2f} s")
        print(
            f"Mean={np.mean(x_raw):.6g}, std={np.std(x_raw):.6g}, "
            f"min={np.min(x_raw):.6g}, max={np.max(x_raw):.6g}"
        )

        # baseline removal for spectrum/PSD
        x_proc = x_raw
        if args.baseline == "highpass":
            try:
                x_proc = remove_baseline_highpass(x_raw, fs=args.fs, cutoff_hz=args.hp_cutoff, order=args.hp_order)
                print(f"Applied baseline removal: highpass {args.hp_cutoff} Hz (order {args.hp_order})")
            except Exception as e:
                print(f"ERROR baseline removal for '{tag}': {e} (continuing with raw)")
                x_proc = x_raw
        else:
            print("Baseline removal: none")

        # motion index
        motion = None
        if args.motion_index:
            try:
                x_mi = x_proc if args.mi_on_processed else x_raw
                t_mi, mi = compute_motion_index_series(
                    x_mi,
                    fs=args.fs,
                    win_s=args.mi_win,
                    hop_s=args.mi_hop,
                    low_band=(args.mi_low, args.mi_high),
                    qrs_band=(args.mi_qrs_low, args.mi_qrs_high),
                    welch_nperseg=min(1024, max(128, int(args.mi_win * args.fs))),
                )
                motion = {
                    "t": t_mi,
                    "mi": mi,
                    "low": (args.mi_low, args.mi_high),
                    "qrs": (args.mi_qrs_low, args.mi_qrs_high),
                }
                src = "processed" if args.mi_on_processed else "raw"
                print(f"Motion index computed on: {src}")
            except Exception as e:
                print(f"ERROR motion index for '{tag}': {e} (skipping MI)")
                motion = None

        # output filename for plot
        save_path = None
        if args.save_dir is not None:
            base = sanitize_stem(tag if rec is not None else Path(args.source).name)
            save_path = args.save_dir / f"{base}.png"

        plot_ecg_and_spectrum(
            x_proc=x_proc,
            fs=args.fs,
            label=label,
            x_raw=x_raw if args.baseline != "none" else None,
            tsec=args.tsec,
            fmax=args.fmax,
            nfft=args.nfft,
            welch_nperseg=args.welch_nperseg,
            motion=motion,
            save=save_path,
            show=(not args.no_show),
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""




"""