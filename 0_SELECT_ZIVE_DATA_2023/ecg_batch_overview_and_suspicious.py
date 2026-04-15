"""
https://chatgpt.com/c/69c62f47-b964-838b-81aa-0b1bfd55c7b7

these is a full batch version that:

reads all .npy ECG files from a folder,
saves *_overview.png for the whole 10-minute record,
saves *_suspicious.png with automatically selected suspicious fragments,
saves one CSV summary for all detected suspicious windows.

It uses:

overview plot: min–max envelope across the full signal,
fragment selection based on:
highest local variance,
flat / saturation-like regions,
dropout-like regions,
extreme amplitudes.


python ecg_batch_overview_and_suspicious.py \
  --src-dir /path/to/npy_files \
  --out-dir /path/to/output

python ecg_batch_overview_and_suspicious.py \
  --src-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test \
  --out-dir output_double_pngs




"""




from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Data structures
# =========================

@dataclass
class SuspiciousFragment:
    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    reason: str
    score: float


# =========================
# Loading
# =========================

def load_ecg_npy(path: Path) -> np.ndarray:
    """
    Load ECG signal from .npy file and return as 1D float array.
    Supports:
      - shape (N,)
      - shape (N,1)
      - shape (1,N)
    """
    arr = np.load(path, allow_pickle=False)

    if arr.ndim == 1:
        ecg = arr
    elif arr.ndim == 2 and 1 in arr.shape:
        ecg = arr.reshape(-1)
    else:
        raise ValueError(
            f"Unsupported ECG array shape in {path.name}: {arr.shape}"
        )

    ecg = np.asarray(ecg, dtype=float).ravel()

    if ecg.size == 0:
        raise ValueError(f"Empty ECG array in {path.name}")

    return ecg


def iter_npy_files(src_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(src_dir.glob(pattern))


# =========================
# Overview plot
# =========================

def compute_envelope(
    ecg: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compress ECG into n_bins using min/max envelope.

    Returns:
        x_bins: bin indices
        y_min: min value in each bin
        y_max: max value in each bin
        bin_size: number of original samples in one bin
    """
    n = len(ecg)
    n_bins = max(1, min(n_bins, n))
    bin_size = int(np.ceil(n / n_bins))

    padded_len = n_bins * bin_size
    pad = padded_len - n
    if pad > 0:
        ecg = np.pad(ecg, (0, pad), mode="edge")

    reshaped = ecg.reshape(n_bins, bin_size)
    y_min = reshaped.min(axis=1)
    y_max = reshaped.max(axis=1)
    x_bins = np.arange(n_bins)

    return x_bins, y_min, y_max, bin_size


def save_ecg_overview_png(
    ecg: np.ndarray,
    out_path: Path,
    fs: int = 200,
    width_px: int = 1600,
    height_in: float = 4.5,
    dpi: int = 100,
    title: str | None = None,
) -> None:
    n = len(ecg)
    duration_sec = n / fs
    n_bins = min(width_px, n)

    x_bins, y_min, y_max, bin_size = compute_envelope(ecg, n_bins=n_bins)
    t = (x_bins * bin_size + bin_size / 2) / fs

    fig = plt.figure(figsize=(width_px / dpi, height_in), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.fill_between(t, y_min, y_max, linewidth=0, alpha=0.8)
    y_mid = 0.5 * (y_min + y_max)
    ax.plot(t, y_mid, linewidth=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if title is None:
        title = out_path.stem
    ax.set_title(f"{title} | duration={duration_sec:.1f}s | samples={n}")

    minute_marks = np.arange(0, duration_sec + 0.1, 60)
    ax.set_xticks(minute_marks)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# Suspicious fragment detection
# =========================

def sliding_windows(
    n_samples: int,
    win_size: int,
    step: int,
) -> Iterable[tuple[int, int]]:
    if win_size <= 0:
        raise ValueError("win_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")

    if n_samples < win_size:
        yield 0, n_samples
        return

    for start in range(0, n_samples - win_size + 1, step):
        end = start + win_size
        yield start, end

    # ensure last tail is also covered
    last_start = n_samples - win_size
    last_end = n_samples
    if last_start >= 0:
        if (n_samples - win_size) % step != 0:
            yield last_start, last_end


def compute_window_scores(
    ecg: np.ndarray,
    fs: int,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    flat_diff_thr: float = 0.005,
    zero_thr: float = 1e-6,
    dropout_var_thr: float = 1e-4,
) -> list[dict]:
    """
    Compute suspiciousness scores for overlapping windows.

    Notes:
    - Thresholds depend on ECG amplitude scale and may need tuning.
    - Works best as an initial screening tool.
    """
    ecg = np.asarray(ecg, dtype=float).ravel()

    win_size = max(1, int(round(window_sec * fs)))
    step = max(1, int(round(step_sec * fs)))

    results: list[dict] = []

    for start, end in sliding_windows(len(ecg), win_size, step):
        w = ecg[start:end]
        if w.size == 0:
            continue

        d = np.diff(w)

        var_score = float(np.var(w))
        std_score = float(np.std(w))
        amp_score = float(np.max(np.abs(w)))
        ptp_score = float(np.ptp(w))

        if d.size > 0:
            flat_fraction = float(np.mean(np.abs(d) < flat_diff_thr))
            repeated_fraction = float(np.mean(np.abs(d) < zero_thr))
            diff_score = float(np.max(np.abs(d)))
        else:
            flat_fraction = 0.0
            repeated_fraction = 0.0
            diff_score = 0.0

        zero_fraction = float(np.mean(np.abs(w) < zero_thr))

        dropout_score = 0.0
        if var_score < dropout_var_thr:
            dropout_score += 1.0
        dropout_score += repeated_fraction
        dropout_score += zero_fraction

        results.append(
            {
                "start": start,
                "end": end,
                "start_sec": start / fs,
                "end_sec": end / fs,
                "var_score": var_score,
                "std_score": std_score,
                "amp_score": amp_score,
                "ptp_score": ptp_score,
                "flat_score": flat_fraction,
                "zero_fraction": zero_fraction,
                "repeated_fraction": repeated_fraction,
                "dropout_score": dropout_score,
                "diff_score": diff_score,
            }
        )

    return results


def pick_top_windows(
    results: list[dict],
    key: str,
    top_n: int,
    largest: bool = True,
) -> list[dict]:
    return sorted(results, key=lambda x: x[key], reverse=largest)[:top_n]


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def merge_selected_windows(
    selected: list[SuspiciousFragment],
    min_separation_sec: float,
    fs: int,
) -> list[SuspiciousFragment]:
    """
    Remove overlapping / near-duplicate windows.
    Assumes selected list is already in priority order.
    """
    kept: list[SuspiciousFragment] = []
    gap = int(round(min_separation_sec * fs))

    for item in selected:
        s1, e1 = item.start_idx, item.end_idx
        too_close = False

        for k in kept:
            s2, e2 = k.start_idx, k.end_idx
            if overlap(s1 - gap, e1 + gap, s2, e2):
                too_close = True
                break

        if not too_close:
            kept.append(item)

    return kept


def detect_suspicious_fragments(
    ecg: np.ndarray,
    fs: int = 200,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    top_n_each: int = 4,
    min_separation_sec: float = 2.0,
    flat_diff_thr: float = 0.005,
    zero_thr: float = 1e-6,
    dropout_var_thr: float = 1e-4,
) -> tuple[list[SuspiciousFragment], list[dict]]:
    results = compute_window_scores(
        ecg=ecg,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        flat_diff_thr=flat_diff_thr,
        zero_thr=zero_thr,
        dropout_var_thr=dropout_var_thr,
    )

    selected: list[SuspiciousFragment] = []

    for x in pick_top_windows(results, "var_score", top_n_each, largest=True):
        selected.append(
            SuspiciousFragment(
                start_idx=x["start"],
                end_idx=x["end"],
                start_sec=x["start_sec"],
                end_sec=x["end_sec"],
                reason="highest local variance",
                score=x["var_score"],
            )
        )

    for x in pick_top_windows(results, "flat_score", top_n_each, largest=True):
        selected.append(
            SuspiciousFragment(
                start_idx=x["start"],
                end_idx=x["end"],
                start_sec=x["start_sec"],
                end_sec=x["end_sec"],
                reason="flat / saturation-like",
                score=x["flat_score"],
            )
        )

    for x in pick_top_windows(results, "dropout_score", top_n_each, largest=True):
        selected.append(
            SuspiciousFragment(
                start_idx=x["start"],
                end_idx=x["end"],
                start_sec=x["start_sec"],
                end_sec=x["end_sec"],
                reason="dropout-like",
                score=x["dropout_score"],
            )
        )

    for x in pick_top_windows(results, "amp_score", top_n_each, largest=True):
        selected.append(
            SuspiciousFragment(
                start_idx=x["start"],
                end_idx=x["end"],
                start_sec=x["start_sec"],
                end_sec=x["end_sec"],
                reason="extreme amplitude",
                score=x["amp_score"],
            )
        )

    for x in pick_top_windows(results, "diff_score", top_n_each, largest=True):
        selected.append(
            SuspiciousFragment(
                start_idx=x["start"],
                end_idx=x["end"],
                start_sec=x["start_sec"],
                end_sec=x["end_sec"],
                reason="sharp derivative / spike-like",
                score=x["diff_score"],
            )
        )

    selected = merge_selected_windows(
        selected=selected,
        min_separation_sec=min_separation_sec,
        fs=fs,
    )

    return selected, results


# =========================
# Suspicious plot
# =========================

def save_suspicious_fragments_png(
    ecg: np.ndarray,
    fragments: list[SuspiciousFragment],
    out_path: Path,
    fs: int = 200,
    max_plots: int = 10,
    ncols: int = 2,
    title: str | None = None,
) -> None:
    fragments = fragments[:max_plots]

    if not fragments:
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No suspicious fragments detected",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        if title is None:
            title = out_path.stem
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(fragments)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 2.8 * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, frag in zip(axes, fragments):
        s, e = frag.start_idx, frag.end_idx
        w = ecg[s:e]
        t = np.arange(s, e) / fs

        ax.plot(t, w, linewidth=0.8)
        ax.set_title(
            f"{frag.reason}\n"
            f"{frag.start_sec:.1f}-{frag.end_sec:.1f}s | score={frag.score:.4g}",
            fontsize=10,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amp")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    if title is None:
        title = out_path.stem

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================
# CSV output
# =========================

def write_fragments_csv(
    rows: list[dict],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "start_idx",
        "end_idx",
        "start_sec",
        "end_sec",
        "reason",
        "score",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create batch ECG overview and suspicious-fragment PNGs from .npy files."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Directory containing .npy ECG files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where output PNGs and CSV will be saved",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=200,
        help="Sampling frequency in Hz (default: 200)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .npy files recursively",
    )

    # Overview plot params
    parser.add_argument(
        "--width-px",
        type=int,
        default=1600,
        help="Approximate overview plot width in pixels / envelope bins (default: 1600)",
    )
    parser.add_argument(
        "--overview-height-in",
        type=float,
        default=4.5,
        help="Overview figure height in inches (default: 4.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI (default: 100)",
    )

    # Suspicious fragment params
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Fragment window length in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=1.0,
        help="Sliding step in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--top-n-each",
        type=int,
        default=4,
        help="Top windows to take for each suspiciousness metric before deduplication (default: 4)",
    )
    parser.add_argument(
        "--min-separation-sec",
        type=float,
        default=2.0,
        help="Minimum separation between selected suspicious windows in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-fragment-plots",
        type=int,
        default=10,
        help="Maximum suspicious fragments to show in one PNG (default: 10)",
    )
    parser.add_argument(
        "--fragment-cols",
        type=int,
        default=2,
        help="Number of columns in suspicious fragment page (default: 2)",
    )

    # Thresholds
    parser.add_argument(
        "--flat-diff-thr",
        type=float,
        default=0.005,
        help="Threshold for near-flat consecutive sample differences (default: 0.005)",
    )
    parser.add_argument(
        "--zero-thr",
        type=float,
        default=1e-6,
        help="Threshold for near-zero samples (default: 1e-6)",
    )
    parser.add_argument(
        "--dropout-var-thr",
        type=float,
        default=1e-4,
        help="Variance threshold for dropout-like windows (default: 1e-4)",
    )

    args = parser.parse_args()

    src_dir = args.src_dir
    out_dir = args.out_dir

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    npy_files = iter_npy_files(src_dir, recursive=args.recursive)
    if not npy_files:
        print(f"No .npy files found in: {src_dir}")
        return

    overview_dir = out_dir / "overview_png"
    suspicious_dir = out_dir / "suspicious_png"
    csv_path = out_dir / "suspicious_fragments.csv"

    print(f"Found {len(npy_files)} .npy files")

    ok_count = 0
    fail_count = 0
    csv_rows: list[dict] = []

    for i, npy_path in enumerate(npy_files, start=1):
        try:
            ecg = load_ecg_npy(npy_path)

            overview_path = overview_dir / f"{npy_path.stem}_overview.png"
            suspicious_path = suspicious_dir / f"{npy_path.stem}_suspicious.png"

            save_ecg_overview_png(
                ecg=ecg,
                out_path=overview_path,
                fs=args.fs,
                width_px=args.width_px,
                height_in=args.overview_height_in,
                dpi=args.dpi,
                title=f"{npy_path.stem} - ECG overview",
            )

            fragments, _scores = detect_suspicious_fragments(
                ecg=ecg,
                fs=args.fs,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                top_n_each=args.top_n_each,
                min_separation_sec=args.min_separation_sec,
                flat_diff_thr=args.flat_diff_thr,
                zero_thr=args.zero_thr,
                dropout_var_thr=args.dropout_var_thr,
            )

            save_suspicious_fragments_png(
                ecg=ecg,
                fragments=fragments,
                out_path=suspicious_path,
                fs=args.fs,
                max_plots=args.max_fragment_plots,
                ncols=args.fragment_cols,
                title=f"{npy_path.stem} - suspicious fragments",
            )

            for frag in fragments:
                csv_rows.append(
                    {
                        "filename": npy_path.name,
                        "start_idx": frag.start_idx,
                        "end_idx": frag.end_idx,
                        "start_sec": f"{frag.start_sec:.3f}",
                        "end_sec": f"{frag.end_sec:.3f}",
                        "reason": frag.reason,
                        "score": f"{frag.score:.8g}",
                    }
                )

            ok_count += 1
            print(
                f"[{i}/{len(npy_files)}] OK   {npy_path.name} "
                f"-> {overview_path.name}, {suspicious_path.name}"
            )

        except Exception as exc:
            fail_count += 1
            print(f"[{i}/{len(npy_files)}] FAIL {npy_path.name}: {exc}")

    write_fragments_csv(csv_rows, csv_path)

    print()
    print(f"Done. Saved: {ok_count}, Failed: {fail_count}")
    print(f"Overview PNGs:    {overview_dir}")
    print(f"Suspicious PNGs:  {suspicious_dir}")
    print(f"Fragments CSV:    {csv_path}")


if __name__ == "__main__":
    main()