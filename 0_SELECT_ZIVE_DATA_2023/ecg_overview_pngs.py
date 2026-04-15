"""

https://chatgpt.com/c/69c62f47-b964-838b-81aa-0b1bfd55c7b7

script that:

- reads all .npy ECG files from a folder,
- creates a compact full-record overview,
- saves one PNG per record.

It uses the min–max envelope method, which is much better than plotting 
every 5th or 10th sample for long ECG quality screening.

compact view is:

- compress the whole signal to screen width,
- for each horizontal pixel/bin, calculate min and max,
- draw the envelope of the signal

A few practical notes:

For your 10-minute, 200 Hz signals, default --width-px 1600 is reasonable.
If you want even more compact pictures, reduce it to 1000.
If you want slightly more detail, increase it to 2000.

python ecg_overview_pngs.py --src-dir /path/to/npy_files --out-dir /path/to/output_pngs

python ecg_overview_pngs.py --src-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui --out-dir output_pngs
python ecg_overview_pngs.py --src-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test --out-dir output_pngs





"""


from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_ecg_npy(path: Path) -> np.ndarray:
    """
    Load ECG signal from .npy file and return as 1D float array.
    """
    arr = np.load(path, allow_pickle=False)

    if arr.ndim == 1:
        ecg = arr
    elif arr.ndim == 2:
        # If array is shape (N,1) or (1,N), flatten it
        if 1 in arr.shape:
            ecg = arr.reshape(-1)
        else:
            raise ValueError(
                f"Unsupported 2D shape for ECG file {path.name}: {arr.shape}. "
                f"Expected 1D or single-channel 2D array."
            )
    else:
        raise ValueError(
            f"Unsupported array ndim for ECG file {path.name}: {arr.ndim}"
        )

    return np.asarray(ecg, dtype=float)


def compute_envelope(
    ecg: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compress ECG into n_bins using min/max envelope.

    Returns:
        x_bins: bin indices
        y_min: min value in each bin
        y_max: max value in each bin
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

    return x_bins, y_min, y_max


def save_ecg_overview_png(
    ecg: np.ndarray,
    out_path: Path,
    fs: int = 200,
    width_px: int = 1600,
    height_in: float = 4.5,
    dpi: int = 100,
    title: str | None = None,
) -> None:
    """
    Save compact ECG overview as PNG using min/max envelope.
    """
    n = len(ecg)
    duration_sec = n / fs
    n_bins = min(width_px, n)

    x_bins, y_min, y_max = compute_envelope(ecg, n_bins=n_bins)

    # Convert bin index to seconds (bin centers)
    bin_size = int(np.ceil(n / n_bins))
    t = (x_bins * bin_size + bin_size / 2) / fs

    fig = plt.figure(figsize=(width_px / dpi, height_in), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.fill_between(t, y_min, y_max, linewidth=0, alpha=0.8)

    # Optional middle line for rough trend
    y_mid = 0.5 * (y_min + y_max)
    ax.plot(t, y_mid, linewidth=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if title is None:
        title = out_path.stem
    ax.set_title(f"{title} | duration={duration_sec:.1f}s | samples={n}")

    # X ticks every minute
    minute_marks = np.arange(0, duration_sec + 0.1, 60)
    ax.set_xticks(minute_marks)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def iter_npy_files(src_dir: Path, recursive: bool = False) -> list[Path]:
    """
    Find .npy files in source directory.
    """
    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(src_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact ECG overview PNGs from .npy files."
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
        help="Directory where PNG files will be saved",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=200,
        help="Sampling frequency in Hz (default: 200)",
    )
    parser.add_argument(
        "--width-px",
        type=int,
        default=1600,
        help="Approximate plot width in pixels / number of envelope bins (default: 1600)",
    )
    parser.add_argument(
        "--height-in",
        type=float,
        default=4.5,
        help="Figure height in inches (default: 4.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI (default: 100)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .npy files recursively",
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

    print(f"Found {len(npy_files)} .npy files")

    ok_count = 0
    fail_count = 0

    for i, npy_path in enumerate(npy_files, start=1):
        try:
            ecg = load_ecg_npy(npy_path)

            out_path = out_dir / f"{npy_path.stem}.png"

            save_ecg_overview_png(
                ecg=ecg,
                out_path=out_path,
                fs=args.fs,
                width_px=args.width_px,
                height_in=args.height_in,
                dpi=args.dpi,
                title=npy_path.stem,
            )

            ok_count += 1
            print(f"[{i}/{len(npy_files)}] OK   {npy_path.name} -> {out_path.name}")

        except Exception as exc:
            fail_count += 1
            print(f"[{i}/{len(npy_files)}] FAIL {npy_path.name}: {exc}")

    print()
    print(f"Done. Saved: {ok_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()