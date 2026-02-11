#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------
# I/O helpers
# ----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return obj


def zive_read_file_1ch(filename):
    with open(filename, 'rb') as f:  # Use 'rb' to read binary file
        a = np.fromfile(f, dtype=np.dtype('>i4'))  # Read file content as big-endian 4-byte integers
    
    ADCmax = 0x800000
    Vref = 2.5
    b = (a - ADCmax / 2) * 2 * Vref / ADCmax / 3.5 * 1000  # Corrected the calculation by adding multiplication symbol
    ecg_signal = b - np.mean(b)
    
    return ecg_signal


def load_int16_bin(path: Path, endian: str) -> np.ndarray:
    """
    endian: 'be' or 'le'
    """
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"BIN size not divisible by 2 (expected int16): {path}")
    dt = np.dtype(">i2") if endian == "be" else np.dtype("<i2")
    sig = np.frombuffer(raw, dtype=dt)
    return sig.astype(np.int16, copy=False)


# ----------------------------
# Metadata normalization
# ----------------------------
def normalize_rpeaks(meta: Dict[str, Any]) -> List[Tuple[int, str]]:
    """
    Returns list of (sampleIndex, annotationValue) sorted by sampleIndex.
    Expects: meta["rpeaks"] = [{"sampleIndex": int, "annotationValue": "N"}, ...]
    """
    out: List[Tuple[int, str]] = []
    rpeaks = meta.get("rpeaks")
    if not isinstance(rpeaks, list):
        return out
    for rp in rpeaks:
        if not isinstance(rp, dict):
            continue
        si = rp.get("sampleIndex")
        av = rp.get("annotationValue", "N")
        if isinstance(si, int):
            out.append((si, str(av)))
    out.sort(key=lambda x: x[0])
    return out


def normalize_noise_intervals(meta: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Prefers meta["noises_annotated"] if present and a list; otherwise meta["noises"].

    Accepts:
      - [{"startIndex": int, "endIndex": int}, ...]
      - or [[start, end], ...]
    Returns list of (startIndex, endIndex) with end > start.
    """
    items = meta.get("noises_annotated")
    if not isinstance(items, list):
        items = meta.get("noises")

    out: List[Tuple[int, int]] = []
    if not isinstance(items, list):
        return out

    for it in items:
        a = b = None
        if isinstance(it, dict):
            a = it.get("startIndex")
            b = it.get("endIndex")
        elif isinstance(it, list) and len(it) == 2:
            a, b = it[0], it[1]

        if isinstance(a, int) and isinstance(b, int) and b > a:
            out.append((a, b))

    out.sort(key=lambda x: x[0])
    return out


def slice_items_in_range(
    rpeaks: List[Tuple[int, str]],
    noises: List[Tuple[int, int]],
    start: int,
    end: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, int]]]:
    rp = [(i, a) for (i, a) in rpeaks if start <= i < end]
    nz: List[Tuple[int, int]] = []
    for (a, b) in noises:
        if b <= start or a >= end:
            continue
        nz.append((max(a, start), min(b, end)))
    return rp, nz


# ----------------------------
# Plotting
# ----------------------------
def plot_record_chunks(
    sig: np.ndarray,
    rpeaks: List[Tuple[int, str]],
    noises: List[Tuple[int, int]],
    fs: Optional[float],
    xmode: str,
    chunk_samples: int,
    title_prefix: str,
    show_unknown_ann: bool,
) -> None:
    allowed = {"N", "S", "V", "U"}
    marker_map = {"N": "o", "S": "^", "V": "s", "U": "x"}

    n = sig.size
    num_chunks = math.ceil(n / chunk_samples)

    for k in range(num_chunks):
        start = k * chunk_samples
        end = min((k + 1) * chunk_samples, n)

        seg = sig[start:end]

        if xmode == "time":
            if fs is None or fs <= 0:
                raise ValueError("xmode=time requires --fs")
            x = (np.arange(start, end) / fs).astype(float)
            xlab = "Time (s)"
        else:
            x = np.arange(start, end)
            xlab = "Sample index"

        rp_seg, nz_seg = slice_items_in_range(rpeaks, noises, start, end)

        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(111)

        ax.plot(x, seg, linewidth=0.8)

        # # Shade noise intervals
        # for (a, b) in nz_seg:
        #     xa = (a / fs) if xmode == "time" else a
        #     xb = (b / fs) if xmode == "time" else b
        #     ax.axvspan(xa, xb, alpha=0.2)

        # Group rpeaks by annotation for clean legend
        by_ann: Dict[str, List[int]] = {}
        for (i, ann) in rp_seg:
            ann = str(ann)
            if (ann not in allowed) and (not show_unknown_ann):
                continue
            by_ann.setdefault(ann, []).append(i)

        for ann in sorted(by_ann.keys()):
            idxs = by_ann[ann]
            xs = [(i / fs) if xmode == "time" else i for i in idxs]
            ys = [sig[i] for i in idxs]  # use global signal for exact amplitude
            ax.scatter(xs, ys, marker=marker_map.get(ann, "."), s=25, label=f"Rpeaks {ann}")

        # Ticks requirement
        if xmode == "time":
            ax.xaxis.set_major_locator(MultipleLocator(60))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(100))

        ax.set_xlabel(xlab)
        ax.set_ylabel("ECG (int16 units)")
        ax.set_title(f"{title_prefix} | chunk {k+1}/{num_chunks} | samples {start}:{end}")

        if by_ann:
            ax.legend(loc="upper right", ncols=4, fontsize=9)

        ax.grid(True, linewidth=0.3, alpha=0.4)
        plt.tight_layout()
        plt.show()


# ----------------------------
# Main
# ----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Plot one ECG record from an unzipped folder with rpeaks (N,S,V,U) and noise intervals."
    )
    p.add_argument("root_dir", type=Path, help="Root folder containing sequence subfolders")
    p.add_argument("--sequence-id", required=True, help="Sequence folder name (e.g. SEQ_1)")
    p.add_argument("--recording-id", required=True, help="Recording id (stem of .bin/.json, e.g. 1001_2)")
    p.add_argument("--fs", type=float, default=None, help="Sampling frequency Hz (required for xmode=time)")
    p.add_argument("--xmode", choices=["index", "time"], default="index", help="X axis in indexes or time (s)")
    p.add_argument("--endian", choices=["be", "le"], default="be", help="Binary endianness for int16 (default: be)")
    p.add_argument(
        "--chunk-samples",
        type=int,
        default=None,
        help="Chunk size in samples (separate window per chunk). Overrides --chunk-seconds.",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=300.0,
        help="Chunk size in seconds if --chunk-samples not given. Default 300s (5 min).",
    )
    p.add_argument(
        "--show-unknown-ann",
        action="store_true",
        help="Also show rpeaks with annotations other than N,S,V,U.",
    )
    args = p.parse_args(argv)

    seq_dir = args.root_dir / args.sequence_id
    rec_dir = seq_dir / "recordings"

    json_path = rec_dir / f"{args.recording_id}.json"
    bin_path = rec_dir / f"{args.recording_id}.bin"

    if not json_path.exists():
        print(f"ERROR: Missing JSON: {json_path}", file=sys.stderr)
        return 1
    if not bin_path.exists():
        print(f"ERROR: Missing BIN: {bin_path}", file=sys.stderr)
        return 1

    meta = load_json(json_path)
    sig = zive_read_file_1ch(bin_path)

    rpeaks = normalize_rpeaks(meta)
    noises = normalize_noise_intervals(meta)

    # Determine chunk_samples
    if args.chunk_samples is not None:
        chunk_samples = args.chunk_samples
    else:
        if args.fs is not None and args.fs > 0:
            chunk_samples = int(round(args.chunk_seconds * args.fs))
        else:
            # fallback for index mode without fs
            chunk_samples = 60000

    chunk_samples = max(1000, int(chunk_samples))

    title_prefix = f"Seq={args.sequence_id} | Rec={args.recording_id} | N={sig.size:,} samples"
    if args.fs:
        title_prefix += f" | fs={args.fs:g} Hz"

    plot_record_chunks(
        sig=sig,
        rpeaks=rpeaks,
        noises=noises,
        fs=args.fs,
        xmode=args.xmode,
        chunk_samples=chunk_samples,
        title_prefix=title_prefix,
        show_unknown_ann=args.show_unknown_ann,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example usage:
conda activate ITP259
How to run (examples)
A) X axis = indexes, ticks every 100, split into 60,000-sample windows
python3 plot_record_from_folder.py /data/ecg_patient 
  --sequence-id SEQ_1 --recording-id 1001_2 \
  --xmode index --chunk-samples 60000

A) X axis = indexes, ticks every 100, split into 60,000-sample windows
python3 plot_record_from_folder.py user-65955b5f50e02b125d4998ad \
  --sequence-id 659ec124870b3d1d1630be39 --recording-id 659ebcdd870b3d1d6e30bb61 \
  --xmode index --fs 200 --chunk-samples 60000


B) X axis = time, ticks every 60 s, split into 5-minute windows
python3 plot_record_from_folder.py user-65955b5f50e02b125d4998ad \
  --sequence-id 659ec124870b3d1d1630be39 --recording-id 659ebcdd870b3d1d6e30bb61 \
  --xmode time --fs 200 --chunk-seconds 20

C) If your .bin is little-endian
python3 plot_record_from_folder.py /data/ecg_patient \
  --sequence-id SEQ_1 --recording-id 1001_2 \
  --xmode time --fs 256 --chunk-seconds 300 --endian le

"""