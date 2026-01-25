#!/usr/bin/env python3
"""
CLI wrapper to summarize ECG records using helpers from analyze_ecg_zip_structure_refactoring2.py
Handles data files with .bin / three-digit extensions (size//4 samples) or .npy (np.load(...).size).
Uses only noises_annotated from JSON.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import analyze_ecg_zip_structure_refactoring2 as analyzer


def find_data_file(json_path: Path) -> Tuple[Path, str] | None:
    """Return matching data file and extension for a given JSON (same stem)."""
    stem = json_path.stem
    rec_dir = json_path.parent
    # priority: .npy, .bin, then any three-digit extension
    candidates = [
        rec_dir / f"{stem}.npy",
        rec_dir / f"{stem}.bin",
    ]
    candidates += sorted(
        [p for p in rec_dir.glob(f"{stem}.*") if p.suffix[1:].isdigit() and len(p.suffix) == 4]
    )
    for p in candidates:
        if p.exists():
            return p, p.suffix
    return None


def sample_count(path: Path, ext: str) -> int:
    ext = ext.lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(getattr(arr, "size", 0))
    return int(path.stat().st_size // 4)


def load_json(path: Path):
    return analyzer._safe_json_load_path(path)


def file_size(path: Path) -> int:
    return path.stat().st_size


def infer_sequence_id(json_path: Path) -> str:
    # expected layout: <seq>/recordings/<id>.json
    return json_path.parent.parent.name if json_path.parent.name == "recordings" else json_path.parent.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ECG records in a folder (non-zip).")
    ap.add_argument("data_dir", type=Path, help="Root folder containing sequence subfolders with recordings/")
    ap.add_argument("--fs", type=int, required=True, help="Sampling frequency in Hz")
    ap.add_argument("--out", type=Path, default=None, help="CSV output path (default: <data_dir>/records_summary.csv)")
    args = ap.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    json_files = sorted(data_dir.rglob("*.json"))
    records: list[Dict] = []

    for jp in json_files:
        match = find_data_file(jp)
        if match is None:
            print(f"WARN: no data file found for {jp}")
            continue
        data_path, data_ext = match
        seq_id = infer_sequence_id(jp)
        rec_id = jp.stem
        rec, rec_dict = analyzer.summarize_record(
            sequence_id=seq_id,
            recording_id=rec_id,
            json_ref=jp,
            data_ref=data_path,
            data_ext=data_ext,
            load_json_fn=load_json,
            file_size_fn=file_size,
            sample_count_fn=sample_count,
            fs=args.fs,
        )
        records.append(rec_dict)
        print(rec_dict)

    df = pd.DataFrame(records)
    out_path = args.out or (data_dir / "records_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
