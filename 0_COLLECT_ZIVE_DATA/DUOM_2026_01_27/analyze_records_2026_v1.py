#!/usr/bin/env python3
"""
CLI wrapper to summarize ECG records using helpers from analyze_ecg_zip_structure_refactoring2.py
Handles data files with .bin / three-digit extensions (size//4 samples) or .npy (np.load(...).size).
Uses only noises_annotated from JSON.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Union, Any, Callable, Optional, List
from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd

JsonLikeRef = Union[str, Path]

# -----------------------------
# Data models
# -----------------------------
@dataclass
class RecordSummary:
    sequenceId: str
    recordingId: Optional[str]
    file_name: str
    channel_count: Optional[int]
    user_id: Optional[str]

    bin_bytes: int
    samples: int
    duration_s: Optional[float]

    flags_count: int
    flags: List[str]

    rpeaks_count: int
    ann_n_count: int
    ann_s_count: int
    ann_v_count: int
    ann_u_count: int

    json_ok: bool

    noises_count: int
    noises_fraction: Optional[float]
    
    annotated_noises_count: int
    annotated_noises_fraction: Optional[float]

    has_comment: bool
    json_keys_correct: bool


def _flatten_flags(flags: Any) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, list):
        return [str(x) for x in flags]
    # unexpected type
    return [str(flags)]

def _summarize_rpeaks(rpeaks: Any) -> Tuple[int, Optional[int], Optional[int], Dict[str, int]]:
    """
    rpeaks: list of {"sampleIndex": int, "annotationValue": str}
    Returns (count, first_idx, last_idx, annotation_counts)
    """
    if not isinstance(rpeaks, list) or len(rpeaks) == 0:
        return 0, None, None, {}
    idxs: List[int] = []
    ann: Dict[str, int] = {}
    for rp in rpeaks:
        if not isinstance(rp, dict):
            continue
        si = rp.get("sampleIndex")
        av = rp.get("annotationValue")
        if isinstance(si, int):
            idxs.append(si)
        if isinstance(av, str):
            ann[av] = ann.get(av, 0) + 1
        elif av is not None:
            ann[str(av)] = ann.get(str(av), 0) + 1
    if not idxs:
        return 0, None, None, ann
    idxs.sort()
    return len(idxs), idxs[0], idxs[-1], ann

def _sum_noise_samples(noises: Any) -> Tuple[int, int]:
    """
    Returns (interval_count, total_samples_covered) for noises list.
    Accepts items like {"startIndex": x, "endIndex": y}.
    """
    if not isinstance(noises, list):
        return 0, 0
    cnt = 0
    total = 0
    for it in noises:
        if not isinstance(it, dict):
            continue
        a = it.get("startIndex")
        b = it.get("endIndex")
        if isinstance(a, int) and isinstance(b, int) and b > a:
            cnt += 1
            total += (b - a)
    return cnt, total


def _safe_json_load_path(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_record(
    sequenceId: str,
    # recordingId: str,
    file_name: str,
    json_ref: JsonLikeRef,
    data_ref: JsonLikeRef,
    data_ext: str,
    *,
    load_json_fn: Callable[[JsonLikeRef], Any],
    file_size_fn: Callable[[JsonLikeRef], int],
    sample_count_fn: Callable[[JsonLikeRef, str], int],
    fs: int,
) -> Tuple[RecordSummary, Dict[str, Any]]:
    """
    Compute per-record statistics and return both RecordSummary and its dict form.

    This helper is designed to be reusable from other scripts: provide your own
    load_json_fn / file_size_fn / sample_count_fn depending on whether data lives
    in a ZIP or filesystem and what the data extension is.
    """
    try:
        meta = load_json_fn(json_ref)
        json_ok = True
    except Exception as exc:
        print(f"WARN: failed to load JSON for {sequenceId}/{file_name}: {exc}")
        meta = {}
        json_ok = False

    bsz = file_size_fn(data_ref)
    samples = sample_count_fn(data_ref, data_ext)
    # print("meta:", meta)
    channel_count = meta.get("channelCount") if isinstance(meta, dict) else None
    user_id = meta.get("userId") if isinstance(meta, dict) else None
    recordingId = meta.get("recordingId") if isinstance(meta, dict) else None
    flags_list = _flatten_flags(meta.get("flags") if isinstance(meta, dict) else None)

    rpeaks = meta.get("rpeaks") if isinstance(meta, dict) else None
    rpk_cnt, _rpk_first, _rpk_last, rpk_ann = _summarize_rpeaks(rpeaks)

    ann_counts = meta.get("rpeakAnnotationCounts") if isinstance(meta, dict) else None
    ann_counts_clean: Dict[str, int] = {}
    if isinstance(ann_counts, dict):
        for k, v in ann_counts.items():
            if k == "__info":
                continue
            if isinstance(v, int):
                ann_counts_clean[str(k)] = v
    else:
        ann_counts_clean = rpk_ann

    ann_n = int(ann_counts_clean.get("N", 0))
    ann_s = int(ann_counts_clean.get("S", 0))
    ann_v = int(ann_counts_clean.get("V", 0))
    ann_u = int(ann_counts_clean.get("U", 0))

    ann_noises = meta.get("noises_annotated") if isinstance(meta, dict) else None
    ann_nz_cnt, ann_nz_samples = _sum_noise_samples(ann_noises)
    noises = meta.get("noises") if isinstance(meta, dict) else None
    nz_cnt, nz_samples = _sum_noise_samples(noises)

    duration_s = (samples / fs) if (fs is not None and fs > 0 and samples > 0) else None
    noises_fraction = (nz_samples / samples) if (samples > 0) else None
    annotated_noises_fraction = (ann_nz_samples / samples) if (samples > 0) else None

    allowed_keys = {
        "channelCount",
        "comment",
        "flags",
        "noises",
        "noises_annotated",
        "recordingId",
        "rpeakAnnotationCounts",
        "rpeaks",
        "userId",
    }
    meta_keys = set(meta.keys()) if isinstance(meta, dict) else set()
    json_keys_correct = meta_keys.issubset(allowed_keys) if meta_keys else False

    rec = RecordSummary(
        sequenceId=sequenceId,
        recordingId=recordingId,
        file_name=Path(data_ref).name,
        channel_count=int(channel_count) if isinstance(channel_count, int) else None,
        user_id=str(user_id) if isinstance(user_id, str) else None,
        bin_bytes=int(bsz),
        samples=int(samples),
        duration_s=float(duration_s) if duration_s is not None else None,
        flags_count=len(flags_list),
        flags=flags_list,
        rpeaks_count=int(rpk_cnt),
        ann_n_count=ann_n,
        ann_s_count=ann_s,
        ann_v_count=ann_v,
        ann_u_count=ann_u,
        json_ok=json_ok,
        noises_count=int(nz_cnt),
        noises_fraction=float(noises_fraction) if noises_fraction is not None else None,
        annotated_noises_count=int(ann_nz_cnt),
        annotated_noises_fraction=float(annotated_noises_fraction) if annotated_noises_fraction is not None else None,
        has_comment=bool(meta.get("comment")) if isinstance(meta, dict) else False,
        json_keys_correct=json_keys_correct,
    )
    rec_dict = asdict(rec)
    rec_dict["_annotated_noises_samples"] = nz_samples  # helper key for higher-level aggregation
    return rec, rec_dict



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
    return _safe_json_load_path(path)


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
        print(f"Processing JSON: {jp}")
        match = find_data_file(jp)
        if match is None:
            print(f"WARN: no data file found for {jp}")
            continue
        data_path, data_ext = match
        seq_id = infer_sequence_id(jp)
        rec_id = jp.stem
        rec, rec_dict = summarize_record(
            sequenceId=seq_id,
            # recordingId=rec_id,
            file_name=match[0].name,
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

    # 1) Add row number (nr)
    df = df.copy()
    df.insert(0, "nr", range(1, len(df) + 1))

    # 2) Select only the columns you want (skip any missing ones safely)
    cols = [
        "nr", "file_name", "recordingId", "user_id", "samples", "duration_s",
        "rpeaks_count", "ann_n_count", "ann_s_count", "ann_v_count", "ann_u_count",
        "annotated_noises_count", "annotated_noises_fraction", "noises_count", "noises_fraction",
    ]
    cols = [c for c in cols if c in df.columns]

    # 3) Print as table: one record per row, header once
    df_sel = df.loc[:, cols]
    
    out_path = args.out or (data_dir / "records_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} records to {out_path}")
    print(f"Data directory: {data_dir}")
    print("Summary statistics:")

    print(
        df_sel.to_string(
            index=False,
            float_format=None,
            formatters={
                "duration_s": lambda x: f"{x:.2f}" if pd.notna(x) else "",
                "annotated_noises_fraction": lambda x: f"{x:.4f}" if pd.notna(x) else "",
            }
        )
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python3 analyze_records.py user-65955b5f50e02b125d4998ad/659ec124870b3d1d1630be39/recordings --fs 200 --out records_summary.csv
# python3 analyze_records.py 659ebcdd870b3d1d6e30bb61 --fs 200 --out records_summary.csv
# python3 analyze_records.py /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_FOR_TRAINING --fs 200 --out records_summary.csv