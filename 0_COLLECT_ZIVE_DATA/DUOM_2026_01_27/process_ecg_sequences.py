#!/usr/bin/env python3
"""
process_ecg_sequences.py

Goal
----
Process ECG data sequences for a single patient from either:
  - a .zip file containing multiple sequence folders, or
  - an extracted folder containing multiple sequence folders.

For every sequence folder, the script:
  1) Analyzes the "recordings/" (or "Records/") subfolder (per ~10-minute record JSON + data file).
  2) Generates a per-sequence "records summary" (table like analyze_records_2026_v3.py output),
     with one intentional change: the patient uid/userId is NOT included as a column in the records table.
     Instead, userId is stored in the Excel meta sheet and in the global summary.
  3) Parses sequence-level metadata (sequence-metadata.json, gaps.json, merged-sequence.bin if present),
     and generates a "sequence summary" (single-row table with derived metrics).
  4) Writes both summaries as CSV and XLSX into the corresponding output sequence folder.

After all sequences are processed, the script writes global aggregated summaries (CSV + XLSX)
into the root output folder.

Usage
-----
  python process_ecg_sequences.py path/to/user-XXXX.zip
  python process_ecg_sequences.py path/to/unzipped_patient_folder

Optional:
  --fs 200              Sampling frequency (Hz), default 200
  --out OUTPUT_DIR      Root output folder. If omitted:
                           - for folder input: the input folder itself
                           - for zip input:   <zip_parent>/<zip_stem>_out

Notes
-----
- This script includes and adapts core record-parsing logic from analyze_records_2026_v3.py
  (schema handling for rpeaks, rpeakAnnotationCounts, noises/noises_annotated, etc.).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Logging
# -----------------------------
LOG = logging.getLogger("process_ecg_sequences")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )


# -----------------------------
# Record-level parsing (adapted from analyze_records_2026_v3.py)
# -----------------------------
JsonLikeRef = Union[str, Path]


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

    ml_s_count: int
    ml_v_count: int
    ml_u_count: int

    json_ok: bool

    ml_noises_count: int
    ml_noises_fraction: Optional[float]

    annotated_ml_noises_count: int
    annotated_ml_noises_fraction: Optional[float]

    has_comment: bool
    json_keys_correct: bool


def _flatten_flags(flags: Any) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, list):
        return [str(x) for x in flags]
    return [str(flags)]


def _pick_variant_dict(obj: Any, keys_preference: Tuple[str, ...] = ("merged", "human", "ml")) -> Any:
    """If obj is dict with variants, return preferred variant value."""
    if not isinstance(obj, dict):
        return None
    for k in keys_preference:
        if k in obj:
            return obj.get(k)
    for k, v in obj.items():
        if k == "__info":
            continue
        return v
    return None


def _extract_rpeaks_list(rpeaks: Any) -> List[Dict[str, Any]]:
    """Supports list[dict] OR dict with variants {"merged":[...], ...}."""
    if isinstance(rpeaks, list):
        return [x for x in rpeaks if isinstance(x, dict)]
    if isinstance(rpeaks, dict):
        v = _pick_variant_dict(rpeaks, ("merged", "human", "ml"))
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_ann_counts(ann_counts: Any) -> Dict[str, int]:
    """
    Supports:
      - flat dict: {"N": 10, "S": 2, ...}
      - variant dict: {"merged": {"N":10}, "ml": {...}, ...}
    Returns cleaned mapping for the chosen variant.
    """
    if not isinstance(ann_counts, dict):
        return {}

    # Variant dict?
    if any(isinstance(v, dict) for k, v in ann_counts.items() if k != "__info"):
        v = _pick_variant_dict(ann_counts, ("merged", "human", "ml"))
        if isinstance(v, dict):
            out: Dict[str, int] = {}
            for k, val in v.items():
                if k == "__info":
                    continue
                if isinstance(val, int):
                    out[str(k)] = int(val)
                elif isinstance(val, (float, np.floating)) and float(val).is_integer():
                    out[str(k)] = int(val)
            return out
        return {}

    # Flat dict
    out2: Dict[str, int] = {}
    for k, v in ann_counts.items():
        if k == "__info":
            continue
        if isinstance(v, int):
            out2[str(k)] = int(v)
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out2[str(k)] = int(v)
    return out2


def _extract_rpeaks_list_for_variant(rpeaks: Any, variant: str) -> List[Dict[str, Any]]:
    """Extract rpeaks list for a specific variant (e.g., 'ml')."""
    if isinstance(rpeaks, dict):
        v = rpeaks.get(variant)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_ann_counts_for_variant(ann_counts: Any, variant: str) -> Dict[str, int]:
    """Extract rpeakAnnotationCounts mapping for a specific variant (e.g., 'ml')."""
    if not isinstance(ann_counts, dict):
        return {}
    v = ann_counts.get(variant)
    if not isinstance(v, dict):
        return {}
    out: Dict[str, int] = {}
    for k, val in v.items():
        if k == "__info":
            continue
        if isinstance(val, int):
            out[str(k)] = int(val)
        elif isinstance(val, (float, np.floating)) and float(val).is_integer():
            out[str(k)] = int(val)
    return out


def _summarize_rpeaks(rpeaks: Any) -> Tuple[int, Optional[int], Optional[int], Dict[str, int]]:
    """Return (count, first_idx, last_idx, annotation_counts) for the chosen variant."""
    lst = _extract_rpeaks_list(rpeaks)
    if not lst:
        return 0, None, None, {}

    idxs: List[int] = []
    ann: Dict[str, int] = {}
    for rp in lst:
        si = rp.get("sampleIndex")
        av = rp.get("annotationValue")
        if isinstance(si, int):
            idxs.append(si)
        elif isinstance(si, (float, np.floating)) and float(si).is_integer():
            idxs.append(int(si))

        if isinstance(av, str):
            ann[av] = ann.get(av, 0) + 1
        elif av is not None:
            s = str(av)
            ann[s] = ann.get(s, 0) + 1

    if not idxs:
        return len(lst), None, None, ann

    idxs.sort()
    return len(lst), idxs[0], idxs[-1], ann


def _sum_noise_samples(noises: Any, fs: Optional[int]) -> Tuple[int, int]:
    """
    Returns (interval_count, total_samples_covered).

    Supports list items like:
      - {"startIndex": x, "endIndex": y}
      - {"startSample": x, "endSample": y}
      - {"startTime": t0, "endTime": t1}  (seconds -> samples via fs)
      - [startIndex, endIndex] or (startIndex, endIndex)

    Also supports dict with variants (merged/human/ml) similarly to rpeaks.
    """
    if isinstance(noises, dict):
        v = _pick_variant_dict(noises, ("merged", "human", "ml"))
        noises = v

    if not isinstance(noises, list):
        return 0, 0

    cnt = 0
    total = 0
    for it in noises:
        a = b = None

        if isinstance(it, dict):
            a = it.get("startIndex", it.get("startSample"))
            b = it.get("endIndex", it.get("endSample"))

            if (a is None or b is None) and fs:
                t0 = it.get("startTime")
                t1 = it.get("endTime")
                if (
                    isinstance(t0, (int, float, np.floating))
                    and isinstance(t1, (int, float, np.floating))
                    and t1 > t0
                ):
                    a = int(round(float(t0) * fs))
                    b = int(round(float(t1) * fs))

        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            a, b = it[0], it[1]

        if isinstance(a, (float, np.floating)) and float(a).is_integer():
            a = int(a)
        if isinstance(b, (float, np.floating)) and float(b).is_integer():
            b = int(b)

        if isinstance(a, int) and isinstance(b, int) and b > a:
            cnt += 1
            total += (b - a)

    return cnt, total


def _safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def _file_size(path: Path) -> int:
    return int(path.stat().st_size)


def _sample_count(path: Path, ext: str) -> int:
    ext = (ext or "").lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(getattr(arr, "size", 0))
    # .bin, .ddd, or raw -> int32 samples
    return int(path.stat().st_size // 4)


def find_data_file_fs(json_path: Path) -> Optional[Tuple[Path, str]]:
    """
    Return matching data file and extension for a given JSON (filesystem).

    Supports:
      - <id>.json           -> <id>.npy / <id>.bin / <id>.<ddd> / <id>
      - <id>.<ddd>.json     -> <id>.<ddd>
    """
    stem = json_path.stem
    rec_dir = json_path.parent

    # JSON is '<id>.<ddd>.json' and data is '<id>.<ddd>'
    exact = rec_dir / stem
    if exact.exists() and exact.is_file() and exact.suffix[1:].isdigit() and len(exact.suffix) == 4:
        return exact, exact.suffix

    candidates = [
        rec_dir / f"{stem}.npy",
        rec_dir / f"{stem}.bin",
        rec_dir / f"{stem}",  # raw file with no suffix
    ]
    candidates += sorted([p for p in rec_dir.glob(f"{stem}.*") if p.suffix[1:].isdigit() and len(p.suffix) == 4])

    for p in candidates:
        if p.exists() and p.is_file():
            return p, p.suffix
    return None


def summarize_record(sequenceId: str, file_name: str, json_path: Path, data_path: Path, data_ext: str, fs: int) -> Tuple[RecordSummary, Dict[str, Any]]:
    """Compute per-record statistics (schema-agnostic) and return both dataclass and dict."""
    try:
        meta = _safe_json_load(json_path)
        json_ok = True
    except Exception as exc:
        LOG.warning("Failed to load JSON for %s/%s: %s", sequenceId, file_name, exc)
        meta = {}
        json_ok = False

    bsz = _file_size(data_path)
    samples = _sample_count(data_path, data_ext)

    channel_count = meta.get("channelCount") if isinstance(meta, dict) else None
    user_id = meta.get("userId") if isinstance(meta, dict) else None
    recordingId = meta.get("recordingId") if isinstance(meta, dict) else None
    flags_list = _flatten_flags(meta.get("flags") if isinstance(meta, dict) else None)

    rpeaks_any = meta.get("rpeaks") if isinstance(meta, dict) else None
    rpk_cnt, _rpk_first, _rpk_last, rpk_ann = _summarize_rpeaks(rpeaks_any)

    ann_counts_any = meta.get("rpeakAnnotationCounts") if isinstance(meta, dict) else None
    ann_counts_clean = _extract_ann_counts(ann_counts_any)
    if not ann_counts_clean:
        ann_counts_clean = rpk_ann

    ann_n = int(ann_counts_clean.get("N", 0))
    ann_s = int(ann_counts_clean.get("S", 0))
    ann_v = int(ann_counts_clean.get("V", 0))
    ann_u = int(ann_counts_clean.get("U", 0))

    # ML-only annotation counts
    ml_counts = _extract_ann_counts_for_variant(ann_counts_any, "ml")
    if not ml_counts:
        ml_rpeaks = _extract_rpeaks_list_for_variant(rpeaks_any, "ml")
        tmp: Dict[str, int] = {}
        for rp in ml_rpeaks:
            av = rp.get("annotationValue")
            if isinstance(av, str):
                tmp[av] = tmp.get(av, 0) + 1
            elif av is not None:
                s = str(av)
                tmp[s] = tmp.get(s, 0) + 1
        ml_counts = tmp

    ml_s = int(ml_counts.get("S", 0))
    ml_v = int(ml_counts.get("V", 0))
    ml_u = int(ml_counts.get("U", 0))

    ann_noises = meta.get("noises_annotated") if isinstance(meta, dict) else None
    ann_nz_cnt, ann_nz_samples = _sum_noise_samples(ann_noises, fs)

    noises = meta.get("noises") if isinstance(meta, dict) else None
    ml_nz_cnt, nz_samples = _sum_noise_samples(noises, fs)

    duration_s = (samples / fs) if (fs and fs > 0 and samples > 0) else None
    ml_noises_fraction = (nz_samples / samples) * 100.0 if samples > 0 else None
    annotated_ml_noises_fraction = (ann_nz_samples / samples) * 100.0 if samples > 0 else None

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
        sequenceId=str(sequenceId),
        recordingId=str(recordingId) if isinstance(recordingId, str) else None,
        file_name=str(file_name),
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
        ml_s_count=int(ml_s),
        ml_v_count=int(ml_v),
        ml_u_count=int(ml_u),
        json_ok=bool(json_ok),
        ml_noises_count=int(ml_nz_cnt),
        ml_noises_fraction=float(ml_noises_fraction) if ml_noises_fraction is not None else None,
        annotated_ml_noises_count=int(ann_nz_cnt),
        annotated_ml_noises_fraction=float(annotated_ml_noises_fraction) if annotated_ml_noises_fraction is not None else None,
        has_comment=bool(meta.get("comment")) if isinstance(meta, dict) else False,
        json_keys_correct=bool(json_keys_correct),
    )

    rec_dict = asdict(rec)
    rec_dict["_noises_samples"] = int(nz_samples)
    rec_dict["_annotated_noises_samples"] = int(ann_nz_samples)
    return rec, rec_dict


# -----------------------------
# Time decoding from record filename
# -----------------------------
def decode_unix_from_record_stem(stem: str) -> Optional[_dt.datetime]:
    """
    Example:
      stem = "1761426.035" -> remove dots -> "1761426035" (unix seconds, UTC)

    Returns timezone-aware UTC datetime or None if parsing fails.
    """
    s = stem.replace(".", "").strip()
    if not s.isdigit():
        return None
    try:
        ts = int(s)
        # reasonable sanity check: unix seconds in [2000..2100]
        dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
        if dt.year < 2000 or dt.year > 2100:
            return None
        return dt
    except Exception:
        return None


# -----------------------------
# Sequence-level parsing
# -----------------------------
def parse_gaps_json(gaps_path: Path, fs: int) -> Dict[str, Any]:
    """
    Expected schema example:
      {"gaps": [[startIndex, endIndex], ...]}
    """
    out: Dict[str, Any] = {
        "gaps_json_present": False,
        "gaps_count": 0,
        "gaps_samples": 0,
        "gaps_duration_s": None,
    }
    if not gaps_path.exists():
        return out

    out["gaps_json_present"] = True
    try:
        data = _safe_json_load(gaps_path)
        gaps = data.get("gaps") if isinstance(data, dict) else None
        if not isinstance(gaps, list):
            return out

        total = 0
        cnt = 0
        for it in gaps:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                a, b = it[0], it[1]
                if isinstance(a, int) and isinstance(b, int) and b > a:
                    cnt += 1
                    total += (b - a)
        out["gaps_count"] = cnt
        out["gaps_samples"] = total
        out["gaps_duration_s"] = float(total / fs) if fs > 0 else None
        return out
    except Exception as exc:
        LOG.warning("Failed to parse gaps.json (%s): %s", gaps_path, exc)
        return out


def parse_sequence_metadata(meta_path: Path) -> Dict[str, Any]:
    """
    sequence-metadata.json may include fields like:
      sequenceId, userId, status, startedAt, endedAt, recordingCount, totalGapCount, totalDurationMs, ...
    """
    out: Dict[str, Any] = {"sequence_metadata_present": False}
    if not meta_path.exists():
        return out
    out["sequence_metadata_present"] = True
    try:
        data = _safe_json_load(meta_path)
        if isinstance(data, dict):
            # keep only simple scalars and selected fields
            keep = [
                "sequenceId", "userId", "status", "startedAt", "endedAt",
                "recordingCount", "totalGapCount", "totalDurationMs", "generatedAt",
            ]
            for k in keep:
                if k in data:
                    out[k] = data[k]
        return out
    except Exception as exc:
        LOG.warning("Failed to parse sequence-metadata.json (%s): %s", meta_path, exc)
        return out


def merged_sequence_stats(merged_path: Path, fs: int) -> Dict[str, Any]:
    """
    merged-sequence.bin expected int32 samples.
    """
    out: Dict[str, Any] = {
        "merged_present": False,
        "merged_bytes": 0,
        "merged_samples": None,
        "merged_duration_s": None,
    }
    if not merged_path.exists():
        return out
    out["merged_present"] = True
    try:
        b = _file_size(merged_path)
        out["merged_bytes"] = int(b)
        samples = int(b // 4)  # int32
        out["merged_samples"] = samples
        out["merged_duration_s"] = float(samples / fs) if fs > 0 else None
        return out
    except Exception as exc:
        LOG.warning("Failed to read merged sequence stats (%s): %s", merged_path, exc)
        return out


def parse_extra_sequence_jsons(seq_dir: Path) -> Dict[str, Any]:
    """
    Best-effort parser for additional top-level JSON metadata files in a sequence folder
    (excluding recordings/*.json, gaps.json, sequence-metadata.json).

    It does NOT try to deeply normalize arbitrary schemas; it just records presence and top-level keys.
    """
    exclude = {"gaps.json", "sequence-metadata.json"}
    extra_files: List[str] = []
    ok_cnt = 0
    keys_union: set[str] = set()

    for p in sorted(seq_dir.glob("*.json")):
        if p.name in exclude:
            continue
        extra_files.append(p.name)
        try:
            data = _safe_json_load(p)
            if isinstance(data, dict):
                keys_union |= {str(k) for k in data.keys()}
            ok_cnt += 1
        except Exception as exc:
            LOG.debug("Extra metadata JSON failed to parse (%s): %s", p, exc)

    return {
        "extra_meta_files": ",".join(extra_files) if extra_files else "",
        "extra_meta_count": int(len(extra_files)),
        "extra_meta_json_ok_count": int(ok_cnt),
        "extra_meta_keys": ",".join(sorted(keys_union)) if keys_union else "",
    }


# -----------------------------
# Output writers
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_df_csv_xlsx(
    df: pd.DataFrame,
    csv_path: Path,
    xlsx_path: Path,
    *,
    sheet_name: str = "summary",
    number_formats: Optional[Dict[str, str]] = None,
    extra_sheets: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """
    Writes DataFrame to CSV and XLSX. Optionally applies Excel number formats for named columns
    on the main sheet, and writes additional sheets.
    """
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        if extra_sheets:
            for sname, sdf in extra_sheets.items():
                sdf.to_excel(writer, index=False, sheet_name=sname)

        if number_formats:
            ws = writer.book[sheet_name]
            headers = [cell.value for cell in ws[1]]
            col_idx = {name: i + 1 for i, name in enumerate(headers) if name is not None}

            for col, fmt in number_formats.items():
                if col not in col_idx:
                    continue
                c = col_idx[col]
                for r in range(2, ws.max_row + 1):
                    ws.cell(row=r, column=c).number_format = fmt


def kv_df(d: Dict[str, Any]) -> pd.DataFrame:
    """Key-value dict to 2-column DataFrame (stable order)."""
    rows = [{"key": k, "value": d.get(k)} for k in d.keys()]
    return pd.DataFrame(rows)


# -----------------------------
# Records summary table formatting
# -----------------------------
def build_records_summary_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Produce a compact records summary DataFrame similar to analyze_records_2026_v3.py,
    with renamed columns. user_id is mapped to 'uid' and then removed per requirements.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.copy()
    df.insert(0, "nr", range(1, len(df) + 1))

    cols = [
        "nr", "file_name", "recordingId", "user_id", "samples", "duration_s",
        "rpeaks_count", "ann_n_count", "ann_s_count", "ann_v_count", "ann_u_count",
        "ml_s_count", "ml_v_count", "ml_u_count",
        "annotated_ml_noises_count", "annotated_ml_noises_fraction",
        "ml_noises_count", "ml_noises_fraction",
        "flags",
        "json_ok",
    ]
    cols = [c for c in cols if c in df.columns]
    df_sel = df.loc[:, cols]

    col_short = {
        "recordingId": "rec_id",
        "user_id": "uid",
        "duration_s": "dur_s",
        "rpeaks_count": "rpk_cnt",
        "annotated_ml_noises_count": "ann_nz_cnt",
        "annotated_ml_noises_fraction": "ann_nz_frac",
        "ml_noises_count": "ml_nz_cnt",
        "ml_noises_fraction": "ml_nz_frac",
        "ann_n_count": "annN",
        "ann_s_count": "annS",
        "ann_v_count": "annV",
        "ann_u_count": "annU",
        "ml_s_count": "mlS",
        "ml_v_count": "mlV",
        "ml_u_count": "mlU",
        "json_ok": "json_ok",
    }
    df_print = df_sel.rename(columns=col_short)

    # Remove uid from the records table (requirement).
    if "uid" in df_print.columns:
        df_print = df_print.drop(columns=["uid"])

    return df_print


# -----------------------------
# Per-sequence processing
# -----------------------------
def find_recordings_dir(seq_dir: Path) -> Optional[Path]:
    for cand in (seq_dir / "recordings", seq_dir / "Records", seq_dir / "RECORDS"):
        if cand.exists() and cand.is_dir():
            return cand
    # case-insensitive fallback
    for d in seq_dir.iterdir():
        if d.is_dir() and d.name.lower() in ("recordings", "records"):
            return d
    return None


def analyze_records_dir(records_dir: Path, sequenceId: str, fs: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze all record JSONs in records_dir.
    Returns:
      - records summary DataFrame (compact, uid removed)
      - meta dict (userId, start/end time decoded, counts, etc.)
    """
    json_files = sorted(records_dir.glob("*.json"))
    if not json_files:
        LOG.warning("No JSON records found in %s", records_dir)
        return pd.DataFrame(), {
            "sequenceId": sequenceId,
            "records_dir": str(records_dir),
            "records_json_count": 0,
        }

    records: List[Dict[str, Any]] = []
    user_ids: List[str] = []

    for jp in json_files:
        match = find_data_file_fs(jp)
        if match is None:
            LOG.warning("No data file found for %s", jp)
            continue
        data_path, data_ext = match
        file_name = data_path.name

        _rec, rec_dict = summarize_record(sequenceId=sequenceId, file_name=file_name, json_path=jp, data_path=data_path, data_ext=data_ext, fs=fs)
        records.append(rec_dict)

        uid = rec_dict.get("user_id")
        if isinstance(uid, str) and uid:
            user_ids.append(uid)

    df_print = build_records_summary_table(records)

    # Decode start/end UTC from first/last JSON filename stems
    first_stem = json_files[0].stem
    last_stem = json_files[-1].stem
    start_dt = decode_unix_from_record_stem(first_stem)
    end_dt = decode_unix_from_record_stem(last_stem)

    meta: Dict[str, Any] = {
        "sequenceId": sequenceId,
        "records_dir": str(records_dir),
        "records_json_count": int(len(json_files)),
        "records_analyzed_count": int(len(records)),
        "userId": user_ids[0] if user_ids else None,
        "sequence_start_utc": start_dt.isoformat() if start_dt else None,
        "sequence_end_utc": end_dt.isoformat() if end_dt else None,
        "fs_hz": int(fs),
    }

    # Add some aggregates from the records table (if present)
    if not df_print.empty:
        # dur_s and noise fractions are numeric in raw dict; after rename they might be object.
        # We'll recompute from original records list to be safe.
        dur_s = [r.get("duration_s") for r in records]
        dur_s = [float(x) for x in dur_s if isinstance(x, (int, float)) and x is not None]
        meta["records_total_duration_s"] = float(sum(dur_s)) if dur_s else None

        # annotated noise fraction mean
        ann_frac = [r.get("annotated_ml_noises_fraction") for r in records]
        ann_frac = [float(x) for x in ann_frac if isinstance(x, (int, float)) and x is not None]
        meta["records_mean_ann_noise_frac_pct"] = float(sum(ann_frac) / len(ann_frac)) if ann_frac else None

    return df_print, meta


def process_sequence(seq_dir: Path, out_seq_dir: Path, fs: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Process a single sequence folder.
    Writes:
      - records_summary.csv / records_summary.xlsx
      - sequence_summary.csv / sequence_summary.xlsx

    Returns:
      - flat sequence summary row (dict)
      - records DataFrame (for global aggregation) with sequenceId included
    """
    ensure_dir(out_seq_dir)

    seq_id = seq_dir.name
    rec_dir = find_recordings_dir(seq_dir)
    if rec_dir is None:
        LOG.warning("Skipping %s (no recordings/Records subfolder found)", seq_dir)
        return {"sequenceId": seq_id, "skipped": True}, pd.DataFrame()

    # 1) Record-level analysis
    df_records, rec_meta = analyze_records_dir(rec_dir, sequenceId=seq_id, fs=fs)

    # Write records summary
    records_csv = out_seq_dir / "records_summary.csv"
    records_xlsx = out_seq_dir / "records_summary.xlsx"

    # Ensure numeric for Excel formatting
    df_records_out = df_records.copy()
    for c in ("dur_s", "ann_nz_frac", "ml_nz_frac"):
        if c in df_records_out.columns:
            df_records_out[c] = pd.to_numeric(df_records_out[c], errors="coerce")

    number_formats = {
        "dur_s": "0.00",
        "ann_nz_frac": '0.0"%"',
        "ml_nz_frac": '0.0"%"',
    }

    extra_sheets = {
        "meta": kv_df(rec_meta),
    }

    write_df_csv_xlsx(
        df_records_out,
        records_csv,
        records_xlsx,
        sheet_name="records",
        number_formats=number_formats,
        extra_sheets=extra_sheets,
    )

    # 2) Sequence-level metadata parsing
    seq_meta_path = seq_dir / "sequence-metadata.json"
    gaps_path = seq_dir / "gaps.json"
    merged_path = seq_dir / "merged-sequence.bin"

    seq_meta = parse_sequence_metadata(seq_meta_path)
    gaps_info = parse_gaps_json(gaps_path, fs=fs)
    merged_info = merged_sequence_stats(merged_path, fs=fs)
    extra_info = parse_extra_sequence_jsons(seq_dir)

    # 3) Build single-row sequence summary
    row: Dict[str, Any] = {
        "sequenceId": seq_id,
        "userId": seq_meta.get("userId") or rec_meta.get("userId"),
        "status": seq_meta.get("status"),
        "startedAt_meta": seq_meta.get("startedAt"),
        "endedAt_meta": seq_meta.get("endedAt"),
        "sequence_start_utc_from_filename": rec_meta.get("sequence_start_utc"),
        "sequence_end_utc_from_filename": rec_meta.get("sequence_end_utc"),
        "fs_hz": fs,
        "recordingCount_meta": seq_meta.get("recordingCount"),
        "records_json_count": rec_meta.get("records_json_count"),
        "records_analyzed_count": rec_meta.get("records_analyzed_count"),
        "records_total_duration_s": rec_meta.get("records_total_duration_s"),
        "records_mean_ann_noise_frac_pct": rec_meta.get("records_mean_ann_noise_frac_pct"),
        "totalDurationMs_meta": seq_meta.get("totalDurationMs"),
        "totalDuration_s_meta": (float(seq_meta.get("totalDurationMs")) / 1000.0) if isinstance(seq_meta.get("totalDurationMs"), (int, float)) else None,
        "totalGapCount_meta": seq_meta.get("totalGapCount"),
        "gaps_count_gapsjson": gaps_info.get("gaps_count"),
        "gaps_samples_gapsjson": gaps_info.get("gaps_samples"),
        "gaps_duration_s_gapsjson": gaps_info.get("gaps_duration_s"),
        "merged_present": merged_info.get("merged_present"),
        "merged_bytes": merged_info.get("merged_bytes"),
        "merged_samples": merged_info.get("merged_samples"),
        "merged_duration_s": merged_info.get("merged_duration_s"),
        "sequence_metadata_present": seq_meta.get("sequence_metadata_present", False),
        "extra_meta_count": extra_info.get("extra_meta_count"),
        "extra_meta_json_ok_count": extra_info.get("extra_meta_json_ok_count"),
        "extra_meta_files": extra_info.get("extra_meta_files"),
        "extra_meta_keys": extra_info.get("extra_meta_keys"),
        "gaps_json_present": gaps_info.get("gaps_json_present", False),
    }

    # Derived: gaps fraction vs merged samples
    try:
        ms = merged_info.get("merged_samples")
        gs = gaps_info.get("gaps_samples")
        if isinstance(ms, int) and ms > 0 and isinstance(gs, int) and gs >= 0:
            row["gaps_frac_pct_vs_merged"] = float(gs / ms * 100.0)
        else:
            row["gaps_frac_pct_vs_merged"] = None
    except Exception:
        row["gaps_frac_pct_vs_merged"] = None

    seq_df = pd.DataFrame([row])

    # Write sequence summary
    seq_csv = out_seq_dir / "sequence_summary.csv"
    seq_xlsx = out_seq_dir / "sequence_summary.xlsx"
    write_df_csv_xlsx(
        seq_df,
        seq_csv,
        seq_xlsx,
        sheet_name="sequence",
        number_formats={
            "records_total_duration_s": "0.00",
            "records_mean_ann_noise_frac_pct": '0.0"%"',
            "totalDuration_s_meta": "0.00",
            "gaps_duration_s_gapsjson": "0.00",
            "merged_duration_s": "0.00",
            "gaps_frac_pct_vs_merged": '0.0"%"',
        },
        extra_sheets={"meta": kv_df(row)},
    )

    # For global records aggregation, add sequenceId column
    df_records_global = df_records.copy()
    if not df_records_global.empty:
        df_records_global.insert(0, "sequenceId", seq_id)

    LOG.info("Sequence %s: wrote %s, %s, %s, %s", seq_id, records_csv.name, records_xlsx.name, seq_csv.name, seq_xlsx.name)

    return row, df_records_global


# -----------------------------
# Input handling and sequence discovery
# -----------------------------
def is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def extract_zip_to_temp(zip_path: Path) -> Tuple[Path, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="ecg_sequences_")
    tmp_dir = Path(tmp.name)
    LOG.info("Extracting zip to temp dir: %s", tmp_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    return tmp_dir, tmp


def resolve_output_root(input_path: Path, out_arg: Optional[Path]) -> Path:
    if out_arg:
        return out_arg
    if is_zip_file(input_path):
        return input_path.parent / f"{input_path.stem}_out"
    return input_path


def discover_sequence_dirs(root_dir: Path) -> List[Path]:
    """
    Finds sequence folders by looking for directories that contain a 'recordings' or 'Records' subfolder.
    Returns unique parents of such folders.
    """
    seq_dirs: set[Path] = set()
    for d in root_dir.rglob("*"):
        if not d.is_dir():
            continue
        if d.name.lower() in ("recordings", "records"):
            seq_dirs.add(d.parent)

    # Prefer shallower paths if nested
    seq_list = sorted(seq_dirs, key=lambda p: (len(p.parts), str(p)))
    return seq_list


def map_output_seq_dir(seq_dir: Path, input_root: Path, output_root: Path) -> Path:
    """
    For folder input: writes into the same sequence folder.
    For zip input (temp extracted): writes into output_root/<sequenceId>.
    We detect zip-input scenario by checking whether output_root is not inside input_root.
    """
    # If output_root is the same as input_root, write into seq_dir
    try:
        if output_root.resolve() == input_root.resolve():
            return seq_dir
    except Exception:
        pass
    return output_root / seq_dir.name


# -----------------------------
# Global aggregation
# -----------------------------
def write_global_outputs(
    output_root: Path,
    seq_rows: List[Dict[str, Any]],
    records_all: List[pd.DataFrame],
) -> None:
    ensure_dir(output_root)

    df_seq = pd.DataFrame(seq_rows)
    if not df_seq.empty:
        df_seq = df_seq.sort_values(by=["sequence_start_utc_from_filename", "sequenceId"], na_position="last")

    # Sequences summary
    seq_csv = output_root / "patient_sequences_summary.csv"
    seq_xlsx = output_root / "patient_sequences_summary.xlsx"
    write_df_csv_xlsx(
        df_seq,
        seq_csv,
        seq_xlsx,
        sheet_name="sequences",
        number_formats={
            "records_total_duration_s": "0.00",
            "records_mean_ann_noise_frac_pct": '0.0"%"',
            "totalDuration_s_meta": "0.00",
            "gaps_duration_s_gapsjson": "0.00",
            "merged_duration_s": "0.00",
            "gaps_frac_pct_vs_merged": '0.0"%"',
        },
    )

    # Records summary (optional but useful)
    if records_all:
        df_rec = pd.concat([d for d in records_all if not d.empty], ignore_index=True) if any(not d.empty for d in records_all) else pd.DataFrame()
    else:
        df_rec = pd.DataFrame()

    if not df_rec.empty:
        rec_csv = output_root / "patient_records_summary.csv"
        rec_xlsx = output_root / "patient_records_summary.xlsx"

        df_rec_out = df_rec.copy()
        for c in ("dur_s", "ann_nz_frac", "ml_nz_frac"):
            if c in df_rec_out.columns:
                df_rec_out[c] = pd.to_numeric(df_rec_out[c], errors="coerce")

        write_df_csv_xlsx(
            df_rec_out,
            rec_csv,
            rec_xlsx,
            sheet_name="records",
            number_formats={
                "dur_s": "0.00",
                "ann_nz_frac": '0.0"%"',
                "ml_nz_frac": '0.0"%"',
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Process ECG patient sequences from a zip or folder.")
    ap.add_argument("path", type=Path, help="Path to patient .zip OR extracted patient folder.")
    ap.add_argument("--fs", type=int, default=200, help="Sampling frequency (Hz). Default: 200")
    ap.add_argument("--out", type=Path, default=None, help="Output root directory. Default: <input> (folder) or <zip>_out (zip).")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    setup_logging(args.verbose)

    in_path: Path = args.path
    if not in_path.exists():
        raise SystemExit(f"Input path does not exist: {in_path}")

    output_root = resolve_output_root(in_path, args.out)
    ensure_dir(output_root)

    tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
    try:
        if is_zip_file(in_path):
            input_root, tmp_ctx = extract_zip_to_temp(in_path)
        else:
            input_root = in_path

        seq_dirs = discover_sequence_dirs(input_root)
        if not seq_dirs:
            LOG.error("No sequence folders found under: %s", input_root)
            return 2

        LOG.info("Found %d sequence folder(s).", len(seq_dirs))

        seq_rows: List[Dict[str, Any]] = []
        records_all: List[pd.DataFrame] = []

        for seq_dir in seq_dirs:
            out_seq_dir = map_output_seq_dir(seq_dir, input_root=input_root, output_root=output_root)
            try:
                row, df_rec = process_sequence(seq_dir, out_seq_dir, fs=args.fs)
                seq_rows.append(row)
                if not df_rec.empty:
                    records_all.append(df_rec)
            except Exception as exc:
                LOG.exception("Sequence failed: %s (%s)", seq_dir, exc)
                seq_rows.append({"sequenceId": seq_dir.name, "error": str(exc)})

        write_global_outputs(output_root, seq_rows, records_all)

        LOG.info("Done. Global outputs in: %s", output_root)
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

"""
python process_ecg_sequences.py path/to/user.zip --fs 200
python process_ecg_sequences.py path/to/user.zip --out /some/output/folder

python process_ecg_sequences.py user-6581e16ce2b0bd5f0e7540a4 -v

"""