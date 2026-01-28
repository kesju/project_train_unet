#!/usr/bin/env python3
"""Collect noise intervals from a text file and join with IDs from records_summary.csv.

What it does
- Parses List_of_noises_secs_in_zive_records.txt blocks:
    fileName = "XXXX.npy"
    [ {"startTime": ..., "endTime": ...}, ... ]
- Looks up each file_name in records_summary.csv to attach:
    uid, rec_id, ann_nz_cnt
- Validates: len(intervals) == ann_nz_cnt for matched files (adds columns)

Outputs
- Flat: one row per interval (file_name, uid, rec_id, startTime, endTime, ann_nz_cnt, intervals_cnt, cnt_match, cnt_diff)
- Grouped: one row per file with 'intervals' as JSON list + the same count columns
- Mismatch report (if any) and missing files list

Example
  python collect_noises_with_ids_v3.py \
      --noises_txt List_of_noises_secs_in_zive_records.txt \
      --summary_csv records_summary.csv \
      --out_prefix noises_with_ids
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_FN_RE = re.compile(r"fileName\s*=\s*['\"]([^'\"]+)['\"]")


def _norm_name(x: Any) -> str:
    """Normalize a file reference to just its basename."""
    if x is None:
        return ""
    s = str(x).strip().strip("'").strip('"')
    return Path(s).name


def parse_noises_txt(txt_path: Path) -> Dict[str, List[Dict[str, float]]]:
    """Parse custom text file into {file_name: [{startTime,endTime}, ...], ...}."""
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: Dict[str, List[Dict[str, float]]] = {}

    i = 0
    while i < len(lines):
        m = _FN_RE.search(lines[i])
        if not m:
            i += 1
            continue

        fname = _norm_name(m.group(1))
        i += 1

        # Skip blank and comment lines until the JSON list begins.
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        while i < len(lines) and lines[i].lstrip().startswith("#"):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

        # Accumulate JSON list by balancing square brackets.
        buf: List[str] = []
        depth = 0
        started = False

        while i < len(lines):
            line = lines[i]
            if not started:
                if "[" in line:
                    started = True
                    depth += line.count("[") - line.count("]")
                    buf.append(line)
                    if depth <= 0:
                        i += 1
                        break
            else:
                depth += line.count("[") - line.count("]")
                buf.append(line)
                if depth <= 0:
                    i += 1
                    break
            i += 1

        if not buf:
            raise ValueError(f"No interval list found after fileName={fname}")

        list_str = "\n".join(buf).strip()
        try:
            intervals = json.loads(list_str)
        except Exception as exc:
            raise ValueError(f"Failed parsing intervals for {fname}: {exc}\n---\n{list_str}") from exc

        if not isinstance(intervals, list):
            raise ValueError(f"Intervals for {fname} must be a JSON list, got: {type(intervals)}")

        cleaned: List[Dict[str, float]] = []
        for it in intervals:
            if not isinstance(it, dict) or "startTime" not in it or "endTime" not in it:
                raise ValueError(f"Bad interval item for {fname}: {it!r}")
            cleaned.append({"startTime": float(it["startTime"]), "endTime": float(it["endTime"])})

        out[fname] = cleaned

    return out


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        # tolerate "1.0" etc.
        return int(float(s))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noises_txt", type=Path, required=True)
    ap.add_argument("--summary_csv", type=Path, required=True)
    ap.add_argument("--out_prefix", type=Path, default=Path("noises_with_ids"))

    args = ap.parse_args()

    blocks = parse_noises_txt(args.noises_txt)

    df = pd.read_csv(args.summary_csv, dtype=str).copy()
    required_cols = ["file_name", "rec_id", "uid", "ann_nz_cnt"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"records_summary.csv is missing required columns: {missing_cols}. Found: {list(df.columns)}"
        )

    df["file_name_norm"] = df["file_name"].apply(_norm_name)

    # Handle duplicates in summary by keeping first, but report.
    dup_mask = df["file_name_norm"].duplicated(keep=False)
    dup_files = sorted(df.loc[dup_mask, "file_name_norm"].unique().tolist())
    if dup_files:
        print(f"WARNING: records_summary.csv has duplicate file_name rows for {len(dup_files)} files. Keeping first.")

    df_first = df.drop_duplicates(subset=["file_name_norm"], keep="first").set_index("file_name_norm")
    lookup = df_first.to_dict(orient="index")

    flat_rows: List[Dict[str, Any]] = []
    grouped_rows: List[Dict[str, Any]] = []
    missing_files: List[str] = []
    mismatch_rows: List[Dict[str, Any]] = []

    for file_name, intervals in blocks.items():
        key = _norm_name(file_name)
        row = lookup.get(key)

        if row is None:
            missing_files.append(key)
            continue

        ann_nz_cnt = _to_int_or_none(row.get("ann_nz_cnt"))
        intervals_cnt = len(intervals)
        cnt_match = (ann_nz_cnt is not None and intervals_cnt == ann_nz_cnt)
        cnt_diff = (intervals_cnt - ann_nz_cnt) if ann_nz_cnt is not None else None

        # record mismatch (per file) if any
        if ann_nz_cnt is None or not cnt_match:
            mismatch_rows.append(
                {
                    "file_name": key,
                    "uid": row.get("uid"),
                    "rec_id": row.get("rec_id"),
                    "intervals_cnt": intervals_cnt,
                    "ann_nz_cnt": ann_nz_cnt,
                    "cnt_match": bool(cnt_match) if ann_nz_cnt is not None else False,
                    "cnt_diff": cnt_diff,
                }
            )

        # grouped output
        grouped_rows.append(
            {
                "file_name": key,
                "uid": row.get("uid"),
                "rec_id": row.get("rec_id"),
                "ann_nz_cnt": ann_nz_cnt,
                "intervals_cnt": intervals_cnt,
                "cnt_match": bool(cnt_match) if ann_nz_cnt is not None else False,
                "cnt_diff": cnt_diff,
                "intervals": intervals,  # list of dicts
            }
        )

        # flat output: ✅ ALL elements in intervals list
        for it in intervals:
            flat_rows.append(
                {
                    "file_name": key,
                    "uid": row.get("uid"),
                    "rec_id": row.get("rec_id"),
                    "startTime": float(it["startTime"]),
                    "endTime": float(it["endTime"]),
                    "ann_nz_cnt": ann_nz_cnt,
                    "intervals_cnt": intervals_cnt,
                    "cnt_match": bool(cnt_match) if ann_nz_cnt is not None else False,
                    "cnt_diff": cnt_diff,
                }
            )

    # Write outputs
    out_prefix = args.out_prefix

    df_flat = pd.DataFrame(flat_rows)
    if not df_flat.empty:
        df_flat = df_flat.sort_values(["file_name", "startTime"], kind="stable")

    df_grouped = pd.DataFrame(grouped_rows)
    if not df_grouped.empty:
        df_grouped = df_grouped.sort_values(["file_name"], kind="stable")

    df_mismatch = pd.DataFrame(mismatch_rows)
    if not df_mismatch.empty and "file_name" in df_mismatch.columns:
        df_mismatch = df_mismatch.sort_values(["file_name"], kind="stable")

    out_flat_csv = out_prefix.with_name(out_prefix.name + "_flat.csv")
    out_flat_json = out_prefix.with_name(out_prefix.name + "_flat.json")
    out_grouped_csv = out_prefix.with_name(out_prefix.name + "_grouped.csv")
    out_grouped_json = out_prefix.with_name(out_prefix.name + "_grouped.json")
    out_mismatch_csv = out_prefix.with_name(out_prefix.name + "_mismatches.csv")
    out_missing_txt = out_prefix.with_name(out_prefix.name + "_missing_files.txt")

    df_flat.to_csv(out_flat_csv, index=False)

    out_flat_json.write_text(json.dumps(flat_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # grouped CSV stores intervals as JSON string in one cell
    df_grouped_csv = df_grouped.copy()
    if not df_grouped_csv.empty:
        df_grouped_csv["intervals"] = df_grouped_csv["intervals"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df_grouped_csv.to_csv(out_grouped_csv, index=False)

    out_grouped_json.write_text(json.dumps(grouped_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    df_mismatch.to_csv(out_mismatch_csv, index=False)

    out_missing_txt.write_text("\n".join(missing_files) + ("\n" if missing_files else ""), encoding="utf-8")

    # Summary
    parsed_files = len(blocks)
    matched_files = df_grouped["file_name"].nunique() if not df_grouped.empty else 0
    total_intervals = len(flat_rows)
    mismatches = len(df_mismatch)

    print(f"Parsed files in txt: {parsed_files}")
    print(f"Matched files in summary: {matched_files}")
    print(f"Total intervals exported: {total_intervals}")
    print(f"Missing files: {len(missing_files)} -> {out_missing_txt}")
    print(f"Count mismatches: {mismatches} -> {out_mismatch_csv}")
    if dup_files:
        print(f"Duplicate summary rows (kept first) for files: {dup_files[:10]}{'...' if len(dup_files)>10 else ''}")
    print(f"Wrote: {out_flat_csv}")
    print(f"Wrote: {out_grouped_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python collect_noises_with_ids_v3.py \
  --noises_txt List_of_noises_secs_in_zive_records.txt \
  --summary_csv records_summary.csv \
  --out_prefix noises_with_ids

"""