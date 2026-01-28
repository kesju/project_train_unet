#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_FN_RE = re.compile(r"fileName\s*=\s*['\"]([^'\"]+)['\"]")


def _norm_name(s: Any) -> str:
    """Normalize a file name/path to just the filename (no directories), stripped of quotes/spaces."""
    if s is None:
        return ""
    s = str(s).strip().strip("'").strip('"')
    return Path(s).name


def _is_missing(v: Any) -> bool:
    """True if value is None/empty/'nan' (case-insensitive)."""
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan" or s.lower() == "none"


def parse_noises_txt(txt_path: Path) -> Dict[str, List[Dict[str, float]]]:
    """
    Parses blocks like:
      fileName = "1000_1.npy"
      [
        {"startTime": 1.23, "endTime": 4.56},
        ...
      ]
    Returns { "<filename>": [{"startTime":..., "endTime":...}, ...], ... }
    """
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

        # skip blanks and comment blocks
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        while i < len(lines) and lines[i].lstrip().startswith("#"):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

        # capture JSON list (may span multiple lines)
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

        intervals = json.loads("\n".join(buf).strip())
        if not isinstance(intervals, list):
            raise ValueError(f"Intervals for {fname} must be a JSON list")

        cleaned: List[Dict[str, float]] = []
        for it in intervals:
            if not isinstance(it, dict) or "startTime" not in it or "endTime" not in it:
                raise ValueError(f"Bad interval item for {fname}: {it!r}")
            cleaned.append(
                {"startTime": float(it["startTime"]), "endTime": float(it["endTime"])}
            )

        out[fname] = cleaned

    return out


def pick_recording_id_column(df: pd.DataFrame) -> str:
    for c in ("recordingId", "recording_id", "rec_id"):
        if c in df.columns:
            return c
    raise ValueError("records_summary.csv must contain one of: recordingId, recording_id, rec_id")


def pick_basename_column(df: pd.DataFrame) -> str:
    for c in ("basename", "base_name"):
        if c in df.columns:
            return c
    raise ValueError("records_summary.csv must contain column 'basename' (or 'base_name')")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noises_txt", type=Path, required=True)
    ap.add_argument("--summary_csv", type=Path, required=True)
    ap.add_argument("--out_txt", type=Path, required=True)
    ap.add_argument("--fs", type=float, default=200.0)
    args = ap.parse_args()

    blocks = parse_noises_txt(args.noises_txt)

    df = pd.read_csv(args.summary_csv, dtype=str)
    if "file_name" not in df.columns:
        raise ValueError(f"records_summary.csv must contain 'file_name'. Found: {list(df.columns)}")

    rid_col = pick_recording_id_column(df)
    bname_col = pick_basename_column(df)

    df = df.copy()
    df["file_name_norm"] = df["file_name"].apply(_norm_name)

    # If duplicates exist, keep the last row for that file_name_norm
    df_keyed = df.drop_duplicates(subset=["file_name_norm"], keep="last")

    recid_map = df_keyed.set_index("file_name_norm")[rid_col].to_dict()
    bname_map = df_keyed.set_index("file_name_norm")[bname_col].to_dict()

    converted_blocks: List[str] = []
    missing: List[str] = []
    total_intervals = 0
    converted_files = 0

    for file_name, intervals in blocks.items():
        fn = _norm_name(file_name)

        rec_id = recid_map.get(fn)
        bname = bname_map.get(fn)

        # require recordingId match; basename is added if available
        if _is_missing(rec_id):
            missing.append(fn)
            continue

        converted_files += 1

        out_list = []
        for it in intervals:
            s = float(it["startTime"])
            e = float(it["endTime"])
            out_list.append({"startIndex": int(s * args.fs), "endIndex": int(e * args.fs)})

        total_intervals += len(out_list)

        converted_blocks.append(f"file_name = '{file_name}'")
        converted_blocks.append(f"basename = '{'' if _is_missing(bname) else str(bname).strip()}'")
        converted_blocks.append(f"recordingId = '{str(rec_id).strip()}'")
        converted_blocks.append(json.dumps(out_list, ensure_ascii=False, indent=4))
        converted_blocks.append("")

    args.out_txt.write_text("\n".join(converted_blocks).rstrip() + "\n", encoding="utf-8")

    print(f"Parsed files in txt: {len(blocks)}")
    print(f"Converted files (matched in summary): {converted_files}")
    print(f"Total intervals converted: {total_intervals}")
    print(f"Missing files (not found in summary): {len(missing)}")
    if missing:
        print("First 10 missing:", missing[:10])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Example:

python convert_noises_secs_to_indexes_v3.py \
  --noises_txt List_of_noises_secs_in_zive_records.txt \
  --summary_csv records_summary.csv \
  --out_txt noises_with_ids.txt \
  --fs 200
"""
