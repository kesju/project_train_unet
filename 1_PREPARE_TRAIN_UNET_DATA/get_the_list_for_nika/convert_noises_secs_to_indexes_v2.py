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
    if s is None:
        return ""
    s = str(s).strip().strip("'").strip('"')
    return Path(s).name

def parse_noises_txt(txt_path: Path) -> Dict[str, List[Dict[str, float]]]:
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

        while i < len(lines) and lines[i].strip() == "":
            i += 1
        while i < len(lines) and lines[i].lstrip().startswith("#"):
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1

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
            cleaned.append({"startTime": float(it["startTime"]), "endTime": float(it["endTime"])})
        out[fname] = cleaned
    return out

def pick_recording_id_column(df: pd.DataFrame) -> str:
    for c in ("recordingId", "recording_id", "rec_id"):
        if c in df.columns:
            return c
    raise ValueError("records_summary.csv must contain one of: recordingId, recording_id, rec_id")

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

    df = df.copy()
    df["file_name_norm"] = df["file_name"].apply(_norm_name)
    mapping = df.set_index("file_name_norm")[rid_col].to_dict()

    converted_blocks: List[str] = []
    missing: List[str] = []
    total_intervals = 0
    converted_files = 0

    for file_name, intervals in blocks.items():
        fn = _norm_name(file_name)
        rec_id = mapping.get(fn)
        if rec_id is None or str(rec_id).strip() == "" or str(rec_id).lower() == "nan":
            missing.append(fn)
            continue

        converted_files += 1
        out_list = []
        for it in intervals:
            s = float(it["startTime"])
            e = float(it["endTime"])
            out_list.append({"startIndex": int(s * args.fs), "endIndex": int(e * args.fs)})
        total_intervals += len(out_list)

        # ✅ change label from fileName to recordingId
        converted_blocks.append(f"file_name = '{file_name}'")
        converted_blocks.append(f"recordingId = '{rec_id}'")
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
python convert_noises_secs_to_indexes_v2.py \
  --noises_txt List_of_noises_secs_in_zive_records.txt \
  --summary_csv records_summary.csv \
  --out_txt noises_with_ids.txt \
  --fs 200

"""