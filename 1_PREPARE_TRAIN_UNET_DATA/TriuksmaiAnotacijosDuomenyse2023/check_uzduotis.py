#!/usr/bin/env python3
from __future__ import annotations

# https://chatgpt.com/c/697c7369-a60c-8396-a0c5-e72d46a7d44e

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RE_FOR_NOISE = re.compile(r"^\s*for_noise_annotation\s*:\s*$")
RE_FILE = re.compile(r"^\s*file_name\s*=\s*['\"]([^'\"]+)['\"]\s*$")
RE_BASE = re.compile(r"^\s*basename\s*=\s*['\"]?([^'\"]+)['\"]?\s*$")
RE_RECID = re.compile(r"^\s*recordingId\s*=\s*['\"]([^'\"]+)['\"]\s*$")


def normalize_basename_str(s: Any) -> str:
    """Normalize numeric-looking strings without rounding (keeps exact digits, removes trailing zeros)."""
    if s is None:
        return ""
    s = str(s).strip().strip(",").strip().strip('"').strip("'")
    # If it looks like a number, normalize trailing zeros in decimal part
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def parse_for_noise_annotation(lines: List[str]) -> List[str]:
    """
    Parse the block:
      for_noise_annotation:
      [
      1631280.395
      ...
      ]
    NOTE: numbers may be listed one-per-line, without commas.
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        if RE_FOR_NOISE.match(lines[i]):
            # seek the opening '['
            i += 1
            while i < len(lines) and "[" not in lines[i]:
                i += 1
            # now parse until ']'
            i += 1
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("]"):
                    break
                if t and t not in ("[", "]"):
                    out.append(normalize_basename_str(t))
                i += 1
            break
        i += 1
    return out


def _capture_json_list(lines: List[str], start_idx: int) -> Tuple[Optional[str], int]:
    """
    Capture a JSON list starting at a line that begins with '[' until bracket depth returns to 0.
    Returns (json_text or None, next_index).
    """
    depth = 0
    buf: List[str] = []
    i = start_idx

    while i < len(lines):
        line = lines[i].rstrip("\n")
        if not buf:
            # must start with '['
            if line.strip().startswith("["):
                buf.append(line)
                depth += line.count("[") - line.count("]")
            else:
                return None, start_idx + 1
        else:
            buf.append(line)
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                return "\n".join(buf), i + 1
        i += 1

    return None, len(lines)


def parse_records(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Parse repeated blocks like:
      file_name = '...'
      basename = '...'
      recordingId = '...'
      [
        {...}
      ]
    """
    records: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        m = RE_FILE.match(line)
        if m:
            # flush previous record
            if cur is not None:
                records.append(cur)
            cur = {"file_name": m.group(1), "basename": None, "recordingId": None, "intervals": None}
            i += 1
            continue

        if cur is not None:
            m = RE_BASE.match(line)
            if m:
                cur["basename"] = normalize_basename_str(m.group(1))
                i += 1
                continue

            m = RE_RECID.match(line)
            if m:
                cur["recordingId"] = m.group(1)
                i += 1
                continue

            # intervals JSON list (starts with '[') inside a record
            if cur.get("intervals") is None and line.strip().startswith("["):
                json_text, j = _capture_json_list(lines, i)
                if json_text:
                    try:
                        intervals = json.loads(json_text)
                        cur["intervals"] = intervals
                    except json.JSONDecodeError:
                        cur["intervals"] = "JSON_DECODE_ERROR"
                        cur["intervals_raw"] = json_text
                    i = j
                    continue

        i += 1

    if cur is not None:
        records.append(cur)

    # normalize missing intervals to empty list
    for r in records:
        if r.get("intervals") is None:
            r["intervals"] = []
    return records


def print_table(rows: List[Tuple[str, str, str, int]]) -> None:
    headers = ("basename", "file_name", "recordingId", "#intervals")
    data = [headers] + [(a, b, c, str(d)) for a, b, c, d in rows]
    widths = [max(len(str(row[k])) for row in data) for k in range(4)]

    def fmt(row):
        return " | ".join(str(row[k]).ljust(widths[k]) for k in range(4))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt((r[0], r[1], r[2], str(r[3]))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Check for_noise_annotation basenames vs parsed records in Uzduotis.txt")
    ap.add_argument("txt_path", nargs="?", default="Uzduotis.txt", help="Path to Uzduotis.txt (default: Uzduotis.txt)")
    args = ap.parse_args()

    path = Path(args.txt_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    bn_list = parse_for_noise_annotation(lines)
    records = parse_records(lines)

    # Build lookup basename -> list[record]
    by_bn: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        bn = normalize_basename_str(r.get("basename"))
        if not bn:
            continue
        by_bn.setdefault(bn, []).append(r)

    # 1) print converted list of strings
    print("Converted for_noise_annotation list (strings):")
    print(json.dumps(bn_list, indent=2, ensure_ascii=False))

    # 2-4) check existence + count intervals
    missing: List[str] = []
    rows: List[Tuple[str, str, str, int]] = []
    total_intervals = 0

    print("\nCheck results:")
    for bn in bn_list:
        bn_norm = normalize_basename_str(bn)
        recs = by_bn.get(bn_norm)
        if not recs:
            print(f"  MISSING basename: {bn_norm}")
            missing.append(bn_norm)
            continue

        # If multiple records share same basename, we report each
        for r in recs:
            intervals = r.get("intervals", [])
            if intervals == "JSON_DECODE_ERROR":
                n = 0
                print(f"  OK basename={bn_norm} | file_name={r.get('file_name')} | recordingId={r.get('recordingId')} | intervals=JSON_DECODE_ERROR")
            elif isinstance(intervals, list):
                n = len(intervals)
                print(f"  OK basename={bn_norm} | file_name={r.get('file_name')} | recordingId={r.get('recordingId')} | intervals={n}")
            else:
                n = 0
                print(f"  OK basename={bn_norm} | file_name={r.get('file_name')} | recordingId={r.get('recordingId')} | intervals=UNKNOWN_FORMAT")

            total_intervals += n
            rows.append((bn_norm, str(r.get("file_name") or ""), str(r.get("recordingId") or ""), n))

    print("\nSummary:")
    print(f"  basenames_in_list: {len(bn_list)}")
    print(f"  found: {len(bn_list) - len(missing)}")
    print(f"  missing: {len(missing)}")
    if missing:
        print("  missing_basenames:", ", ".join(missing))
    print(f"  total_intervals_found: {total_intervals}")

    if rows:
        print("\nPer-basename table:")
        print_table(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""

python3 check_uzduotis.py 03_SarasasTriuksmuZymejimui.txt
"""