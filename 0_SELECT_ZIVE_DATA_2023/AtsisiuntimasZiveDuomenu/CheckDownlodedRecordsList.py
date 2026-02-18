#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "recordings" not in data or not isinstance(data["recordings"], list):
        raise ValueError(f"Invalid manifest structure in: {path}")
    return data


def load_ids(txt_path: Path) -> List[str]:
    """
    Accepts newline-separated, comma-separated, or mixed IDs.
    Strips quotes/apostrophes and de-dups while preserving order.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace(",", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    parts = [p.strip().strip("'\"") for p in raw.split(" ") if p.strip()]

    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check recordingIds vs manifest.json and print records with non-success status")
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/data/manifest.json"),
                    help="Path to manifest.json")
    ap.add_argument("--ids", type=Path, default=Path("/mnt/data/01_100_list.txt"),
                    help="Text file containing recordingId list (newline/comma separated)")
    ap.add_argument("--success-value", default="success",
                    help='Value considered successful for the "status" field (default: "success")')
    ap.add_argument("--print-missing", action="store_true",
                    help="Also print missing IDs (to stderr) if they are not found in manifest")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    recordings: List[Dict[str, Any]] = manifest["recordings"]

    # Index by recordingId
    by_id: Dict[str, Dict[str, Any]] = {}
    for rec in recordings:
        if isinstance(rec, dict) and "recordingId" in rec:
            rid = str(rec["recordingId"]).strip()
            by_id[rid] = rec

    ids = load_ids(args.ids)

    missing: List[str] = []
    bad_status: List[str] = []

    success_value = str(args.success_value).strip().lower()

    # Print header for clarity (optional; comment out if you want pure JSON only)
    # print("Records with non-success status:\n", file=sys.stderr)

    for rid in ids:
        rec = by_id.get(rid)
        if rec is None:
            missing.append(rid)
            continue

        status = str(rec.get("status", "")).strip().lower()
        if status != success_value:
            bad_status.append(rid)
            # Print the full record from "recordings"
            print(json.dumps(rec, ensure_ascii=False, indent=2))

    # Summary to stderr (won't break JSON output piping)
    print(f"IDs in txt: {len(ids)}", file=sys.stderr)
    print(f"Found in manifest: {len(ids) - len(missing)}", file=sys.stderr)
    print(f"Missing in manifest: {len(missing)}", file=sys.stderr)
    print(f"Non-success status records printed: {len(bad_status)}", file=sys.stderr)

    if args.print_missing and missing:
        print("\nMISSING recordingIds (in txt, absent in manifest):", file=sys.stderr)
        for rid in missing:
            print(rid, file=sys.stderr)

    # Exit code: 0 if everything OK; 2 if missing; 3 if bad status; 4 if both
    rc = 0
    if missing:
        rc |= 2
    if bad_status:
        rc |= 3
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""

Just check and list missing IDs

python CheckDownlodedRecordsList.py --ids "01_100/01_100_list.txt" --manifest "01_100/manifest.json"


2) Print full JSON record for each ID that IS found (like your example)

python CheckDownlodedRecordsList.py --print-found


3) Save missing IDs to a file (JSONL)

python CheckDownlodedRecordsList.py --only-missing --out-missing-jsonl missing_ids.jsonl

"""