#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict


def format_elapsed_minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{minutes:.1f} min"


def format_elapsed_hhmm(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read ZIVE ECG records and matching JSON metadata."
    )

    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Folder with ECG data and JSON files",
    )

    parser.add_argument(
        "--exclude-list",
        type=Path,
        default=None,
        help="Path to exclude list file",
    )

    parser.add_argument(
        "--parallel-path",
        type=Path,
        default=Path.home() / "DI/2025_ZIVEO/PROJECT_TRAIN_UNET/SUPL_FUNCTIONS",
        help="Folder containing zive_data_read_utils.py",
    )

    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Process all matched records. If omitted, only first 5 records are processed.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    src = args.dir.expanduser().resolve()
    exclude_list = args.exclude_list.expanduser().resolve() if args.exclude_list else None
    parallel_path = args.parallel_path.expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Input directory does not exist: {src}")

    if exclude_list is not None and not exclude_list.exists():
        raise FileNotFoundError(f"Exclude list file does not exist: {exclude_list}")

    if not parallel_path.exists():
        raise FileNotFoundError(f"Parallel path does not exist: {parallel_path}")

    sys.path.append(str(parallel_path))

    try:
        from zive_data_read_utils import list_ecg_records, read_json_file
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import zive_data_read_utils from: {parallel_path}"
        ) from e

    scan_result = list_ecg_records(
        folder=src,
        data_format="auto",
        exclude_list=exclude_list,
    )

    records = scan_result.records
    summary = scan_result.summary

    print(f"total_json       : {summary.total_json}")
    print(f"excluded         : {summary.excluded}")
    print(f"matched          : {summary.matched}")
    print(f"unmatched_json   : {summary.unmatched_json}")
    print(f"records returned : {len(records)}")

    records_to_process = records if args.all_records else records[:5]

    print(f"\nFound {len(records)} matched records")
    print(f"Processing {len(records_to_process)} record(s)")

    total_cycle_start = time.perf_counter()
    record_nr = 0

    for rec in records_to_process:
        print("\n" + "-" * 90)
        record_nr += 1

        elapsed_from_start_s = time.perf_counter() - total_cycle_start
        elapsed_from_start_min = format_elapsed_minutes(elapsed_from_start_s)

        if rec.ecg_path is not None:
            print(
                f"{record_nr}/{len(records_to_process)} | "
                f"{rec.basename} | {rec.ecg_path.name} | {rec.json_path.name} | "
                f"elapsed: {elapsed_from_start_min}",
                flush=True,
            )
        else:
            print(
                f"{record_nr}/{len(records_to_process)} | "
                f"{rec.basename} | <missing ecg> | {rec.json_path.name} | "
                f"elapsed: {elapsed_from_start_min}",
                flush=True,
            )

        if rec.ecg_path is None:
            msg = f"No matching ECG file for JSON '{rec.json_path.name}'"
            print(msg, flush=True)
            continue

        metadata: Dict[str, Any] = read_json_file(rec.json_path)
        _ = metadata

    total_cycle_elapsed_s = time.perf_counter() - total_cycle_start
    print("\n" + "=" * 90)
    print(f"Total cycle time: {format_elapsed_hhmm(total_cycle_elapsed_s)} (hh:mm)")
    print("=" * 90)


if __name__ == "__main__":
    main()
