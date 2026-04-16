#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check which ECG data files and JSON files:
1) are listed in a text file and exist in a folder,
2) are listed in a text file but do not exist in a folder,
3) exist in the folder but are not included in the list.

Normalization rule for names from the text file:
    1632728.75   -> 1632728.750
    1633591.7    -> 1633591.700
    1630737.000  -> 1630737.000

Examples of folder contents:
    data file: 1631050.134
    json file: 1631050.134.json
"""

from __future__ import annotations

import argparse
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable


NUMERIC_NAME_RE = re.compile(r"^\d+\.\d+$")
JSON_NUMERIC_NAME_RE = re.compile(r"^(\d+\.\d+)\.json$", re.IGNORECASE)


def normalize_numeric_name(name: str) -> str | None:
    """
    Normalize numeric filename to exactly 3 decimal places.

    Examples:
        '1632728.75'   -> '1632728.750'
        '1633591.7'    -> '1633591.700'
        '1630737.000'  -> '1630737.000'

    Returns None if the value is not a valid numeric filename.
    """
    name = name.strip()
    if not name:
        return None

    if not NUMERIC_NAME_RE.match(name):
        return None

    try:
        value = Decimal(name)
    except InvalidOperation:
        return None

    # Force exactly 3 decimal digits
    value = value.quantize(Decimal("0.000"), rounding=ROUND_HALF_UP)
    return format(value, "f")


def read_and_normalize_list(list_file: Path) -> tuple[list[str], list[str]]:
    """
    Read list file and normalize all valid numeric names.
    Returns:
        normalized_unique_sorted
        invalid_lines
    """
    normalized = []
    invalid_lines = []

    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            norm = normalize_numeric_name(raw)
            if norm is None:
                invalid_lines.append(raw)
            else:
                normalized.append(norm)

    normalized_unique_sorted = sorted(set(normalized))
    return normalized_unique_sorted, invalid_lines


def scan_folder(folder: Path) -> tuple[set[str], set[str], list[str]]:
    """
    Scan folder and detect:
      - data files with names like 1631050.134
      - json files with names like 1631050.134.json

    Returns:
      data_names_set         -> normalized numeric names for data files
      json_names_set         -> normalized numeric names for json stems
      unrecognized_files     -> filenames that do not match either pattern
    """
    data_names: set[str] = set()
    json_names: set[str] = set()
    unrecognized_files: list[str] = []

    for p in folder.iterdir():
        if not p.is_file():
            continue

        fname = p.name

        # JSON file: 1631050.134.json
        m = JSON_NUMERIC_NAME_RE.match(fname)
        if m:
            base = m.group(1)
            norm = normalize_numeric_name(base)
            if norm is not None:
                json_names.add(norm)
            else:
                unrecognized_files.append(fname)
            continue

        # Data file: 1631050.134
        if NUMERIC_NAME_RE.match(fname):
            norm = normalize_numeric_name(fname)
            if norm is not None:
                data_names.add(norm)
            else:
                unrecognized_files.append(fname)
            continue

        unrecognized_files.append(fname)

    return data_names, json_names, sorted(unrecognized_files)


def write_list(path: Path, items: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def build_report(
    folder: Path,
    list_file: Path,
    output_dir: Path,
    list_names: list[str],
    invalid_lines: list[str],
    data_in_folder: set[str],
    json_in_folder: set[str],
    unrecognized_files: list[str],
) -> str:
    list_set = set(list_names)

    listed_data_exists = sorted(list_set & data_in_folder)
    listed_data_missing = sorted(list_set - data_in_folder)

    listed_json_exists = sorted(list_set & json_in_folder)
    listed_json_missing = sorted(list_set - json_in_folder)

    data_not_in_list = sorted(data_in_folder - list_set)
    json_not_in_list = sorted(json_in_folder - list_set)

    both_exist = sorted(list_set & data_in_folder & json_in_folder)
    missing_any = sorted(
        name for name in list_set
        if name not in data_in_folder or name not in json_in_folder
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save normalized list
    write_list(output_dir / "normalized_list.txt", list_names)

    # Save invalid lines from the source list
    write_list(output_dir / "invalid_lines_in_list.txt", invalid_lines)

    # Save comparison results
    write_list(output_dir / "listed_data_exists.txt", listed_data_exists)
    write_list(output_dir / "listed_data_missing.txt", listed_data_missing)

    write_list(output_dir / "listed_json_exists.txt", listed_json_exists)
    write_list(output_dir / "listed_json_missing.txt", listed_json_missing)

    write_list(output_dir / "data_in_folder_not_in_list.txt", data_not_in_list)
    write_list(output_dir / "json_in_folder_not_in_list.txt", json_not_in_list)

    write_list(output_dir / "listed_both_data_and_json_exist.txt", both_exist)
    write_list(output_dir / "listed_missing_data_or_json.txt", missing_any)

    write_list(output_dir / "all_data_files_in_folder.txt", sorted(data_in_folder))
    write_list(output_dir / "all_json_files_in_folder.txt", sorted(json_in_folder))
    write_list(output_dir / "unrecognized_files_in_folder.txt", unrecognized_files)

    report_lines = []
    report_lines.append("CHECK OF DATA AND JSON FILES AGAINST LIST")
    report_lines.append("=" * 70)
    report_lines.append(f"Folder      : {folder}")
    report_lines.append(f"List file   : {list_file}")
    report_lines.append(f"Output dir  : {output_dir}")
    report_lines.append("")

    report_lines.append("LIST SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Valid unique normalized names in list : {len(list_names)}")
    report_lines.append(f"Invalid lines in list                 : {len(invalid_lines)}")
    report_lines.append("")

    report_lines.append("FOLDER SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Data files in folder                  : {len(data_in_folder)}")
    report_lines.append(f"JSON files in folder                  : {len(json_in_folder)}")
    report_lines.append(f"Unrecognized files in folder          : {len(unrecognized_files)}")
    report_lines.append("")

    report_lines.append("COMPARISON AGAINST LIST")
    report_lines.append("-" * 70)
    report_lines.append(f"Listed data files that exist          : {len(listed_data_exists)}")
    report_lines.append(f"Listed data files missing             : {len(listed_data_missing)}")
    report_lines.append(f"Listed JSON files that exist          : {len(listed_json_exists)}")
    report_lines.append(f"Listed JSON files missing             : {len(listed_json_missing)}")
    report_lines.append("")
    report_lines.append(f"Listed names where BOTH data+json exist : {len(both_exist)}")
    report_lines.append(f"Listed names missing data or json       : {len(missing_any)}")
    report_lines.append("")
    report_lines.append(f"Data files in folder but not in list  : {len(data_not_in_list)}")
    report_lines.append(f"JSON files in folder but not in list  : {len(json_not_in_list)}")

    report_text = "\n".join(report_lines)

    with (output_dir / "report_summary.txt").open("w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which data and JSON files exist/miss versus a list of numeric filenames."
    )
    parser.add_argument(
        "--folder",
        required=True,
        type=Path,
        help='Folder with files, e.g. "Atsisiusti_visi_anotuoti_duomenys_26_04_14"',
    )
    parser.add_argument(
        "--list-file",
        required=True,
        type=Path,
        help='Text file with filenames, e.g. "visi_zive_irasai_annot-Darb.txt"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("check_files_results"),
        help="Output directory for result text files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = args.folder.expanduser().resolve()
    list_file = args.list_file.expanduser().resolve()

    if args.output_dir.is_absolute():
        output_dir = args.output_dir.expanduser()
    else:
        output_dir = (Path.cwd() / args.output_dir).expanduser()

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    if not list_file.exists() or not list_file.is_file():
        raise FileNotFoundError(f"List file does not exist: {list_file}")

    list_names, invalid_lines = read_and_normalize_list(list_file)
    data_in_folder, json_in_folder, unrecognized_files = scan_folder(folder)

    report_text = build_report(
        folder=folder,
        list_file=list_file,
        output_dir=output_dir,
        list_names=list_names,
        invalid_lines=invalid_lines,
        data_in_folder=data_in_folder,
        json_in_folder=json_in_folder,
        unrecognized_files=unrecognized_files,
    )

    print(report_text)
    print("\nDetailed files saved in:")
    print(output_dir)

if __name__ == "__main__":
    main()