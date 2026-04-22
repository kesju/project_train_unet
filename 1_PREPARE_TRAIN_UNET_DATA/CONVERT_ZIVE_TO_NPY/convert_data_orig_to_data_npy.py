#!/usr/bin/env python3
"""
Convert ZIVE data to NPY format with logging.
Modified to insert "basename" after "userId" in output JSON.

https://chatgpt.com/c/69e79536-0278-83eb-826a-d182a5add8d1

What the Python script does:

    reads the requested list of original basenames
    reads ConversionTable.xlsx and maps basename -> filename
    checks presence of source ECG and source JSON
    converts source ECG into .npy
    copies JSON and inserts "basename": "<original_basename>" right after "userId"
    skips existing outputs unless --overwrite is used
    prints per-record status
    saves:
    conversion_report.csv
    conversion_summary.txt

Supported list file format:

    one basename per line
    comma-separated values
    mixed commas and lines
    # comments allowed


Important details:

    basename normalization is included, so values like 1636480.01 become 1636480.010
    output filenames are taken from the Excel filename column
    the script uses your uploaded helper functions load_ecg_npy and read_json_file from zive_data_read_utils.py

    One small fix for the .sh example:
    the last line in the script is commented for --overwrite, so if you want overwrite enabled,
    put it inside the command like this:


"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

parallel_path = Path.home() / "DI/2025_ZIVEO/PROJECT_TRAIN_UNET/SUPL_FUNCTIONS"
sys.path.append(str(parallel_path))

try:
    from zive_data_read_utils import read_json_file, load_ecg_npy
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Could not import zive_data_read_utils from: {parallel_path}"
    ) from e
    

@dataclass
class ConversionItemResult:
    requested_basename: str
    normalized_basename: str
    target_filename: str
    source_ecg_exists: bool
    source_json_exists: bool
    target_npy: str
    target_json: str
    status: str
    note: str
    samples: Optional[int] = None


def normalize_basename(value: Any) -> str:
    s = str(value).strip()
    if not s:
        return s
    try:
        return f"{float(s):.3f}"
    except ValueError:
        return s


def read_list_file(list_file: Path) -> List[str]:
    text = list_file.read_text(encoding="utf-8")
    items: List[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        items.extend([p for p in parts if p])
    return items


def load_conversion_mapping(conversion_table: Path) -> Dict[str, str]:
    df = pd.read_excel(conversion_table, dtype={"basename": str, "filename": str})

    required_cols = {"basename", "filename"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Conversion table is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    mapping: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        raw_basename = str(row["basename"]).strip()
        raw_filename = str(row["filename"]).strip()

        if not raw_basename or not raw_filename or raw_basename.lower() == "nan" or raw_filename.lower() == "nan":
            continue

        basename = normalize_basename(raw_basename)
        filename = Path(raw_filename).stem

        if basename in mapping and mapping[basename] != filename:
            duplicates.setdefault(basename, [mapping[basename]])
            if filename not in duplicates[basename]:
                duplicates[basename].append(filename)

        mapping[basename] = filename

    if duplicates:
        lines = [f"{basename}: {names}" for basename, names in sorted(duplicates.items())]
        raise ValueError(
            "Duplicate basenames with different filenames found in conversion table:\n"
            + "\n".join(lines)
        )

    return mapping


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def setup_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"conversion_log_{stamp}.txt"

    logger = logging.getLogger("convert_data_orig_to_data_npy")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger, log_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ECG records from data_orig format "
            "(basename + basename.json) into data_npy format "
            "(filename.npy + filename.json) using ConversionTable.xlsx, "
            "with timestamped logging."
        )
    )
    parser.add_argument("--data-orig", required=True, type=Path,
                        help="Folder with original ECG records, e.g. 1630807.618 and 1630807.618.json")
    parser.add_argument("--data-npy", required=True, type=Path,
                        help="Output folder for converted files, e.g. 1001_1.npy and 1001_1.json")
    parser.add_argument("--conversion-table", required=True, type=Path,
                        help='Excel file with columns "basename" and "filename"')
    parser.add_argument("--list-file", required=True, type=Path,
                        help="Text file with requested basenames to convert")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Folder for log and report files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files in data_npy")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    data_orig: Path = args.data_orig.expanduser().resolve()
    data_npy: Path = args.data_npy.expanduser().resolve()
    conversion_table: Path = args.conversion_table.expanduser().resolve()
    list_file: Path = args.list_file.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    overwrite: bool = args.overwrite

    for p, label in [
        (data_orig, "--data-orig"),
        (conversion_table, "--conversion-table"),
        (list_file, "--list-file"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    if not data_orig.is_dir():
        raise NotADirectoryError(f"--data-orig must be a directory: {data_orig}")

    data_npy.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logger(output_dir)

    requested = read_list_file(list_file)
    if not requested:
        raise ValueError(f"No basenames found in list file: {list_file}")

    mapping = load_conversion_mapping(conversion_table)

    results: List[ConversionItemResult] = []
    total_requested = len(requested)
    converted = 0
    skipped_existing = 0
    missing_mapping = 0
    missing_source = 0
    errors = 0

    logger.info("=== ECG conversion started ===")
    logger.info("data_orig        : %s", data_orig)
    logger.info("data_npy         : %s", data_npy)
    logger.info("conversion_table : %s", conversion_table)
    logger.info("list_file        : %s", list_file)
    logger.info("output_dir       : %s", output_dir)
    logger.info("overwrite        : %s", overwrite)
    logger.info("requested count  : %s", total_requested)

    for idx, requested_basename in enumerate(requested, start=1):
        normalized_basename = normalize_basename(requested_basename)
        target_filename = mapping.get(normalized_basename, "")

        logger.info("[%s/%s] requested=%s normalized=%s", idx, total_requested, requested_basename, normalized_basename)

        if not target_filename:
            logger.warning("status=MISSING_MAPPING | basename not found in conversion table")
            results.append(
                ConversionItemResult(
                    requested_basename=requested_basename,
                    normalized_basename=normalized_basename,
                    target_filename="",
                    source_ecg_exists=False,
                    source_json_exists=False,
                    target_npy="",
                    target_json="",
                    status="MISSING_MAPPING",
                    note="basename not found in conversion table",
                )
            )
            missing_mapping += 1
            continue

        src_ecg = data_orig / normalized_basename
        src_json = data_orig / f"{normalized_basename}.json"
        dst_npy = data_npy / f"{target_filename}.npy"
        dst_json = data_npy / f"{target_filename}.json"

        src_ecg_exists = src_ecg.exists() and src_ecg.is_file()
        src_json_exists = src_json.exists() and src_json.is_file()

        if not src_ecg_exists or not src_json_exists:
            note_parts = []
            if not src_ecg_exists:
                note_parts.append("source ECG missing")
            if not src_json_exists:
                note_parts.append("source JSON missing")
            note = "; ".join(note_parts)
            logger.warning("status=MISSING_SOURCE | %s", note)
            results.append(
                ConversionItemResult(
                    requested_basename=requested_basename,
                    normalized_basename=normalized_basename,
                    target_filename=target_filename,
                    source_ecg_exists=src_ecg_exists,
                    source_json_exists=src_json_exists,
                    target_npy=str(dst_npy),
                    target_json=str(dst_json),
                    status="MISSING_SOURCE",
                    note=note,
                )
            )
            missing_source += 1
            continue

        if not overwrite and (dst_npy.exists() or dst_json.exists()):
            logger.info("status=SKIPPED_EXISTS | target file already exists; use --overwrite to replace")
            results.append(
                ConversionItemResult(
                    requested_basename=requested_basename,
                    normalized_basename=normalized_basename,
                    target_filename=target_filename,
                    source_ecg_exists=True,
                    source_json_exists=True,
                    target_npy=str(dst_npy),
                    target_json=str(dst_json),
                    status="SKIPPED_EXISTS",
                    note="target file already exists; use --overwrite to replace",
                )
            )
            skipped_existing += 1
            continue

        try:
            signal = load_ecg_npy(src_ecg)
            metadata = read_json_file(src_json)

            metadata_out: Dict[str, Any] = {}
            inserted = False
            for key, value in metadata.items():
                metadata_out[key] = value
                if key == "userId":
                    metadata_out["basename"] = normalized_basename
                    inserted = True

            if not inserted:
                metadata_out["basename"] = normalized_basename

            np.save(dst_npy, np.asarray(signal))
            save_json(dst_json, metadata_out)

            logger.info("status=CONVERTED | npy=%s | json=%s | samples=%s", dst_npy.name, dst_json.name, int(signal.shape[0]))
            results.append(
                ConversionItemResult(
                    requested_basename=requested_basename,
                    normalized_basename=normalized_basename,
                    target_filename=target_filename,
                    source_ecg_exists=True,
                    source_json_exists=True,
                    target_npy=str(dst_npy),
                    target_json=str(dst_json),
                    status="CONVERTED",
                    note="ok",
                    samples=int(signal.shape[0]),
                )
            )
            converted += 1

        except Exception as exc:
            logger.exception("status=ERROR | %s", exc)
            results.append(
                ConversionItemResult(
                    requested_basename=requested_basename,
                    normalized_basename=normalized_basename,
                    target_filename=target_filename,
                    source_ecg_exists=True,
                    source_json_exists=True,
                    target_npy=str(dst_npy),
                    target_json=str(dst_json),
                    status="ERROR",
                    note=str(exc),
                )
            )
            errors += 1

    report_csv = output_dir / "conversion_report.csv"
    summary_txt = output_dir / "conversion_summary.txt"

    report_df = pd.DataFrame([asdict(item) for item in results])
    report_df.to_csv(report_csv, index=False, encoding="utf-8")

    summary_lines = [
        "ECG conversion summary",
        "======================",
        f"data_orig        : {data_orig}",
        f"data_npy         : {data_npy}",
        f"conversion_table : {conversion_table}",
        f"list_file        : {list_file}",
        f"output_dir       : {output_dir}",
        f"overwrite        : {overwrite}",
        f"log_file         : {log_path}",
        "",
        f"requested        : {total_requested}",
        f"converted        : {converted}",
        f"skipped_existing : {skipped_existing}",
        f"missing_mapping  : {missing_mapping}",
        f"missing_source   : {missing_source}",
        f"errors           : {errors}",
        "",
        f"report_csv       : {report_csv}",
    ]
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    logger.info("=== SUMMARY ===")
    for line in summary_lines[10:]:
        logger.info(line)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
