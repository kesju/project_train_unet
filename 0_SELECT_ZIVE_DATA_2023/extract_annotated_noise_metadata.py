#!/usr/bin/env python3
"""Extract annotated noise intervals and summary statistics from Zive metadata JSON."""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable


def load_metadata(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} does not contain a JSON object")
    return data


def normalize_noise_intervals(noises: Any) -> list[tuple[int, int]]:
    """Return valid (startIndex, endIndex) pairs; endIndex is treated as exclusive."""
    if isinstance(noises, dict):
        for key in ("merged", "human", "annotated"):
            if key in noises:
                noises = noises[key]
                break

    if not isinstance(noises, list):
        return []

    intervals: list[tuple[int, int]] = []
    for item in noises:
        start = end = None
        if isinstance(item, dict):
            start = item.get("startIndex", item.get("startSample"))
            end = item.get("endIndex", item.get("endSample"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]

        if isinstance(start, float) and start.is_integer():
            start = int(start)
        if isinstance(end, float) and end.is_integer():
            end = int(end)

        if isinstance(start, int) and isinstance(end, int) and end > start:
            intervals.append((start, end))

    return intervals


def find_data_file(json_path: Path) -> tuple[Path, str] | None:
    """Find the data file paired with a metadata JSON, like update_records_list_v7.py."""
    if json_path.suffix.lower() != ".json":
        return None

    rec_dir = json_path.parent
    stem = json_path.stem

    exact = rec_dir / stem
    if exact.exists():
        suffix = exact.suffix
        if suffix and suffix[1:].isdigit() and len(suffix) == 4:
            return exact, suffix

    candidates = [
        rec_dir / f"{stem}.npy",
        rec_dir / f"{stem}.bin",
        rec_dir / stem,
    ]
    for path in sorted(rec_dir.glob(f"{stem}.*")):
        suffix = path.suffix
        if suffix and suffix[1:].isdigit() and len(suffix) == 4:
            candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            return candidate, candidate.suffix
    return None


def sample_count(data_path: Path, ext: str) -> int:
    """Count samples with the same convention as update_records_list_v7.py."""
    if ext.lower() == ".npy":
        return npy_sample_count(data_path)
    return int(data_path.stat().st_size // 4)


def infer_samples(json_path: Path) -> tuple[int | None, str | None]:
    data_file = find_data_file(json_path)
    if data_file is None:
        return None, None

    data_path, data_ext = data_file
    return sample_count(data_path, data_ext), str(data_path)


def npy_sample_count(npy_path: Path) -> int:
    """Read a .npy header and return the total number of elements."""
    with npy_path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{npy_path} is not a .npy file")

        major, _minor = f.read(2)
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}")

        header = f.read(header_len).decode("latin1")

    shape = ast.literal_eval(header)["shape"]
    if not shape:
        return 1
    return int(math.prod(shape))


def covered_samples(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def extract_annotated_noise(json_path: Path, samples: int | None = None) -> dict[str, Any]:
    meta = load_metadata(json_path)
    intervals = normalize_noise_intervals(meta.get("noises_annotated", []))
    noise_samples = covered_samples(intervals)

    data_path = None
    if samples is None:
        samples, data_path = infer_samples(json_path)

    noise_fraction_percent = None
    if samples and samples > 0:
        noise_fraction_percent = noise_samples / samples * 100.0

    return {
        "json_path": str(json_path),
        "noisy_interval_indexes": intervals,
        "noisy_interval_pairs_count": len(intervals),
        "noise_samples": noise_samples,
        "samples": samples,
        "data_path": data_path,
        "noise_fraction_percent": noise_fraction_percent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract noises_annotated intervals, pair count, and noise percent."
    )
    parser.add_argument("json_path", type=Path, help="Path to metadata JSON file")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Total ECG samples. If omitted, the paired data file is used.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON output.",
    )
    args = parser.parse_args()

    result = extract_annotated_noise(args.json_path, args.samples)
    print(json.dumps(result, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
