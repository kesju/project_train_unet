#!/usr/bin/env python3
"""
Analyze ECG sequences packaged in a ZIP or an already-unzipped folder.

Expected layout (3 sequences in your dataset):
<sequence_id>/
    gaps.json                     # {"gaps": [[start,end], ...]} indices in merged sequence
    merged-sequence.bin           # int16 ECG samples with gaps inserted (gap samples are included)
    recordings/
        <recording_id>.bin        # int16 ECG samples for that record (no gaps)
        <recording_id>.json       # metadata: flags, rpeaks, noises, etc.

What the script does:
- Discovers sequences (top-level folders containing gaps.json).
- For every record, reads its JSON + BIN size and summarizes:
  flags, rpeaks count + annotation distribution, noise intervals, durations.
- For every sequence, summarizes:
  number of records, total samples/duration, gap samples/duration, noise coverage, flag totals.
- Writes:
  - report.json (full structured summary)
  - records.csv (one row per record)
  - sequences.csv (one row per sequence)
- Prints a concise human-readable report.

Notes
- ECG samples are treated as int16. This matches your merged-sequence.bin sizing in the provided ZIP.
- Sampling frequency (fs) is required: durations and statistics rely on the given fs.
"""

# https://chatgpt.com/c/69737468-8a44-8327-bcde-1ef6276bf3be

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
JsonLikeRef = Union[str, Path]

def zive_read_file_1ch(filename):
    with open(filename, 'rb') as f:  # Use 'rb' to read binary file
        a = np.fromfile(f, dtype=np.dtype('>i4'))  # Read file content as big-endian 4-byte integers
    
    ADCmax = 0x800000
    Vref = 2.5
    b = (a - ADCmax / 2) * 2 * Vref / ADCmax / 3.5 * 1000  # Corrected the calculation by adding multiplication symbol
    ecg_signal = b - np.mean(b)
    
    return ecg_signal


def _safe_json_load(zf: zipfile.ZipFile, member: str) -> Any:
    with zf.open(member) as f:
        return json.load(f)


def _safe_json_load_path(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_int16_sized(byte_size: int) -> bool:
    return (byte_size % 2) == 0


def _samples_from_int16_bytes(byte_size: int) -> int:
    return byte_size // 2


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


def _flatten_flags(flags: Any) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, list):
        return [str(x) for x in flags]
    # unexpected type
    return [str(flags)]


def _sum_noise_samples(noises: Any) -> Tuple[int, int]:
    """
    Returns (interval_count, total_samples_covered) for noises list.
    Accepts items like {"startIndex": x, "endIndex": y}.
    """
    if not isinstance(noises, list):
        return 0, 0
    cnt = 0
    total = 0
    for it in noises:
        if not isinstance(it, dict):
            continue
        a = it.get("startIndex")
        b = it.get("endIndex")
        if isinstance(a, int) and isinstance(b, int) and b > a:
            cnt += 1
            total += (b - a)
    return cnt, total


def _summarize_rpeaks(rpeaks: Any) -> Tuple[int, Optional[int], Optional[int], Dict[str, int]]:
    """
    rpeaks: list of {"sampleIndex": int, "annotationValue": str}
    Returns (count, first_idx, last_idx, annotation_counts)
    """
    if not isinstance(rpeaks, list) or len(rpeaks) == 0:
        return 0, None, None, {}
    idxs: List[int] = []
    ann: Dict[str, int] = {}
    for rp in rpeaks:
        if not isinstance(rp, dict):
            continue
        si = rp.get("sampleIndex")
        av = rp.get("annotationValue")
        if isinstance(si, int):
            idxs.append(si)
        if isinstance(av, str):
            ann[av] = ann.get(av, 0) + 1
        elif av is not None:
            ann[str(av)] = ann.get(str(av), 0) + 1
    if not idxs:
        return 0, None, None, ann
    idxs.sort()
    return len(idxs), idxs[0], idxs[-1], ann


# -----------------------------
# Data models
# -----------------------------
@dataclass
class RecordSummary:
    sequence_id: str
    recording_id: str
    # json_path: str
    channel_count: Optional[int]
    user_id: Optional[str]

    bin_bytes: int
    samples: int
    duration_s: Optional[float]

    flags_count: int
    flags: List[str]

    rpeaks_count: int
    rpeaks_first: Optional[int]
    rpeaks_last: Optional[int]
    rpeak_ann_counts: Dict[str, int]

    json_ok: bool

    noises_count: int
    noises_samples: int
    noises_fraction: Optional[float]

    has_comment: bool
    json_keys: List[str]


@dataclass
class SequenceSummary:
    sequence_id: str
    records: int

    merged_bytes: int
    merged_samples: int
    merged_duration_s: Optional[float]

    records_bytes_sum: int
    records_samples_sum: int
    records_duration_s: Optional[float]

    gap_intervals: int
    gap_samples: int
    gap_duration_s: Optional[float]
    gap_fraction_of_merged: float

    noises_intervals_total: int
    noises_samples_total: int
    noises_fraction_of_records: Optional[float]

    flags_totals: Dict[str, int]
    rpeak_ann_totals: Dict[str, int]


# -----------------------------
# Core analysis
# -----------------------------
def discover_sequences(zf: zipfile.ZipFile) -> List[str]:
    """
    Returns top-level folders that contain <seq>/gaps.json.
    """
    names = zf.namelist()
    seqs = sorted({n.split("/")[0] for n in names if n.endswith("/gaps.json")})
    return seqs


def discover_sequences_dir(root: Path) -> List[str]:
    """Top-level directories that contain gaps.json when data is already extracted."""
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "gaps.json").exists()])


def list_recording_pairs(zf: zipfile.ZipFile, seq: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (recording_id, json_member, bin_member)
    """
    prefix = f"{seq}/recordings/"
    names = [n for n in zf.namelist() if n.startswith(prefix)]
    jsons = {Path(n).stem: n for n in names if n.endswith(".json")}
    bins = {Path(n).stem: n for n in names if n.endswith(".bin")}
    ids = sorted(set(jsons) & set(bins))
    return [(rid, jsons[rid], bins[rid]) for rid in ids]


def list_recording_pairs_dir(root: Path, seq: str) -> List[Tuple[str, Path, Path]]:
    """(recording_id, json_path, bin_path) for extracted data."""
    rec_dir = root / seq / "recordings"
    if not rec_dir.is_dir():
        return []
    jsons = {p.stem: p for p in rec_dir.glob("*.json")}
    bins = {p.stem: p for p in rec_dir.glob("*.bin")}
    ids = sorted(set(jsons) & set(bins))
    return [(rid, jsons[rid], bins[rid]) for rid in ids]


def analyze(input_path: Path, out_dir: Path, fs: Optional[int]) -> None:
    """
    Analyze ECG package provided either as a ZIP file or an already-unzipped folder
    that follows the same layout.
    """

    if fs is None:
        raise RuntimeError("Sampling frequency (--fs) is required; inference is disabled.")

    out_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(
        sequences: List[str],
        list_pairs_fn: Callable[[str], List[Tuple[str, JsonLikeRef, JsonLikeRef]]],
        load_json_fn: Callable[[JsonLikeRef], Any],
        file_size_fn: Callable[[JsonLikeRef], int],
        merged_size_fn: Callable[[str], int],
        gaps_loader_fn: Callable[[str], Any],
        source_label: str,
    ) -> Tuple[List[RecordSummary], List[SequenceSummary], Optional[int]]:
        if not sequences:
            raise RuntimeError("No sequences found. Expected top-level folders containing gaps.json.")

        record_rows: List[RecordSummary] = []
        seq_rows: List[SequenceSummary] = []

        for seq in sequences:
            pairs = list_pairs_fn(seq)

            gaps_obj = gaps_loader_fn(seq)
            gaps = gaps_obj.get("gaps", []) if isinstance(gaps_obj, dict) else []
            gap_samples = 0
            gap_intervals = 0
            if isinstance(gaps, list):
                for it in gaps:
                    if isinstance(it, list) and len(it) == 2 and all(isinstance(x, int) for x in it):
                        a, b = it
                        if b > a:
                            gap_intervals += 1
                            gap_samples += (b - a)

            merged_bytes = merged_size_fn(seq)
            if not _is_int16_sized(merged_bytes):
                raise RuntimeError(f"{seq}/merged-sequence.bin size is not divisible by 2 (unexpected for int16).")
            merged_samples = _samples_from_int16_bytes(merged_bytes)

            records_bytes_sum = 0
            records_samples_sum = 0
            noises_intervals_total = 0
            noises_samples_total = 0
            flags_totals: Dict[str, int] = {}
            rpeak_ann_totals: Dict[str, int] = {}

            for rid, json_member, bin_member in pairs:
                try:
                    meta = load_json_fn(json_member)
                    json_ok = True
                except Exception as exc:  # keep processing even if a single record json is bad
                    print(f"WARN: failed to load JSON for {seq}/{rid}: {exc}")
                    meta = {}
                    json_ok = False
                bsz = file_size_fn(bin_member)
                if not _is_int16_sized(bsz):
                    samples = 0
                else:
                    samples = _samples_from_int16_bytes(bsz)

                channel_count = meta.get("channelCount") if isinstance(meta, dict) else None
                user_id = meta.get("userId") if isinstance(meta, dict) else None

                flags_list = _flatten_flags(meta.get("flags") if isinstance(meta, dict) else None)
                for fl in flags_list:
                    flags_totals[fl] = flags_totals.get(fl, 0) + 1

                rpeaks = meta.get("rpeaks") if isinstance(meta, dict) else None
                rpk_cnt, rpk_first, rpk_last, rpk_ann = _summarize_rpeaks(rpeaks)

                ann_counts = meta.get("rpeakAnnotationCounts") if isinstance(meta, dict) else None
                ann_counts_clean: Dict[str, int] = {}
                if isinstance(ann_counts, dict):
                    for k, v in ann_counts.items():
                        if k == "__info":
                            continue
                        if isinstance(v, int):
                            ann_counts_clean[str(k)] = v
                else:
                    ann_counts_clean = rpk_ann

                for k, v in ann_counts_clean.items():
                    rpeak_ann_totals[k] = rpeak_ann_totals.get(k, 0) + int(v)

                noises = meta.get("noises") if isinstance(meta, dict) else None
                nz_cnt, nz_samples = _sum_noise_samples(noises)

                noises_intervals_total += nz_cnt
                noises_samples_total += nz_samples

                duration_s = (samples / fs) if (fs is not None and fs > 0 and samples > 0) else None
                noises_fraction = (nz_samples / samples) if (samples > 0) else None

                rec = RecordSummary(
                    sequence_id=seq,
                    recording_id=rid,
                    # json_path=str(json_member),
                    channel_count=int(channel_count) if isinstance(channel_count, int) else None,
                    user_id=str(user_id) if isinstance(user_id, str) else None,
                    bin_bytes=int(bsz),
                    samples=int(samples),
                    duration_s=float(duration_s) if duration_s is not None else None,
                    flags_count=len(flags_list),
                    flags=flags_list,
                    rpeaks_count=int(rpk_cnt),
                    rpeaks_first=rpk_first,
                    rpeaks_last=rpk_last,
                    rpeak_ann_counts=ann_counts_clean,
                    json_ok=json_ok,
                    noises_count=int(nz_cnt),
                    noises_samples=int(nz_samples),
                    noises_fraction=float(noises_fraction) if noises_fraction is not None else None,
                    has_comment=bool(meta.get("comment")) if isinstance(meta, dict) else False,
                    json_keys=sorted(list(meta.keys())) if isinstance(meta, dict) else [],
                )
                record_rows.append(rec)

                records_bytes_sum += int(bsz)
                records_samples_sum += int(samples)

            merged_duration_s = (merged_samples / fs) if (fs is not None and fs > 0) else None
            records_duration_s = (records_samples_sum / fs) if (fs is not None and fs > 0) else None
            gap_duration_s = (gap_samples / fs) if (fs is not None and fs > 0) else None
            gap_fraction = (gap_samples / merged_samples) if merged_samples > 0 else 0.0
            noises_fraction_of_records = (noises_samples_total / records_samples_sum) if records_samples_sum > 0 else None

            seq_rows.append(SequenceSummary(
                sequence_id=seq,
                records=len(pairs),
                merged_bytes=int(merged_bytes),
                merged_samples=int(merged_samples),
                merged_duration_s=float(merged_duration_s) if merged_duration_s is not None else None,
                records_bytes_sum=int(records_bytes_sum),
                records_samples_sum=int(records_samples_sum),
                records_duration_s=float(records_duration_s) if records_duration_s is not None else None,
                gap_intervals=int(gap_intervals),
                gap_samples=int(gap_samples),
                gap_duration_s=float(gap_duration_s) if gap_duration_s is not None else None,
                gap_fraction_of_merged=float(gap_fraction),
                noises_intervals_total=int(noises_intervals_total),
                noises_samples_total=int(noises_samples_total),
                noises_fraction_of_records=float(noises_fraction_of_records) if noises_fraction_of_records is not None else None,
                flags_totals=dict(sorted(flags_totals.items(), key=lambda kv: (-kv[1], kv[0]))),
                rpeak_ann_totals=dict(sorted(rpeak_ann_totals.items(), key=lambda kv: (-kv[1], kv[0]))),
            ))

        print(f"\nSource: {source_label}")
        print(f"Sequences discovered: {len(sequences)}")
        print(f"Sampling frequency (fs): {fs if fs is not None else 'unknown'} Hz")
        print(f"Outputs written to: {out_dir}\n")

        for s in seq_rows:
            print(f"=== Sequence {s.sequence_id} ===")
            print(f"Records: {s.records}")
            print(f"Merged: {s.merged_samples:,} samples ({_fmt_seconds(s.merged_duration_s)}) | {s.merged_bytes:,} bytes")
            print(f"Records sum: {s.records_samples_sum:,} samples ({_fmt_seconds(s.records_duration_s)}) | {s.records_bytes_sum:,} bytes")
            print(f"Gaps: {s.gap_intervals} intervals, {s.gap_samples:,} samples ({_fmt_seconds(s.gap_duration_s)}), {s.gap_fraction_of_merged*100:.2f}% of merged")
            if s.noises_fraction_of_records is not None:
                print(f"Noises: {s.noises_intervals_total} intervals, {s.noises_samples_total:,} samples ({s.noises_fraction_of_records*100:.2f}% of record samples)")
            else:
                print(f"Noises: {s.noises_intervals_total} intervals, {s.noises_samples_total:,} samples")
            if s.flags_totals:
                top_flags = list(s.flags_totals.items())[:8]
                print("Top flags (count of records containing flag): " + ", ".join([f"{k}={v}" for k, v in top_flags]))
            if s.rpeak_ann_totals:
                top_ann = list(s.rpeak_ann_totals.items())[:8]
                print("Top R-peak annotations (total count): " + ", ".join([f"{k}={v}" for k, v in top_ann]))
            print()

        all_keys = {}
        for r in record_rows:
            for k in r.json_keys:
                all_keys[k] = all_keys.get(k, 0) + 1
        if all_keys:
            print("=== JSON keys coverage across records ===")
            for k, v in sorted(all_keys.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"{k:24s} : {v}/{len(record_rows)}")

        return record_rows, seq_rows, fs

    if input_path.is_dir():
        sequences = discover_sequences_dir(input_path)
        record_rows, seq_rows, fs_final = run_analysis(
            sequences=sequences,
            list_pairs_fn=lambda seq: list_recording_pairs_dir(input_path, seq),
            load_json_fn=_safe_json_load_path,
            file_size_fn=lambda p: Path(p).stat().st_size,
            merged_size_fn=lambda seq: (input_path / seq / "merged-sequence.bin").stat().st_size,
            gaps_loader_fn=lambda seq: _safe_json_load_path(input_path / seq / "gaps.json"),
            source_label=str(input_path),
        )
    else:
        with zipfile.ZipFile(input_path) as zf:
            sequences = discover_sequences(zf)
            record_rows, seq_rows, fs_final = run_analysis(
                sequences=sequences,
                list_pairs_fn=lambda seq: list_recording_pairs(zf, seq),
                load_json_fn=lambda member: _safe_json_load(zf, str(member)),
                file_size_fn=lambda member: zf.getinfo(str(member)).file_size,
                merged_size_fn=lambda seq: zf.getinfo(f"{seq}/merged-sequence.bin").file_size,
                gaps_loader_fn=lambda seq: _safe_json_load(zf, f"{seq}/gaps.json"),
                source_label=str(input_path),
            )

    report = {
        "source_path": str(input_path),
        "fs_hz": fs_final,
        "sequences": [asdict(s) for s in seq_rows],
        "records": [asdict(r) for r in record_rows],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    df_records = pd.DataFrame([asdict(r) for r in record_rows])
    df_sequences = pd.DataFrame([asdict(s) for s in seq_rows])

    if not df_records.empty:
        df_records["flags"] = df_records["flags"].apply(lambda x: "|".join(x) if isinstance(x, list) else "")
        df_records["rpeak_ann_counts"] = df_records["rpeak_ann_counts"].apply(lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else "{}")
        df_records["json_keys"] = df_records["json_keys"].apply(lambda x: "|".join(x) if isinstance(x, list) else "")
        # do not include JSON path column; instead include a boolean flag if JSON parsed fine
        df_records["json_ok"] = df_records["json_ok"].astype(bool)

        record_cols = [
            "sequence_id",
            "recording_id",
            "channel_count",
            "user_id",
            "bin_bytes",
            "samples",
            "duration_s",
            "flags_count",
            "flags",
            "rpeaks_count",
            "rpeaks_first",
            "rpeaks_last",
            "rpeak_ann_counts",
            "json_ok",
            "noises_count",
            "noises_samples",
            "noises_fraction",
            "has_comment",
            "json_keys",
        ]
        df_records = df_records[record_cols]
        
    # print("\ndf_records preview:")
    # print(df_records.head().to_string())
    
    if not df_sequences.empty:
        df_sequences["flags_totals"] = df_sequences["flags_totals"].apply(lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else "{}")
        df_sequences["rpeak_ann_totals"] = df_sequences["rpeak_ann_totals"].apply(lambda d: json.dumps(d, ensure_ascii=False) if isinstance(d, dict) else "{}")

    df_records.to_csv(out_dir / "records.csv", index=False)
    df_sequences.to_csv(out_dir / "sequences.csv", index=False)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Analyze ECG sequences in a ZIP or extracted folder.")
    p.add_argument("zip_path", type=Path, help="Path to .zip file or extracted directory")
    p.add_argument("--out", type=Path, default=Path("ecg_zip_report"), help="Output directory (default: ecg_zip_report)")
    p.add_argument("--fs", type=int, required=True, help="Sampling frequency in Hz (required). Inference disabled.")
    args = p.parse_args(argv)

    if not args.zip_path.exists():
        print(f"ERROR: zip_path not found: {args.zip_path}", file=sys.stderr)
        return 2

    try:
        analyze(args.zip_path, args.out, args.fs)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# conda activate ITP259
# python3 analyze_ecg_zip_structure.py user-65955b5f50e02b125d4998ad --fs 200 --out ecg_zip_report

# python 0_COLLECT_ZIVE_DATA/analyze_ecg_zip_structure_refactered.py data.zip --fs 256 --out ecg_zip_report
