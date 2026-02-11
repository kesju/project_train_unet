#!/usr/bin/env python3
"""
Surenka įrašų statistiką iš nurodyto katalogo ARBA .zip archyvo.

Kiekvienas įrašas turi turėti JSON metaduomenų failą ir atitinkamą duomenų failą.
Duomenų failai gali būti:
  - .bin / trijų skaitmenų plėtiniai (size//4 imčių) / failas be papildomo plėtinio, jei JSON yra <id>.<ddd>.json
  - .npy (np.load(...).size)

Palaiko 2 JSON schemas rpeaks / rpeakAnnotationCounts laukams:
  A) "rpeaks" yra sąrašas: [{"sampleIndex": int, "annotationValue": str}, ...]
     "rpeakAnnotationCounts" yra plokščias dict: {"N": 123, "S": 4, ...}
  B) "rpeaks" yra dict su variantais: {"human":[...], "ml":[...], "merged":[...], "__info": "..."}
     "rpeakAnnotationCounts" yra analogiškas dict su variantais: {"human":{...}, "ml":{...}, "merged":{...}, "__info":"..."}

Jei yra "merged" – jis laikomas pagrindiniu (all peaks) ir naudojamas santraukai.

Run examples:
  Folder:
    python3 analyze_records_2026.py /path/to/user-.../recordings --fs 200 --out records_summary
  Zip:
    python3 analyze_records_2026.py recordings.zip --fs 200 --out records_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Dict, Tuple, Union, Any, Callable, Optional, List
from dataclasses import asdict, dataclass
import json
import io
import zipfile

import numpy as np
import pandas as pd


JsonLikeRef = Union[str, Path]

# -----------------------------
# Data models
# -----------------------------
@dataclass
class RecordSummary:
    sequenceId: str
    recordingId: Optional[str]
    file_name: str
    channel_count: Optional[int]
    user_id: Optional[str]

    bin_bytes: int
    samples: int
    duration_s: Optional[float]

    flags_count: int
    flags: List[str]

    rpeaks_count: int
    ann_n_count: int
    ann_s_count: int
    ann_v_count: int
    ann_u_count: int

    ml_s_count: int
    ml_v_count: int
    ml_u_count: int

    json_ok: bool

    ml_noises_count: int
    ml_noises_fraction: Optional[float]

    annotated_ml_noises_count: int
    annotated_ml_noises_fraction: Optional[float]

    has_comment: bool
    json_keys_correct: bool


# -----------------------------
# Helpers: JSON parsing
# -----------------------------
def _flatten_flags(flags: Any) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, list):
        return [str(x) for x in flags]
    # unexpected type
    return [str(flags)]


def _pick_variant_dict(obj: Any, keys_preference: Tuple[str, ...] = ("merged", "human", "ml")) -> Any:
    """
    If obj is dict with variants (e.g. {"merged": ..., "ml": ..., "human": ...}),
    returns the first matching value. Otherwise returns None.
    """
    if not isinstance(obj, dict):
        return None
    for k in keys_preference:
        if k in obj:
            return obj.get(k)
    # fallback: first non-__info value
    for k, v in obj.items():
        if k == "__info":
            continue
        return v
    return None


def _extract_rpeaks_list(rpeaks: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      - list[dict]
      - dict with variants: {"merged":[...], "ml":[...], "human":[...]}
    """
    if isinstance(rpeaks, list):
        return [x for x in rpeaks if isinstance(x, dict)]
    if isinstance(rpeaks, dict):
        v = _pick_variant_dict(rpeaks, ("merged", "human", "ml"))
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_ann_counts(ann_counts: Any) -> Dict[str, int]:
    """
    Supports:
      - flat dict: {"N": 10, "S": 2, ...}
      - variant dict: {"merged": {"N":10}, "ml": {"U":7}, ...}
    Returns cleaned mapping for the chosen variant.
    """
    if not isinstance(ann_counts, dict):
        return {}

    # Variant dict?
    if any(isinstance(v, dict) for k, v in ann_counts.items() if k != "__info"):
        v = _pick_variant_dict(ann_counts, ("merged", "human", "ml"))
        if isinstance(v, dict):
            out: Dict[str, int] = {}
            for k, val in v.items():
                if k == "__info":
                    continue
                if isinstance(val, int):
                    out[str(k)] = int(val)
                elif isinstance(val, (float, np.floating)) and float(val).is_integer():
                    out[str(k)] = int(val)
            return out
        return {}

    # Flat dict
    out2: Dict[str, int] = {}
    for k, v in ann_counts.items():
        if k == "__info":
            continue
        if isinstance(v, int):
            out2[str(k)] = int(v)
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out2[str(k)] = int(v)
    return out2


def _extract_rpeaks_list_for_variant(rpeaks: Any, variant: str) -> List[Dict[str, Any]]:
    """Extract rpeaks list for a specific variant (e.g., 'ml').

    Returns empty list if structure is old schema (list) or the variant is missing.
    """
    if isinstance(rpeaks, dict):
        v = rpeaks.get(variant)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_ann_counts_for_variant(ann_counts: Any, variant: str) -> Dict[str, int]:
    """Extract rpeakAnnotationCounts mapping for a specific variant (e.g., 'ml').

    Returns empty dict if structure is old schema (flat dict) or the variant is missing.
    """
    if not isinstance(ann_counts, dict):
        return {}
    v = ann_counts.get(variant)
    if not isinstance(v, dict):
        return {}
    out: Dict[str, int] = {}
    for k, val in v.items():
        if k == "__info":
            continue
        if isinstance(val, int):
            out[str(k)] = int(val)
        elif isinstance(val, (float, np.floating)) and float(val).is_integer():
            out[str(k)] = int(val)
    return out



def _summarize_rpeaks(rpeaks: Any) -> Tuple[int, Optional[int], Optional[int], Dict[str, int]]:
    """
    Returns (count, first_idx, last_idx, annotation_counts) for the chosen variant.
    """
    lst = _extract_rpeaks_list(rpeaks)
    if not lst:
        return 0, None, None, {}

    idxs: List[int] = []
    ann: Dict[str, int] = {}
    for rp in lst:
        si = rp.get("sampleIndex")
        av = rp.get("annotationValue")
        if isinstance(si, int):
            idxs.append(si)
        elif isinstance(si, (float, np.floating)) and float(si).is_integer():
            idxs.append(int(si))

        if isinstance(av, str):
            ann[av] = ann.get(av, 0) + 1
        elif av is not None:
            s = str(av)
            ann[s] = ann.get(s, 0) + 1

    if not idxs:
        return len(lst), None, None, ann

    idxs.sort()
    return len(lst), idxs[0], idxs[-1], ann


def _sum_noise_samples(noises: Any, fs: Optional[int]) -> Tuple[int, int]:
    """
    Returns (interval_count, total_samples_covered).

    Supports list items like:
      - {"startIndex": x, "endIndex": y}
      - {"startSample": x, "endSample": y}
      - {"startTime": t0, "endTime": t1}  (seconds -> samples via fs)
      - [startIndex, endIndex] or (startIndex, endIndex)
    Also supports dict with variants (merged/human/ml) similarly to rpeaks.
    """
    # Variant dict?
    if isinstance(noises, dict):
        v = _pick_variant_dict(noises, ("merged", "human", "ml"))
        noises = v

    if not isinstance(noises, list):
        return 0, 0

    cnt = 0
    total = 0

    for it in noises:
        a = b = None

        if isinstance(it, dict):
            a = it.get("startIndex", it.get("startSample"))
            b = it.get("endIndex", it.get("endSample"))

            if (a is None or b is None) and fs:
                t0 = it.get("startTime")
                t1 = it.get("endTime")
                if isinstance(t0, (int, float, np.floating)) and isinstance(t1, (int, float, np.floating)) and t1 > t0:
                    # convert seconds -> samples
                    a = int(round(float(t0) * fs))
                    b = int(round(float(t1) * fs))

        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            a, b = it[0], it[1]

        # validate
        if isinstance(a, (float, np.floating)) and float(a).is_integer():
            a = int(a)
        if isinstance(b, (float, np.floating)) and float(b).is_integer():
            b = int(b)

        if isinstance(a, int) and isinstance(b, int) and b > a:
            cnt += 1
            total += (b - a)

    return cnt, total


# -----------------------------
# I/O adapters (FS and ZIP)
# -----------------------------
def _safe_json_load_path(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_fs(path: Path) -> Any:
    return _safe_json_load_path(path)


def file_size_fs(path: Path) -> int:
    return path.stat().st_size


def sample_count_fs(path: Path, ext: str) -> int:
    ext = ext.lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(getattr(arr, "size", 0))
    # .bin, .ddd, or "raw" -> int32 samples
    return int(path.stat().st_size // 4)


def load_json_zip(zf: zipfile.ZipFile, member: str) -> Any:
    with zf.open(member) as f:
        return json.load(f)


def file_size_zip(zf: zipfile.ZipFile, member: str) -> int:
    return int(zf.getinfo(member).file_size)


def sample_count_zip(zf: zipfile.ZipFile, member: str, ext: str) -> int:
    ext = (ext or "").lower()
    if ext == ".npy":
        # zipfile streams are not reliably seekable for np.load -> read into BytesIO
        raw = zf.read(member)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return int(getattr(arr, "size", 0))
    # .bin / .ddd / raw -> int32 samples
    return int(file_size_zip(zf, member) // 4)


# -----------------------------
# Core logic (schema-agnostic)
# -----------------------------
def summarize_record(
    sequenceId: str,
    file_name: str,
    json_ref: JsonLikeRef,
    data_ref: JsonLikeRef,
    data_ext: str,
    *,
    load_json_fn: Callable[[JsonLikeRef], Any],
    file_size_fn: Callable[[JsonLikeRef], int],
    sample_count_fn: Callable[[JsonLikeRef, str], int],
    fs: int,
) -> Tuple[RecordSummary, Dict[str, Any]]:
    """
    Compute per-record statistics and return both RecordSummary and dict.
    The helper is reusable: pass your own I/O functions for FS or ZIP.
    """
    try:
        meta = load_json_fn(json_ref)
        json_ok = True
    except Exception as exc:
        print(f"WARN: failed to load JSON for {sequenceId}/{file_name}: {exc}")
        meta = {}
        json_ok = False

    bsz = int(file_size_fn(data_ref))
    samples = int(sample_count_fn(data_ref, data_ext))

    channel_count = meta.get("channelCount") if isinstance(meta, dict) else None
    user_id = meta.get("userId") if isinstance(meta, dict) else None
    recordingId = meta.get("recordingId") if isinstance(meta, dict) else None
    flags_list = _flatten_flags(meta.get("flags") if isinstance(meta, dict) else None)

    # rpeaks + annotation counts (support old and new schema)
    rpeaks_any = meta.get("rpeaks") if isinstance(meta, dict) else None
    rpk_cnt, _rpk_first, _rpk_last, rpk_ann = _summarize_rpeaks(rpeaks_any)

    ann_counts_any = meta.get("rpeakAnnotationCounts") if isinstance(meta, dict) else None
    ann_counts_clean = _extract_ann_counts(ann_counts_any)

    # If counts missing, fallback to counts derived from rpeaks list
    if not ann_counts_clean:
        ann_counts_clean = rpk_ann

    ann_n = int(ann_counts_clean.get("N", 0))
    ann_s = int(ann_counts_clean.get("S", 0))
    ann_v = int(ann_counts_clean.get("V", 0))
    ann_u = int(ann_counts_clean.get("U", 0))

    # ML-only annotation counts (S/V/U) – optional columns (new schema stores per-variant)
    ml_counts = _extract_ann_counts_for_variant(ann_counts_any, "ml")
    if not ml_counts:
        ml_rpeaks = _extract_rpeaks_list_for_variant(rpeaks_any, "ml")
        tmp: Dict[str, int] = {}
        for rp in ml_rpeaks:
            av = rp.get("annotationValue")
            if isinstance(av, str):
                tmp[av] = tmp.get(av, 0) + 1
            elif av is not None:
                s = str(av)
                tmp[s] = tmp.get(s, 0) + 1
        ml_counts = tmp

    ml_s = int(ml_counts.get("S", 0))
    ml_v = int(ml_counts.get("V", 0))
    ml_u = int(ml_counts.get("U", 0))

    # noises (indices or times)
    ann_noises = meta.get("noises_annotated") if isinstance(meta, dict) else None
    ann_nz_cnt, ann_nz_samples = _sum_noise_samples(ann_noises, fs)
    noises = meta.get("noises") if isinstance(meta, dict) else None
    ml_nz_cnt, nz_samples = _sum_noise_samples(noises, fs)

    duration_s = (samples / fs) if (fs and fs > 0 and samples > 0) else None
    ml_noises_fraction = (nz_samples / samples) * 100.0 if samples > 0 else None
    annotated_ml_noises_fraction = (ann_nz_samples / samples) * 100.0 if samples > 0 else None

    allowed_keys = {
        "channelCount",
        "comment",
        "flags",
        "noises",
        "noises_annotated",
        "recordingId",
        "rpeakAnnotationCounts",
        "rpeaks",
        "userId",
    }
    meta_keys = set(meta.keys()) if isinstance(meta, dict) else set()
    json_keys_correct = meta_keys.issubset(allowed_keys) if meta_keys else False

    rec = RecordSummary(
        sequenceId=str(sequenceId),
        recordingId=str(recordingId) if isinstance(recordingId, str) else None,
        file_name=str(file_name),
        channel_count=int(channel_count) if isinstance(channel_count, int) else None,
        user_id=str(user_id) if isinstance(user_id, str) else None,
        bin_bytes=bsz,
        samples=samples,
        duration_s=float(duration_s) if duration_s is not None else None,
        flags_count=len(flags_list),
        flags=flags_list,
        rpeaks_count=int(rpk_cnt),
        ann_n_count=ann_n,
        ann_s_count=ann_s,
        ann_v_count=ann_v,
        ann_u_count=ann_u,
        ml_s_count=int(ml_s),
        ml_v_count=int(ml_v),
        ml_u_count=int(ml_u),
        json_ok=bool(json_ok),
        ml_noises_count=int(ml_nz_cnt),
        ml_noises_fraction=float(ml_noises_fraction) if ml_noises_fraction is not None else None,
        annotated_ml_noises_count=int(ann_nz_cnt),
        annotated_ml_noises_fraction=float(annotated_ml_noises_fraction) if annotated_ml_noises_fraction is not None else None,
        has_comment=bool(meta.get("comment")) if isinstance(meta, dict) else False,
        json_keys_correct=bool(json_keys_correct),
    )

    rec_dict = asdict(rec)
    rec_dict["_noises_samples"] = nz_samples
    rec_dict["_annotated_noises_samples"] = ann_nz_samples
    return rec, rec_dict


# -----------------------------
# Matching JSON <-> data file
# -----------------------------
def find_data_file_fs(json_path: Path) -> Tuple[Path, str] | None:
    """Return matching data file and extension for a given JSON (filesystem).

    Supports:
      - <id>.json           -> <id>.npy / <id>.bin / <id>.<ddd> / <id>  (rare)
      - <id>.<ddd>.json     -> <id>.<ddd>
    """
    stem = json_path.stem
    rec_dir = json_path.parent

    # Special case: JSON is '<id>.<ddd>.json' and data file is '<id>.<ddd>'
    exact = rec_dir / stem
    if exact.exists() and exact.is_file() and exact.suffix[1:].isdigit() and len(exact.suffix) == 4:
        return exact, exact.suffix

    candidates = [
        rec_dir / f"{stem}.npy",
        rec_dir / f"{stem}.bin",
        rec_dir / f"{stem}",  # just in case
    ]
    candidates += sorted(
        [p for p in rec_dir.glob(f"{stem}.*") if p.suffix[1:].isdigit() and len(p.suffix) == 4]
    )
    for p in candidates:
        if p.exists() and p.is_file():
            return p, p.suffix
    return None


def find_data_file_zip(json_member: str, name_set: set[str]) -> Tuple[str, str] | None:
    """Return matching data member and extension for a given JSON inside ZIP."""
    pp = PurePosixPath(json_member)
    if pp.suffix.lower() != ".json":
        return None
    rec_dir = pp.parent
    stem = pp.stem  # drops .json, keeps e.g. 1761433.128

    # Special case: JSON is '<id>.<ddd>.json' and data is '<id>.<ddd>'
    exact = str(rec_dir / stem)
    ex_suffix = PurePosixPath(exact).suffix
    if exact in name_set and ex_suffix[1:].isdigit() and len(ex_suffix) == 4:
        return exact, ex_suffix

    # candidates: .npy, .bin, plain, then any .ddd
    candidates = [
        str(rec_dir / f"{stem}.npy"),
        str(rec_dir / f"{stem}.bin"),
        str(rec_dir / f"{stem}"),
    ]
    # three-digit extensions in same folder
    prefix = str(rec_dir / stem) + "."
    for n in sorted(name_set):
        if n.startswith(prefix):
            suf = PurePosixPath(n).suffix
            if suf[1:].isdigit() and len(suf) == 4:
                candidates.append(n)

    for c in candidates:
        if c in name_set:
            return c, PurePosixPath(c).suffix
    return None


def infer_sequence_id_fs(json_path: Path) -> str:
    # expected layout: <seq>/recordings/<id>.json
    return json_path.parent.parent.name if json_path.parent.name == "recordings" else json_path.parent.name


def infer_sequence_id_zip(json_member: str, fallback: str) -> str:
    pp = PurePosixPath(json_member)
    parts = list(pp.parts)
    if "recordings" in parts:
        idx = parts.index("recordings")
        if idx >= 1:
            return parts[idx - 1]
    # fallback: parent folder name if exists
    if len(parts) >= 2:
        return parts[-2]
    return fallback


def resolve_out_paths(out_arg, default_dir: Path, default_stem: str = "records_summary"):
    """
    Returns (csv_path, xlsx_path) based on --out.
    --out can be:
      - None -> default_dir/default_stem.{csv,xlsx}
      - a path with or without suffix -> treated as base path/stem
      - a directory -> file name uses default_stem inside that directory
    """
    if out_arg:
        p = Path(out_arg)
    else:
        p = default_dir / default_stem

    if p.exists() and p.is_dir():
        p = p / default_stem
    elif str(p).endswith(("/", "\\")):  # directory-like string
        p = p / default_stem

    base = p.with_suffix("")
    return base.with_suffix(".csv"), base.with_suffix(".xlsx")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize ECG records in a folder or .zip archive.")
    ap.add_argument("data_path", type=Path, help="Root folder containing recordings/*.json OR a .zip with such structure")
    ap.add_argument("--fs", type=int, required=True, help="Sampling frequency in Hz")
    ap.add_argument("--out", type=Path, default=None, help="Output base path (default: <input_dir>/records_summary.{csv,xlsx})")
    args = ap.parse_args()

    in_path: Path = args.data_path

    records: list[Dict[str, Any]] = []

    if in_path.is_file() and in_path.suffix.lower() == ".zip":
        default_dir = in_path.parent
        fallback_seq = in_path.stem

        with zipfile.ZipFile(in_path, "r") as zf:
            names = set(zf.namelist())
            json_members = sorted([n for n in names if n.lower().endswith(".json") and not n.endswith("/")])

            for jm in json_members:
                match = find_data_file_zip(jm, names)
                if match is None:
                    print(f"WARN: no data file found for {jm}")
                    continue
                data_member, data_ext = match
                seq_id = infer_sequence_id_zip(jm, fallback_seq)
                file_name = PurePosixPath(data_member).name

                rec, rec_dict = summarize_record(
                    sequenceId=seq_id,
                    file_name=file_name,
                    json_ref=jm,
                    data_ref=data_member,
                    data_ext=data_ext,
                    load_json_fn=lambda ref, _zf=zf: load_json_zip(_zf, str(ref)),
                    file_size_fn=lambda ref, _zf=zf: file_size_zip(_zf, str(ref)),
                    sample_count_fn=lambda ref, ext, _zf=zf: sample_count_zip(_zf, str(ref), ext),
                    fs=args.fs,
                )
                records.append(rec_dict)

        data_dir_label = in_path
    else:
        data_dir = in_path
        if not data_dir.exists() or not data_dir.is_dir():
            raise SystemExit(f"Input must be a folder or a .zip: {in_path}")
        default_dir = data_dir

        json_files = sorted(data_dir.rglob("*.json"))
        for jp in json_files:
            match = find_data_file_fs(jp)
            if match is None:
                print(f"WARN: no data file found for {jp}")
                continue
            data_path, data_ext = match
            seq_id = infer_sequence_id_fs(jp)
            file_name = data_path.name

            rec, rec_dict = summarize_record(
                sequenceId=seq_id,
                file_name=file_name,
                json_ref=jp,
                data_ref=data_path,
                data_ext=data_ext,
                load_json_fn=load_json_fs,
                file_size_fn=file_size_fs,
                sample_count_fn=sample_count_fs,
                fs=args.fs,
            )
            records.append(rec_dict)

        data_dir_label = data_dir

    df = pd.DataFrame(records)

    if df.empty:
        print("No records found.")
        return 0

    # 1) Add row number (nr)
    df = df.copy()
    df.insert(0, "nr", range(1, len(df) + 1))

    # 2) Select only the columns you want (skip any missing ones safely)
    cols = [
        "nr", "file_name", "recordingId", "user_id", "samples", "duration_s",
        "rpeaks_count", "ann_n_count", "ann_s_count", "ann_v_count", "ann_u_count",
        "ml_s_count", "ml_v_count", "ml_u_count",
        "annotated_ml_noises_count", "annotated_ml_noises_fraction",
        "ml_noises_count", "ml_noises_fraction",
        "flags",
    ]
    cols = [c for c in cols if c in df.columns]
    df_sel = df.loc[:, cols]

    # 3) Short display names
    col_short = {
        "recordingId": "rec_id",
        "user_id": "uid",
        "duration_s": "dur_s",
        "rpeaks_count": "rpk_cnt",
        "annotated_ml_noises_count": "ann_nz_cnt",
        "annotated_ml_noises_fraction": "ann_nz_frac",
        "ml_noises_count": "ml_nz_cnt",
        "ml_noises_fraction": "ml_nz_frac",
        "ann_n_count": "annN",
        "ann_s_count": "annS",
        "ann_v_count": "annV",
        "ann_u_count": "annU",
        "ml_s_count": "mlS",
        "ml_v_count": "mlV",
        "ml_u_count": "mlU",
    }
    df_print = df_sel.rename(columns=col_short)  # type: ignore

    # 4) Insert external-annotating helper columns
    # i = list(df_print.columns).index("nr") + 1
    # df_print.insert(i, "FOR_EXTERNAL_ANNOTATING", "")
    # df_print.insert(i + 1, "NOISES_ANNOTATED", "")

    # 5) Save CSV and XLSX
    csv_path, xlsx_path = resolve_out_paths(args.out, default_dir, "records_summary")
    df_print.to_csv(csv_path, index=False)

    df_out = df_print.copy()

    # Ensure numeric so Excel can format properly
    for c in ["dur_s", "ann_nz_frac", "ml_nz_frac"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sheet = "summary"
        df_out.to_excel(writer, index=False, sheet_name=sheet)
        ws = writer.book[sheet]

        fmt_map = {
            "dur_s": "0.00",          # .2f
            "ann_nz_frac": '0.0"%"',  # .1f
            "ml_nz_frac": '0.0"%"',      # .1f
        }

        headers = [cell.value for cell in ws[1]]
        col_idx = {name: i + 1 for i, name in enumerate(headers) if name is not None}

        for name, fmt in fmt_map.items():
            if name not in col_idx:
                continue
            c = col_idx[name]
            for r in range(2, ws.max_row + 1):  # skip header
                ws.cell(row=r, column=c).number_format = fmt

    # 6) Console output
    print(f"\nInput: {data_dir_label}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print("Summary statistics:")
    print(
        df_print.to_string(
            index=False,
            float_format=None,
            formatters={
                "dur_s": lambda x: f"{x:.2f}" if pd.notna(x) else "",
                "ann_nz_frac": lambda x: f"{x:.1f}%" if pd.notna(x) else "",
                "ml_nz_frac": lambda x: f"{x:.1f}%" if pd.notna(x) else "",
            }
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python3 analyze_records_2026_v3.py user-6581e16ce2b0bd5f0e7540a4/68ffa1cbad358145351671eb/recordings --fs 200 --out records_summary
python3 analyze_records_2026_v3.py /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_COLLECT_ZIVE_DATA/DUOM_2026_01_25/659ebcdd870b3d1d6e30bb61 --fs 200 --out records_summary

"""