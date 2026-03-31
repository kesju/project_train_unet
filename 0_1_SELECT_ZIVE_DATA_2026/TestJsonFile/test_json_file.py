from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd


@dataclass
class RecordSummary:
    sequenceId: str
    recordingId: Optional[str]
    basename: str
    channel_count: Optional[int]
    user_id: Optional[str]

    bin_bytes: int
    samples: int
    duration_s: Optional[float]

    flags_count: int
    flags: List[str]

    rpeaks_count: int
    h_n_count: int
    h_s_count: int
    h_v_count: int
    h_u_count: int

    ml_s_count: int
    ml_v_count: int
    ml_u_count: int

    json_ok: bool

    ml_noises_count: int
    ml_noises_samples: int
    ml_noises_fraction: Optional[float]  # percent 0..100

    h_noises_count: int
    h_noises_samples: int
    h_noises_fraction: Optional[float]  # percent 0..100

    cmt: Optional[str]
    json_keys_correct: bool
    notes: Optional[str] = None


def _flatten_flags(flags: Any) -> List[str]:
    if flags is None:
        return []
    if isinstance(flags, list):
        return [str(x) for x in flags]
    return [str(flags)]


def _pick_variant_dict(obj: Any, keys_preference: Tuple[str, ...] = ("merged", "human", "ml")) -> Any:
    if not isinstance(obj, dict):
        return None
    for k in keys_preference:
        if k in obj:
            return obj.get(k)
    for k, v in obj.items():
        if k == "__info":
            continue
        return v
    return None


def _extract_rpeaks_list(rpeaks: Any) -> List[Dict[str, Any]]:
    if isinstance(rpeaks, list):
        return [x for x in rpeaks if isinstance(x, dict)]
    if isinstance(rpeaks, dict):
        v = _pick_variant_dict(rpeaks, ("merged", "human", "ml"))
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_rpeaks_list_for_variant(rpeaks: Any, variant: str) -> List[Dict[str, Any]]:
    if isinstance(rpeaks, dict):
        v = rpeaks.get(variant)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


def _extract_mrk_counts_for_variant(h_counts: Any, variant: str) -> Dict[str, int]:
    if not isinstance(h_counts, dict):
        return {}
    v = h_counts.get(variant)
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


def _extract_mrk_counts(h_counts: Any) -> Dict[str, int]:
    if not isinstance(h_counts, dict):
        return {}
    # Variant dict?
    if any(isinstance(v, dict) for k, v in h_counts.items() if k != "__info"):
        v = _pick_variant_dict(h_counts, ("merged", "human", "ml"))
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
    for k, v in h_counts.items():
        if k == "__info":
            continue
        if isinstance(v, int):
            out2[str(k)] = int(v)
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out2[str(k)] = int(v)
    return out2


def _summarize_rpeaks(rpeaks: Any) -> Tuple[int, Optional[int], Optional[int], Dict[str, int]]:
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
    # Variant dict?
    if isinstance(noises, dict):
        noises = _pick_variant_dict(noises, ("merged", "human", "ml"))
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
                if (
                    isinstance(t0, (int, float, np.floating))
                    and isinstance(t1, (int, float, np.floating))
                    and t1 > t0
                ):
                    a = int(round(float(t0) * fs))
                    b = int(round(float(t1) * fs))
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            a, b = it[0], it[1]

        if isinstance(a, (float, np.floating)) and float(a).is_integer():
            a = int(a)
        if isinstance(b, (float, np.floating)) and float(b).is_integer():
            b = int(b)

        if isinstance(a, int) and isinstance(b, int) and b > a:
            cnt += 1
            total += (b - a)
    return cnt, total


def load_json_fs(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def file_size_fs(p: Path) -> int:
    return int(p.stat().st_size)


def sample_count_fs(p: Path, ext: str) -> int:
    ext = (ext or "").lower()
    if ext == ".npy":
        arr = np.load(p, allow_pickle=False)
        return int(getattr(arr, "size", 0))
    # Default: assume 4 bytes per sample (int32)
    return int(file_size_fs(p) // 4)


def summarize_record(
    sequenceId: str,
    basename: str,
    json_ref: Path,
    data_ref: Path,
    data_ext: str,
    *,
    load_json_fn: Callable[[Path], Any],
    file_size_fn: Callable[[Path], int],
    sample_count_fn: Callable[[Path, str], int],
    fs: int,
) -> Tuple[RecordSummary, Dict[str, Any]]:
    try:
        meta = load_json_fn(json_ref)
        json_ok = True
    except Exception as exc:
        print(f"WARN: failed to load JSON for {sequenceId}/{basename}: {exc}")
        meta = {}
        json_ok = False

    bsz = int(file_size_fn(data_ref))
    samples = int(sample_count_fn(data_ref, data_ext))

    channel_count = meta.get("channelCount") if isinstance(meta, dict) else None
    user_id = meta.get("userId") if isinstance(meta, dict) else None
    recordingId = meta.get("recordingId") if isinstance(meta, dict) else None
    flags_list = _flatten_flags(meta.get("flags") if isinstance(meta, dict) else None)
    comment = meta.get("comment") if isinstance(meta, dict) else None

    rpeaks_any = meta.get("rpeaks") if isinstance(meta, dict) else None
    rpeaks_count, *_ = _summarize_rpeaks(rpeaks_any)

    mrk_counts_any = meta.get("rpeakAnnotationCounts") if isinstance(meta, dict) else None

    # H counts preference: merged -> human -> derived -> flat
    def _counts_for_h_columns() -> Tuple[Dict[str, int], bool]:
        for var in ("merged", "human"):
            d = _extract_mrk_counts_for_variant(mrk_counts_any, var)
            if d:
                return d, True

        for var in ("merged", "human"):
            rp_list = _extract_rpeaks_list_for_variant(rpeaks_any, var)
            if rp_list:
                tmp: Dict[str, int] = {}
                for rp in rp_list:
                    av = rp.get("annotationValue")
                    if isinstance(av, str):
                        tmp[av] = tmp.get(av, 0) + 1
                    elif av is not None:
                        tmp[str(av)] = tmp.get(str(av), 0) + 1
                if tmp:
                    return tmp, True

        flat = _extract_mrk_counts(mrk_counts_any)
        if flat:
            return flat, True

        return {}, False

    h_counts_primary, has_h_counts = _counts_for_h_columns()
    h_n = int(h_counts_primary.get("N", 0)) if has_h_counts else 0
    h_s = int(h_counts_primary.get("S", 0)) if has_h_counts else 0
    h_v = int(h_counts_primary.get("V", 0)) if has_h_counts else 0
    h_u = int(h_counts_primary.get("U", 0)) if has_h_counts else 0

    # ML counts: explicit "ml" first, else derive from ml rpeaks
    ml_counts = _extract_mrk_counts_for_variant(mrk_counts_any, "ml")
    if not ml_counts:
        ml_rpeaks = _extract_rpeaks_list_for_variant(rpeaks_any, "ml")
        tmp2: Dict[str, int] = {}
        for rp in ml_rpeaks:
            av = rp.get("annotationValue")
            if isinstance(av, str):
                tmp2[av] = tmp2.get(av, 0) + 1
            elif av is not None:
                tmp2[str(av)] = tmp2.get(str(av), 0) + 1
        ml_counts = tmp2

    ml_s = int(ml_counts.get("S", 0))
    ml_v = int(ml_counts.get("V", 0))
    ml_u = int(ml_counts.get("U", 0))

    h_noises = meta.get("noises_annotated") if isinstance(meta, dict) else None
    h_nz_cnt, h_nz_samples = _sum_noise_samples(h_noises, fs)

    ml_noises = meta.get("noises") if isinstance(meta, dict) else None
    ml_nz_cnt, ml_nz_samples = _sum_noise_samples(ml_noises, fs)

    duration_s = (samples / fs) if (fs and fs > 0 and samples > 0) else None

    # IMPORTANT: fractions are ALWAYS percentages (0..100)
    ml_noises_fraction = (ml_nz_samples / samples) * 100.0 if samples > 0 else None
    h_noises_fraction = (h_nz_samples / samples) * 100.0 if samples > 0 else None

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
        basename=str(basename),
        channel_count=int(channel_count) if isinstance(channel_count, int) else None,
        user_id=str(user_id) if isinstance(user_id, str) else None,
        bin_bytes=bsz,
        samples=samples,
        duration_s=float(duration_s) if duration_s is not None else None,
        cmt=str(comment) if isinstance(comment, str) else None,
        flags_count=len(flags_list),
        flags=flags_list,
        rpeaks_count=int(rpeaks_count),
        h_n_count=h_n,
        h_s_count=h_s,
        h_v_count=h_v,
        h_u_count=h_u,
        ml_s_count=int(ml_s),
        ml_v_count=int(ml_v),
        ml_u_count=int(ml_u),
        json_ok=bool(json_ok),
        ml_noises_count=int(ml_nz_cnt),
        ml_noises_samples=int(ml_nz_samples),
        ml_noises_fraction=float(ml_noises_fraction) if ml_noises_fraction is not None else None,
        h_noises_count=int(h_nz_cnt),
        h_noises_samples=int(h_nz_samples),
        h_noises_fraction=float(h_noises_fraction) if h_noises_fraction is not None else None,
        json_keys_correct=bool(json_keys_correct),
    )

    rec_dict = asdict(rec)
    rec_dict["_ml_noises_samples"] = int(ml_nz_samples)
    rec_dict["_h_noises_samples"] = int(h_nz_samples)
    return rec, rec_dict


def read_df_annot_from_json_old(json_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Read and process annotations from a JSON file.

    Expected JSON structure
    -----------------------
    The file is expected to contain key 'rpeaks', where each item has:
    - 'sampleIndex'
    - 'annotationValue'

    Parameters
    ----------
    json_path : Path | str
        Full path to the JSON annotation file.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ['rpeak', 'annot'],
        or None if file does not exist.

    Raises
    ------
    ValueError
        If JSON structure is invalid or required fields are missing.
    TypeError
        If input data types are invalid.
    RuntimeError
        If JSON cannot be parsed.
    """
    all_beats = {'N': 0, 'S': 1, 'V': 2, 'U': 3}

    file_path = Path(json_path).resolve()

    if not file_path.exists():
        return None

    if not file_path.is_file():
        raise ValueError(f"JSON path exists but is not a file: {file_path}")

    try:
        with open(file_path, 'r', encoding='UTF-8', errors='ignore') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file: {file_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {file_path}") from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level JSON object must be dict, got {type(data).__name__} in file {file_path}"
        )

    if "rpeaks" not in data:
        raise ValueError(f"Key 'rpeaks' not found in JSON file: {file_path}")

    rpeaks = data["rpeaks"]
    if rpeaks is None:
        raise ValueError(f"Key 'rpeaks' is None in JSON file: {file_path}")

    if not isinstance(rpeaks, list):
        raise TypeError(
            f"Key 'rpeaks' must be a list, got {type(rpeaks).__name__} in file {file_path}"
        )

    if len(rpeaks) == 0:
        return pd.DataFrame(columns=["rpeak", "annot"])

    try:
        norm = pd.json_normalize(rpeaks)
        df_norm = cast(pd.DataFrame, norm)
    except Exception as exc:
        raise RuntimeError(f"pd.json_normalize failed for file: {file_path}") from exc

    if df_norm.empty:
        return pd.DataFrame(columns=["rpeak", "annot"])

    required_cols = ["sampleIndex", "annotationValue"]
    missing_cols = [col for col in required_cols if col not in df_norm.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols} in normalized rpeaks data from file: {file_path}"
        )

    try:
        rpeak_series = pd.to_numeric(df_norm["sampleIndex"], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(
            f"Column 'sampleIndex' contains non-numeric or invalid values in file: {file_path}"
        ) from exc

    try:
        annot_mapped = df_norm["annotationValue"].map(all_beats).fillna(0).astype(int)
    except Exception as exc:
        raise ValueError(
            f"Failed to map 'annotationValue' to numeric labels in file: {file_path}"
        ) from exc

    df_annot = pd.DataFrame({
        "rpeak": rpeak_series,
        "annot": annot_mapped,
    })

    return df_annot




def read_df_annot_from_json(json_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Read and process annotations from a JSON file.

    Supported JSON formats
    ----------------------
    Old format:
        {
            ...
            "rpeaks": [
                {"sampleIndex": ..., "annotationValue": ...},
                ...
            ]
        }

    New format:
        {
            ...
            "rpeaks": {
                "__info": "...",
                "human": [...],
                "ml": [...],
                "merged": [...]
            }
        }

    For the new format, this function uses:
        rpeaks["merged"]

    Parameters
    ----------
    json_path : Path | str
        Full path to the JSON annotation file.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ['rpeak', 'annot'],
        or None if file does not exist.

    Raises
    ------
    ValueError
        If JSON structure is invalid or required fields are missing.
    TypeError
        If input data types are invalid.
    RuntimeError
        If JSON cannot be parsed or normalized.
    """
    all_beats = {'N': 0, 'S': 1, 'V': 2, 'U': 3}

    file_path = Path(json_path).expanduser().resolve()

    if not file_path.exists():
        return None

    if not file_path.is_file():
        raise ValueError(f"JSON path exists but is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="UTF-8", errors="ignore") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file: {file_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {file_path}") from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level JSON object must be dict, got {type(data).__name__} in file {file_path}"
        )

    if "rpeaks" not in data:
        raise ValueError(f"Key 'rpeaks' not found in JSON file: {file_path}")

    rpeaks_raw = data["rpeaks"]

    if rpeaks_raw is None:
        raise ValueError(f"Key 'rpeaks' is None in JSON file: {file_path}")

    # ------------------------------------------------------------------
    # Support both formats:
    # 1) old: rpeaks is a list
    # 2) new: rpeaks is a dict with keys: human, ml, merged
    # ------------------------------------------------------------------
    if isinstance(rpeaks_raw, list):
        rpeaks = rpeaks_raw

    elif isinstance(rpeaks_raw, dict):
        if "merged" not in rpeaks_raw:
            raise ValueError(
                f"New-format JSON detected, but key 'rpeaks[\"merged\"]' not found in file: {file_path}"
            )

        rpeaks = rpeaks_raw["merged"]

        if rpeaks is None:
            raise ValueError(
                f"Key 'rpeaks[\"merged\"]' is None in JSON file: {file_path}"
            )

        if not isinstance(rpeaks, list):
            raise TypeError(
                f"Key 'rpeaks[\"merged\"]' must be a list, got {type(rpeaks).__name__} in file {file_path}"
            )

    else:
        raise TypeError(
            f"Key 'rpeaks' must be either list (old format) or dict (new format), "
            f"got {type(rpeaks_raw).__name__} in file {file_path}"
        )

    if len(rpeaks) == 0:
        return pd.DataFrame(columns=["rpeak", "annot"])

    try:
        norm = pd.json_normalize(rpeaks)
        df_norm = cast(pd.DataFrame, norm)
    except Exception as exc:
        raise RuntimeError(f"pd.json_normalize failed for file: {file_path}") from exc

    if df_norm.empty:
        return pd.DataFrame(columns=["rpeak", "annot"])

    required_cols = ["sampleIndex", "annotationValue"]
    missing_cols = [col for col in required_cols if col not in df_norm.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols} in normalized rpeaks data from file: {file_path}"
        )

    try:
        rpeak_series = pd.to_numeric(df_norm["sampleIndex"], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(
            f"Column 'sampleIndex' contains non-numeric or invalid values in file: {file_path}"
        ) from exc

    try:
        annot_series = (
            df_norm["annotationValue"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(all_beats)
            .fillna(0)
            .astype(int)
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to map 'annotationValue' to numeric labels in file: {file_path}"
        ) from exc

    df_annot = pd.DataFrame({
        "rpeak": rpeak_series,
        "annot": annot_series,
    })

    return df_annot


# TESTING

# new format:
json_path_new_format = Path("1626941.468.json")
json_path_new_format = Path("1630770.348.json")

print("\nTesting new format JSON:", json_path_new_format)
with open(json_path_new_format, "r", encoding="UTF-8", errors="ignore") as f:
    data = json.load(f)

print(type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys())[:50])
    if "rpeaks" in data:
        print("type(data['rpeaks']):", type(data["rpeaks"]))
        print("len(data['rpeaks']):", len(data["rpeaks"]) if hasattr(data["rpeaks"], "__len__") else "no len")
        if isinstance(data["rpeaks"], list) and len(data["rpeaks"]) > 0:
            print("first rpeak item:", data["rpeaks"][0])




# old format:
json_path_old_format = Path("1000_1.json")            
json_path_old_format = Path("1008_7.json")            
print("\nTesting old format JSON:", json_path_old_format)
with open(json_path_old_format, "r", encoding="UTF-8", errors="ignore") as f:
    data = json.load(f)

print(type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys())[:50])
    if "rpeaks" in data:
        print("type(data['rpeaks']):", type(data["rpeaks"]))
        print("len(data['rpeaks']):", len(data["rpeaks"]) if hasattr(data["rpeaks"], "__len__") else "no len")
        if isinstance(data["rpeaks"], list) and len(data["rpeaks"]) > 0:
            print("first rpeak item:", data["rpeaks"][0])
            
print("\nTesting read_df_annot_from_json_old with old format JSON:")            
df_annot =read_df_annot_from_json_old(json_path_old_format)
print("\nDataFrame from old format JSON:")
counts = df_annot["annot"].value_counts().sort_index()
print(counts)
print(df_annot.head(20))

print
df_annot =read_df_annot_from_json(json_path_old_format)
print("\nDataFrame from old format JSON:")
counts = df_annot["annot"].value_counts().sort_index()
print(counts)
print(df_annot.head(20))

print("\nTesting read_df_annot_from_json with old format JSON:")
df_annot =read_df_annot_from_json(json_path_old_format)
print("\nDataFrame from old format JSON:")
counts = df_annot["annot"].value_counts().sort_index()
print(counts)
print(df_annot.head(20))