
"""
Calculate noise statistics on ecg_start intervals based on outliers, rdropouts, and motion

https://chatgpt.com/c/69a30735-d4d4-8397-9da9-18003b3adf6f

class DenoisingPipelineResult:
    ecg_orig: np.ndarray
    ecg_start: np.ndarray
    ecg_denoised: np.ndarray
    map_gaps: IndexMap
    map_outliers: IndexMap
    map_rdropouts: IndexMap
    map_motions: IndexMap
    gaps_indices: List[Interval]
    outliers_indices_start: List[Interval]
    rdropouts_indices_nout: List[Interval]
    motions_indices_nrd: List[Interval]
    projected_to_orig: Dict[str, List[Interval]]
    projected_to_start: Dict[str, List[Interval]]


Usage:
pipe = ECGDenoisingPipeline(cfg)
res = pipe.run(x, gaps_indices=gaps_indices)
stats = calc_noise_stats_from_result(res)
print(stats["out"], stats["rdr"], stats["noi"], stats["tp_pct"])

"""


from __future__ import annotations

from typing import Any, Iterable, List, Tuple, Dict, Optional


def _interval_to_pair(iv: Any) -> Tuple[int, int]:
    """
    Convert various interval representations to (start, end) inclusive.
    Supports:
      - tuple/list: (start, end)
      - objects with attributes: start/end, begin/end, left/right, i0/i1
      - dict-like: {"start":..., "end":...} etc.
    """
    # tuple/list
    if isinstance(iv, (tuple, list)) and len(iv) >= 2:
        return int(iv[0]), int(iv[1])

    # dict-like
    if isinstance(iv, dict):
        for a, b in (("start", "end"), ("begin", "end"), ("left", "right"), ("i0", "i1"), ("s", "e")):
            if a in iv and b in iv:
                return int(iv[a]), int(iv[b])

    # attribute-based
    for a, b in (("start", "end"), ("begin", "end"), ("left", "right"), ("i0", "i1"), ("s", "e")):
        if hasattr(iv, a) and hasattr(iv, b):
            return int(getattr(iv, a)), int(getattr(iv, b))

    raise TypeError(f"Unsupported Interval type: {type(iv)!r} ({iv!r})")


def _clip_and_normalize(intervals: Iterable[Any], n: int) -> List[Tuple[int, int]]:
    """Normalize to (start,end) inclusive, ensure start<=end, clip to [0, n-1]."""
    out: List[Tuple[int, int]] = []
    if n <= 0:
        return out

    for iv in intervals or []:
        s, e = _interval_to_pair(iv)
        if s > e:
            s, e = e, s

        # clip
        if e < 0 or s > n - 1:
            continue
        s = max(s, 0)
        e = min(e, n - 1)
        if s <= e:
            out.append((s, e))
    return out


def _merge_union(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent inclusive intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:  # overlap or adjacent
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _total_len(intervals: List[Tuple[int, int]]) -> int:
    """Total number of samples covered by inclusive intervals."""
    return sum(e - s + 1 for s, e in intervals)


def _get_projected_to_start(res: Any, key_candidates: Iterable[str]) -> List[Any]:
    """
    Fetch projected intervals to ecg_start from res.projected_to_start.
    Tries multiple possible keys (because naming may differ across versions).
    """
    proj = getattr(res, "projected_to_start", None)
    if not isinstance(proj, dict):
        return []
    for k in key_candidates:
        v = proj.get(k)
        if isinstance(v, list):
            return v
    return []


def calc_noise_stats_from_result(res: Any) -> Dict[str, Any]:
    """
    Returns:
      {
        "out": <count outlier intervals on ecg_start>,
        "rdr": <count rdropout intervals projected to ecg_start>,
        "noi": <count motion intervals projected to ecg_start>,
        "tp_pct": <percent of noisy samples in ecg_start (union of all types)>,
        "tp_samples": <union noisy samples count>,
        "n_start": <len(ecg_start)>,
        "union_intervals": <merged union intervals on ecg_start>
      }
    """
    n_start = len(getattr(res, "ecg_start"))
    if n_start <= 0:
        return {
            "out": 0, "rdr": 0, "noi": 0,
            "tp_pct": 0.0, "tp_samples": 0, "n_start": 0,
            "union_intervals": [],
        }

    # Outliers are already on ecg_start per your pipeline description
    outliers_start_raw = getattr(res, "outliers_indices_start", []) or []
    print(f"Raw outliers on ecg_start: {outliers_start_raw}")

    # rdropouts/motions must be projected to ecg_start
    # Try a few common key spellings to avoid “it works only on my version” issues.
    rdropouts_start_raw = _get_projected_to_start(res, ["rdropouts", "rdropouts", "dropouts", "rdr"])
    print(f"Raw rdropouts projected to ecg_start: {rdropouts_start_raw}")
    motions_start_raw   = _get_projected_to_start(res, ["motions", "motion", "noi", "noise_motions"])
    print(f"Raw motions projected to ecg_start: {motions_start_raw}")

    outliers_start = _clip_and_normalize(outliers_start_raw, n_start)
    rdropouts_start = _clip_and_normalize(rdropouts_start_raw, n_start)
    motions_start = _clip_and_normalize(motions_start_raw, n_start)

    # Counts of intervals (not merged) as requested
    out_cnt = len(outliers_start)
    rdr_cnt = len(rdropouts_start)
    noi_cnt = len(motions_start)

    # Overall noise percent on ecg_start: union (overlaps counted once)
    all_intervals = outliers_start + rdropouts_start + motions_start
    union = _merge_union(all_intervals)
    tp_samples = _total_len(union)
    tp_pct = 100.0 * tp_samples / n_start

    return {
        "out": out_cnt,
        "rdr": rdr_cnt,
        "noi": noi_cnt,
        "tp_pct": tp_pct,
        "tp_samples": tp_samples,
        "n_start": n_start,
        "union_intervals": union,
    }