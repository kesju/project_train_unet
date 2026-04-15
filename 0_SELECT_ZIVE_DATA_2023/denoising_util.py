import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, TypeAlias
from pathlib import Path

Interval: TypeAlias = tuple[int, int]


def fmt_hms_from_sec(sec: float) -> str:
    # HH:MM:SS (hours can exceed 24)
    sec_i = int(round(sec))
    h, rem = divmod(sec_i, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def intervals_to_hms(intervals, fs: float, start_dt):
    """
    intervals: list/array of (i0, i1) sample indices (end exclusive or inclusive—doesn't matter for timestamps)
    returns list of strings: 'HH:MM:SS - HH:MM:SS'
    """
    if intervals is None:
        return []
    out = []
    for a, b in intervals:
        t0 = (start_dt + timedelta(seconds=float(a) / fs)).time()
        t1 = (start_dt + timedelta(seconds=float(b) / fs)).time()
        # If you prefer absolute datetime strings, replace .time() with the full datetime formatting.
        s0 = (start_dt + timedelta(seconds=float(a) / fs)).strftime("%H:%M:%S")
        s1 = (start_dt + timedelta(seconds=float(b) / fs)).strftime("%H:%M:%S")
        out.append(f"{s0} - {s1}")
    return out

def merge_spans_sec(spans_sec):
    """Sort + merge overlapping spans in seconds."""
    spans = [(float(a), float(b)) for a, b in spans_sec if b > a]
    if not spans:
        return []
    spans.sort()
    out = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]

def _overlap_len(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def window_noise_fraction(hrv_df, projected_to_orig, fs, keys=None):
    """
    Returns np.ndarray same length as hrv_df with fraction of each window covered by noise.
    hrv_df must contain 't0','t1' in seconds (ecg_orig axis).
    projected_to_orig spans are in samples (start,end) on ecg_orig axis.
    keys: None => use all categories; else e.g. {'motions','rdropouts'}.
    """
    if hrv_df.empty or not projected_to_orig:
        return np.zeros(len(hrv_df), dtype=float)

    spans_sec = []
    for k, intervals in projected_to_orig.items():
        if keys is not None and k not in keys:
            continue
        for itv in intervals:
            a, b = itv  # change to itv.start, itv.end if needed
            spans_sec.append((a / fs, b / fs))

    spans_sec = _merge_spans_sec(spans_sec)
    if not spans_sec:
        return np.zeros(len(hrv_df), dtype=float)

    t0s = hrv_df["t0"].to_numpy(dtype=float)
    t1s = hrv_df["t1"].to_numpy(dtype=float)

    frac = np.zeros(len(hrv_df), dtype=float)
    for i, (t0, t1) in enumerate(zip(t0s, t1s)):
        win_len = max(1e-12, t1 - t0)
        noise_len = 0.0
        for a, b in spans_sec:
            if b <= t0:
                continue
            if a >= t1:
                break
            noise_len += _overlap_len(t0, t1, a, b)
        frac[i] = noise_len / win_len

    return frac


def _merge_spans_sec(spans_sec):
    """spans_sec: list[(start_sec, end_sec)] -> merged, sorted, non-overlapping."""
    if not spans_sec:
        return []
    spans_sec = sorted((float(a), float(b)) for a, b in spans_sec if b > a)
    merged = [list(spans_sec[0])]
    for a, b in spans_sec[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]

def mask_rr_outside_noise(rr_t_sec, projected_to_orig, fs, keys=None):
    """
    Returns boolean mask True where rr_t_sec is OUTSIDE noise intervals.
    rr_t_sec must be sorted ascending (it is, if r_orig is sorted).
    keys: None = use all noise types, or e.g. keys={"motions","outliers"}.
    """
    if projected_to_orig is None:
        return np.ones_like(rr_t_sec, dtype=bool)

    # collect spans (in seconds)
    spans_sec = []
    for k, intervals in projected_to_orig.items():
        if (keys is not None) and (k not in keys):
            continue
        for itv in intervals:
            a, b = itv  # adjust if dataclass
            spans_sec.append((a / fs, b / fs))

    spans_sec = _merge_spans_sec(spans_sec)
    keep = np.ones_like(rr_t_sec, dtype=bool)

    # for each span, kill rr points in that time range (vectorized by slicing)
    for a_sec, b_sec in spans_sec:
        i0 = np.searchsorted(rr_t_sec, a_sec, side="left")
        i1 = np.searchsorted(rr_t_sec, b_sec, side="right")
        keep[i0:i1] = False

    return keep

# ----------------------------
#  A) IndexMap adapter + composition
# ----------------------------

def _call_indexmap(m, idx):
    """
    Try several common method names on IndexMap to map indices from "child" (after removal)
    to "parent" (before removal). Returns np.ndarray[int].
    Adjust this function if your IndexMap API differs.
    """
    idx = np.asarray(idx, dtype=np.int64)

    # Most common patterns:
    for name in ("to_parent", "to_src", "backward", "project_to_parent", "project_to_src", "map_to_parent"):
        fn = getattr(m, name, None)
        if callable(fn):
            out = fn(idx)
            return np.asarray(out, dtype=np.int64)

    # Sometimes IndexMap is callable
    if callable(m):
        out = m(idx)
        return np.asarray(out, dtype=np.int64)

    raise AttributeError(
        "Don't know how to call your IndexMap. "
        "Please update _call_indexmap() with the correct method name."
    )

from datetime import datetime, timedelta
import numpy as np

def clip_by_time_of_day(x, start_dt: datetime, fs: float,
                        clip_hm_start=(5, 30), clip_hm_end=(8, 25)):
    x = np.asarray(x)

    # Build datetimes for the same day as start_dt
    clip_start_dt = start_dt.replace(hour=clip_hm_start[0], minute=clip_hm_start[1],
                                     second=0, microsecond=0)
    clip_end_dt   = start_dt.replace(hour=clip_hm_end[0],   minute=clip_hm_end[1],
                                     second=0, microsecond=0)

    # If end is "earlier" than start, assume it goes to the next day
    if clip_end_dt <= clip_start_dt:
        clip_end_dt += timedelta(days=1)

    # Convert to sample indices
    start_sec = (clip_start_dt - start_dt).total_seconds()
    end_sec   = (clip_end_dt   - start_dt).total_seconds()

    i0 = int(round(start_sec * fs))
    i1 = int(round(end_sec   * fs))

    # Clamp to array bounds
    i0 = max(0, min(i0, len(x)))
    i1 = max(0, min(i1, len(x)))

    x_clip = x[i0:i1]  # end index is exclusive

    # print("clip_start_dt:", clip_start_dt)
    # print("clip_end_dt  :", clip_end_dt)
    # print(f"clip indexes : i0={i0}, i1={i1} (samples), duration={(i1-i0)/fs:.2f}s")

    return x_clip, i0, i1, clip_start_dt, clip_end_dt


# Example usage:
# start_dt = datetime(2025, 12, 1, 0, 0, 0)   # (day must be 1..31)
# x_clip, i0, i1, clip_start_dt, clip_end_dt = clip_by_time_of_day(x, start_dt, fs=200.0)


def find_project_root_by_name(target="S-ITP-25-9", start: Path | str | None = None) -> Path:
    """Walk up from `start` (or CWD) until a directory named `target` is found."""
    start_path = Path(start or Path.cwd()).resolve()
    for p in (start_path, *start_path.parents):
        if p.name == target:
            return p
    raise FileNotFoundError(f"Could not find '{target}' above {start_path}")

def samples_dict_to_seconds(
    d: Dict[str, List[Interval]],
    fs: float,
    ndigits: int | None = 3,  # set None to skip rounding
) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for key, intervals in d.items():
        if not intervals:
            out[key] = []
            continue
        pairs = []
        for s, e in intervals:
            s_sec, e_sec = s / fs, e / fs
            if ndigits is not None:
                s_sec, e_sec = round(s_sec, ndigits), round(e_sec, ndigits)
            pairs.append((s_sec, e_sec))
        out[key] = pairs
    return out


# ----------------------------
#  B) R-peak detection (NeuroKit2 if available, else simple fallback)
# ----------------------------

def detect_rpeaks_neurokit(ecg, fs):
    """
    Returns rpeaks indices in the input ecg array.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except ImportError:
        return None
    except Exception as e:
        # at least expose what went wrong
        raise RuntimeError(f"nk.ecg_peaks failed: {e}") from e

# ----------------------------
#  C) RR/NN cleaning
# ----------------------------

def clean_rr_ms(rr_ms, rr_t_sec, rr_min_ms=300, rr_max_ms=2000, max_rel_change=0.2):
    """
    Basic artifact/ectopy cleaning for 24h. Returns (rr_ms_clean, rr_t_clean, good_mask).
    rr_t_sec is the timestamp for each RR interval (we use midpoints).
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    rr_t_sec = np.asarray(rr_t_sec, dtype=float)

    good = (rr_ms >= rr_min_ms) & (rr_ms <= rr_max_ms)

    # relative jump vs local median (robust)
    if rr_ms.size >= 5:
        med = pd.Series(rr_ms).rolling(9, center=True, min_periods=1).median().to_numpy()
        rel = np.abs(rr_ms - med) / np.maximum(med, 1e-9)
        good &= (rel <= max_rel_change)

    return rr_ms[good], rr_t_sec[good], good

# ----------------------------
#  D) Windowed HRV (time-domain)
# ----------------------------

def rmssd(rr_ms):
    d = np.diff(rr_ms)
    return np.sqrt(np.mean(d * d)) if d.size else np.nan

def sdnn(rr_ms):
    return np.std(rr_ms, ddof=1) if rr_ms.size >= 2 else np.nan

def pnn50(rr_ms):
    d = np.abs(np.diff(rr_ms))
    return 100.0 * np.mean(d > 50.0) if d.size else np.nan

def mean_hr(rr_ms):
    # RR in ms -> bpm
    return 60000.0 / np.mean(rr_ms) if rr_ms.size else np.nan


