
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

# from pdb import main
from os import pipe
from typing import Any, Iterable, List, Tuple, Dict, Optional

from pathlib import Path

from ecg_denoising_pipeline import DenoisingPipelineConfig, load_ecg_npy
from ecg_denoising_pipeline import ECGDenoisingPipeline
from ecg_denoising_pipeline import load_denoising_config_yaml
from ecg_denoising_pipeline import check_denoising_config
from ecg_denoising_pipeline import resolve_model_path, print_heading, as_seconds
from ecg_denoising_pipeline import DenoisingPipelineResult


from denoising_util import (
    intervals_to_hms,
    samples_dict_to_seconds,
)

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


def calc_noise_stats_from_denoised_result(res: Any) -> Dict[str, Any]:
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

    stats= {
        "out": out_cnt,
        "rdr": rdr_cnt,
        "noi": noi_cnt,
        "tp_pct": tp_pct,
        "tp_samples": tp_samples,
        "n_start": n_start,
        "union_intervals": union,
    }
    return {'out': stats['out'], 'rdr': stats['rdr'], 'noi': stats['noi'], 'tp_pct': stats['tp_pct']}
    

def prepare_denoising_pipeline(
    config_path: Path,
    model_dir: Path,
) -> DenoisingPipelineConfig:
    """Pipeline-aware runner that wires configuration, data loading and execution."""

    config_path = Path(config_path)
    model_dir = Path(model_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # print_heading("Project paths")
    # print("DATA_DIR:", data_dir)
    # print("CONFIG  :", config_path)
    # print("MODEL_DIR:", model_dir)

    cfg = load_denoising_config_yaml(str(config_path))
    fs = float(cfg.fs)

    # print_heading("Denoising pipeline config")
    # friendly_print_denoising_cfg(cfg)
    check_denoising_config(cfg)
    
    # print_heading("Input data")
    # print(f"File: {file_name}")
    # len_secs = math.ceil(len(x) / fs)
    # h, m, s = convert_seconds_to_hms(len_secs)
    # print(f"len(ecg): {len(x)} samples (~{len_secs:.1f} s) | {h:02d}:{m:02d}:{s:02d}")

    # print("path:", path)
    # print(f"Loaded {len(gaps_indices)} gap intervals.")

    cfg.motions.model_name = resolve_model_path(model_dir, cfg.motions.model_name)
    cfg.motions.enabled = True

    # print_heading("UNet model")
    # print("UNet model path:", cfg.motions.model_name)
    # print("Motions enabled:", bool(getattr(cfg.motions, "enabled", True)))

    return cfg

def create_unet_marker(cfg_denoising: DenoisingPipelineConfig) -> str:
    """Create a marker string based on the UNet model name and threshold for output column naming."""
    unet_model_path = Path(cfg_denoising.motions.model_name)
    unet_model_name = unet_model_path.name
    MARKER = unet_model_name.removeprefix("resunet_ecg").removesuffix(".keras")  # -> "_1024_0_5_3_7"
    threshold = cfg_denoising.motions.threshold
    threshold_str = str( threshold).replace('.', '_')
    MARKER += f"_{threshold_str}"
    return MARKER
    

def print_detailed_denoising_results(res_denoising: DenoisingPipelineResult, cfg_denoising: DenoisingPipelineConfig):
    
    print_heading("ECG DENOISING pipeline results")
    print(f"len_original: {len(res_denoising.ecg_orig)}")
    print(f"len_start   : {len(res_denoising.ecg_start)}") 
    print(f"len_denoised   : {len(res_denoising.ecg_denoised)}")

    print("\nMaps (sample intervals):")
    print("map_gaps    :", res_denoising.map_gaps)
    print("map_outliers:", res_denoising.map_outliers)
    print("map_rdropouts:", res_denoising.map_rdropouts)
    print("map_motions :", res_denoising.map_motions)

    print_heading("Detected intervals (in samples)")
    print("Outliers (start):", res_denoising.outliers_indices_start)
    print("Rdropouts (nout):", res_denoising.rdropouts_indices_nout)
    print("Motions (nrd)   :", res_denoising.motions_indices_nrd)

    print_heading("Detected intervals (in seconds)")
    print("Outliers (start):", as_seconds(res_denoising.outliers_indices_start, cfg_denoising.fs))
    print("Rdropouts (nout):", as_seconds(res_denoising.rdropouts_indices_nout, cfg_denoising.fs))
    print("Motions (nrd)   :", as_seconds(res_denoising.motions_indices_nrd, cfg_denoising.fs))

    # print_heading("Detected intervals (as HH:MM:SS, using fixed start_dt)")
    # print("Outliers (start):", intervals_to_hms(res_denoising.outliers_indices_start, cfg_denoising.fs, start_dt))
    # print("Rdropouts (nout):", intervals_to_hms(res_denoising.rdropouts_indices_nout, cfg_denoising.fs, start_dt))
    # print("Motions (nrd)   :", intervals_to_hms(res_denoising.motions_indices_nrd,   cfg_denoising.fs, start_dt))

    print("\nProjected intervals (in samples):")
    print("projected_to_orig", res_denoising.projected_to_orig)
    print("projected_to_start", res_denoising.projected_to_start)

    print("\nProjected intervals (in seconds):")
    proj_orig_sec  = samples_dict_to_seconds(res_denoising.projected_to_orig,  cfg_denoising.fs,ndigits=3)
    proj_start_sec = samples_dict_to_seconds(res_denoising.projected_to_start, cfg_denoising.fs,ndigits=3)

    print("projected_to_orig (s):", proj_orig_sec)
    print("projected_to_start (s):", proj_start_sec)


# def denoise_and_calc_noise_stats(x: np.ndarray, pipe: ECGDenoisingPipeline, model) -> Dict[str, Any]:
    
    
    
    
    
#     return {
#         "out": out_cnt,
#         "rdr": rdr_cnt,
#         "noi": noi_cnt,
#         "tp_pct": tp_pct
#     }

import numpy as np
def denoise_and_calc_noise_stats_imitation(x: np.ndarray) -> Dict[str, Any]:
    """
    Run the denoising pipeline on input signal x, then calculate noise stats from the result.
    This is a convenience function that combines the steps for easier testing.
    """
    # res_denoising = pipe.run(x, gaps_indices=[])
    # stats = calc_noise_stats_from_denoised_result(res_denoising) 
 
    return {
            "out": 1, "rdr": 2, "noi": 3,
            "tp_pct": 0.5,
        } 
 
    
# testing

def main() -> None:  # run your main function that prepares the pipeline and processes records, then calls calc_noise_stats_from_result on each result and prints the stats.
# -------------------------------------------------------------
    # NEW: Prepare denoising ONCE (before main cycle)
    # -------------------------------------------------------------
    
    # cfg_denoising_path = args.cfg_denoising
    # model_dir = args.model_dir
    
    cfg_denoising_path = "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml"
    model_dir = "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET"
    
    #  Preparing the Denoising pipeline with configuration and model paths
    print("\n*****Denoising pipeline config:")
    cfg_denoising = prepare_denoising_pipeline(Path(cfg_denoising_path), Path(model_dir))
    pipe = ECGDenoisingPipeline(cfg_denoising)

    MARKER = create_unet_marker(cfg_denoising)
    print("Generated MARKER for output:", MARKER)   
    
    # start0
    # Input data
    data_dir = Path("/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_ORIG/ecg_zive_npy")
    
    # Pseudo_annotated
    fileNames = ['1019_118.npy'] #
    fileNames = ['1001_4.npy'] #
    fileNames = ['1005_2.npy'] #
    fileNames = ['1008_1.npy'] #
    fileNames = ['1008_1.npy','1008_10.npy'] #

    fileNames = ['1008_10.npy'] #


    for file_name in fileNames:
        print_heading("Input data")
        print(f"File: {file_name}")

        # Load ECG signal and gaps for display
        data_path = Path(data_dir / file_name)
        x = load_ecg_npy(data_path)
        print(f"\nLoaded ECG signal: len(ecg): {len(x)}")

        
        # DENOISING **************************************

        print(f"\nRunning Denoising pipeline with unet {Path(cfg_denoising.motions.model_name).stem} and threshold {cfg_denoising.motions.threshold}...")
        # Execute the pipeline in the notebook with explicit arguments
        res_denoising = pipe.run(x, gaps_indices=[])

                # RESULTS **************************************

        print_detailed_denoising_results(res_denoising, cfg_denoising)

        print("\nCalculating noise stats from denoising results...")
        stats = calc_noise_stats_from_denoised_result(res_denoising)
        # print(stats["out"], stats["rdr"], stats["noi"], stats["tp_pct"])
        print(
            f"Noise stats → "
            f"\nOUT: {stats['out']}, "
            f"\nRDR: {stats['rdr']}, "
            f"\nNOI: {stats['noi']}, "
            f"\nTP%: {stats['tp_pct']:.1f}"
        )
        
if __name__ == "__main__":
    main()