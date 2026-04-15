# Demonstruoja triukšmų šalinimą iš  ECG signalų, naudojant ecg_denoising_pipeline.
# taip pat demonstruoja funkcijų, naudojamų aptiktų triukšmų parametrų skaičiavimui, darbą

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from datetime import datetime, timedelta
# "Run ECG denoising pipeline, ECG ectopy detecting and removing pipeline,
# HRV calculation on a single .npy ECG file.\n"

# from matplotlib.collections import LineCollection
from pathlib import Path
import math, time
# import numpy as np
# from typing import Dict, List, Tuple, Iterable
from ecg_denoising_pipeline.utils import map_mark_denoised_to_start
from ecg_denoising_pipeline.utils import convert_seconds_to_hms, friendly_print_denoising_cfg
from ecg_denoising_pipeline.io_utils import load_gaps, load_array_or_fail
from ecg_denoising_pipeline.config import load_denoising_config_yaml
# from ecg_denoising_pipeline.index_map import Interval
from record_noise_stats import calc_noise_stats_from_result


# --- ecg_denoising_pipeline imports (as in your original) ---
from ecg_denoising_pipeline import (
    run_denoising_pipeline,
    print_heading,
    as_seconds,
)

from ecg_denoising_pipeline.steps import check_denoising_config

from denoising_util import (
    find_project_root_by_name,
    intervals_to_hms,
    samples_dict_to_seconds,
)

from typing import List, Tuple

Interval = Tuple[int, int]

start_time_1 = time.time()

print("\n**********HEART RATE VARIABILITY\n")

HOME = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = find_project_root_by_name(target="PROJECT_TRAIN_UNET", start=HOME)

print("PROJECT ROOT_DIR:", PROJECT_ROOT)
print("PROJECT HOME DIR:", HOME)

print("\n*****ECG DENOISING")
print("\nData, Configuration and Model Paths:")
cfg_denoising_path = PROJECT_ROOT/ 'CONFIG'/ 'denoising_config.yaml'
model_unet_dir = PROJECT_ROOT/'MODEL_UNET'
data_dir = PROJECT_ROOT/"DATA_ORIG"/"ecg_zive_npy"
gaps_dir = PROJECT_ROOT/'DATA/LONG_ECG_AND_SCRIPTS'

print("DATA_DIR:", data_dir)
print("CONFIG  :", cfg_denoising_path)
print("MODEL_DIR:", model_unet_dir)

print_heading("Denoising pipeline config")
cfg_denoising = load_denoising_config_yaml(str(cfg_denoising_path))
# friendly_print_denoising_cfg(cfg_denoising)
check_denoising_config(cfg_denoising)

        #   INPUT DATA **************************************

# start0
# Input data
# Pseudo_annotated
fileNames = ['1019_118.npy'] #
fileNames = ['1001_4.npy'] #
fileNames = ['1005_2.npy'] #
fileNames = ['1008_1.npy'] #
fileNames = ['1008_10.npy'] #
file_name = fileNames[0]


print_heading("Input data")
print(f"File: {file_name}")

# Load ECG signal and gaps for display
x = load_array_or_fail(data_dir, file_name)

len_secs = math.ceil(len(x) / cfg_denoising.fs)
h, m, s = convert_seconds_to_hms(len_secs)
print(f"\nLoaded ECG signal: len(ecg): {len(x)} samples (~{len_secs:.1f} s) duration: {h:02d}:{m:02d}:{s:02d}")

  
        # DENOISING **************************************

print("\nRunning Denoising pipeline...")
# Execute the pipeline in the notebook with explicit arguments
res_denoising, cfg_denoising = run_denoising_pipeline(
    x=x,
    gaps_indices=[],
    config_path=cfg_denoising_path,
    model_dir=model_unet_dir,
    disable_motions=False,
)

        # RESULTS **************************************

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

end_time_1 = time.time()
time_taken_1 = end_time_1 - start_time_1
print(f"\nTime taken for the DENOISING pipeline: {time_taken_1:.2f} secs")  

stats = calc_noise_stats_from_result(res_denoising)
print(stats["out"], stats["rdr"], stats["noi"], stats["tp_pct"])

