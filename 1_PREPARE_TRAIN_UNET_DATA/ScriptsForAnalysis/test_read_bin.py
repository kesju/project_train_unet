#!/usr/bin/env python3
from pathlib import Path
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import json
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# I/O helpers
# ----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return obj


def load_int16_bin(path: Path, endian: str) -> np.ndarray:
    """
    endian: 'be' or 'le'
    """
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"BIN size not divisible by 2 (expected int16): {path}")
    dt = np.dtype(">i2") if endian == "be" else np.dtype("<i2")
    sig = np.frombuffer(raw, dtype=dt)
    return sig.astype(np.int16, copy=False)


def zive_read_file_1ch(filename):
    
    """
    Reads a binary file containing ECG data and processes it to return the ECG signal.
   
   That .bin file is a raw sample stream (no header/metadata) where each ECG sample is stored as 4 bytes:

    Endianness: big-endian

    Type: effectively 24-bit unsigned ADC counts packed into 4 bytes (the most significant byte is always 0x00)

    Layout: [b0 b1 b2 b3] per sample, repeated

    File size: 512,000 bytes ⇒ 128,000 samples (512000 / 4)
    
    Conversion to voltage (mV): 
        1. Convert the 4-byte big-endian integer to a signed integer by subtracting half the ADC range (0x800000).
        2. Scale to voltage using the reference voltage (2.5V) and ADC resolution (3.5).
        3. Center the signal by subtracting the mean.
       
    """
    
    with open(filename, 'rb') as f:  # Use 'rb' to read binary file
        a = np.fromfile(f, dtype=np.dtype('>i4'))  # Read file content as big-endian 4-byte integers
    
    ADCmax = 0x800000
    Vref = 2.5
    b = (a - ADCmax / 2) * 2 * Vref / ADCmax / 3.5 * 1000  # Corrected the calculation by adding multiplication symbol
    ecg_signal = b - np.mean(b)
    
    return ecg_signal


def get_ecg_signal(args_fileName):
  # Extract the file extension
  file_extension = os.path.splitext(args_fileName)[1]

  # Check if the extension is .h5py
  if file_extension.lower() == '.h5':
        # print("The file has a .h5 extension.")
        with h5py.File(args_fileName, 'r') as f:
          ecg_signal = f['dataset'][:]
          
  # Check if the extension is three digits (excluding the dot)
  elif len(file_extension) == 4 and file_extension[1:].isdigit():
        # print("The file has a three-digit extension.")
        ecg_signal = zive_read_file_1ch(args_fileName)
        
  elif file_extension.lower() == '.npy':
        # print("The file has a .npy extension.")
        ecg_signal = np.load(args_fileName, mmap_mode='r')
        
  # If neither condition is true
  else:
        ecg_signal = np.array([])
        print("The file does not have a .h5py extension or a three-digit extension or .npy extension.")
   
  return ecg_signal




bin_path = Path("/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_COLLECT_ZIVE_DATA/"
                "user-65955b5f50e02b125d4998ad/659ec124870b3d1d1630be39/recordings/"
                "659ebcdd870b3d1d6e30bb61.bin")
json_path = bin_path.with_suffix(".json")

meta = load_json(json_path)
endian = 'be'
sig = load_int16_bin(bin_path, endian=endian)


print("dtype:", sig.dtype, "samples:", sig.size)
print("first 20 values:", sig[:20].tolist())


# python3 test_read_bin.py

# \\wsl.localhost\Ubuntu\home\kesju\DI\2025_ZIVEO\PROJECT_TRAIN_UNET\0_COLLECT_ZIVE_DATA\659ebcdd870b3d1d6e30bb61

sig = zive_read_file_1ch(bin_path)
print("dtype:", sig.dtype, "samples:", sig.size)
print("first 20 values:", sig[:20].tolist())


bin_path = Path("/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_COLLECT_ZIVE_DATA/659ebcdd870b3d1d6e30bb61/1704481.565")
sig = zive_read_file_1ch(bin_path)
print("dtype:", sig.dtype, "samples:", sig.size)
print("first 20 values:", sig[:20].tolist())