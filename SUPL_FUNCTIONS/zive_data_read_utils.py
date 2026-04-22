from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Literal, Optional, Union, Any, Dict, Tuple, cast

import numpy as np
import neurokit2 as nk
import warnings

# =========================
# Your ECG loading helpers
# =========================

def zive_read_file_1ch(path: Union[str, Path]) -> np.ndarray:
    """Load a single-channel ECG file stored as big-endian 32-bit integers."""
    with open(os.fspath(path), "rb") as f:
        raw = np.fromfile(f, dtype=">i4")
    if raw.size == 0:
        raise ValueError(f"File '{path}' is empty or not a valid ZIVE binary.")

    rawf = raw.astype(np.float64, copy=False)

    adc_max = 0x800000  # 2^23
    vref = 2.5
    scaled = (rawf - adc_max / 2) * 2 * vref / adc_max / 3.5 * 1000.0
    return scaled - float(scaled.mean())


def load_ecg_npy(path: Union[str, Path]) -> np.ndarray:
    """Load ECG data from .npy, or custom ZIVE binary and return a 1-D float64 array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ECG file not found: {p}")

    ext = p.suffix.lower()
    if ext == ".npy":
        data = np.load(os.fspath(p))
    elif len(ext) == 4 and ext[1:].isdigit():  # custom numeric extension rule
        data = zive_read_file_1ch(p)
    else:
        raise ValueError(f"Unsupported ECG file extension '{ext or '<none>'}'.")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D ECG signal in '{p}', got shape {arr.shape}.")
    return arr


def read_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON metadata as dict."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object/dict in '{p}', got {type(obj).__name__}.")
    return obj


def find_matching_ecg_file(folder: Path, basename: str, data_format: DataFormat) -> Optional[Path]:
    """
    Find ECG file that matches basename from JSON file stem.
    """
    # Format 1:
    # JSON = 1637176.830.json
    # ECG  = 1637176.830
    if data_format in ("auto", "paired_decimal"):
        cand_decimal = folder / basename
        if cand_decimal.exists() and cand_decimal.is_file():
            return cand_decimal

    # Format 2:
    # JSON = 1001_01.json
    # ECG  = 1001_01.npy
    if data_format in ("auto", "paired_npy"):
        cand_npy = folder / f"{basename}.npy"
        if cand_npy.exists() and cand_npy.is_file():
            return cand_npy

    return None
# =========================
# Record list builder
# =========================

@dataclass
class ECGRecordInfo:
    basename: str
    ecg_path: Path
    json_path: Path


DataFormat = Literal["auto", "paired_decimal", "paired_npy"]


from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class ECGRecordScanSummary:
    total_json: int
    excluded: int
    matched: int
    unmatched_json: int


@dataclass
class ECGRecordScanResult:
    records: List[ECGRecordInfo]
    summary: ECGRecordScanSummary


def _normalize_exclude_token(token: str) -> str:
    """
    Normalize one exclusion entry to record basename.
    """
    token = token.strip()
    if not token:
        return ""

    p = Path(token)

    if p.suffix.lower() in {".json", ".npy"}:
        return p.stem

    return p.name


def _load_excluded_basenames(exclude_list: Optional[Union[str, Path]]) -> set[str]:
    """
    Load excluded filenames/basenames from text file.

    Supported syntax:
    - one entry per line
    - comma-separated entries
    - mixed commas and line breaks
    - comments starting with '#'
    """
    if exclude_list is None:
        return set()

    exclude_path = Path(exclude_list)
    if not exclude_path.exists():
        raise FileNotFoundError(f"Exclude file not found: {exclude_path}")
    if not exclude_path.is_file():
        raise ValueError(f"Exclude path is not a file: {exclude_path}")

    text = exclude_path.read_text(encoding="utf-8")
    if not text.strip():
        return set()

    tokens: list[str] = []

    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",")]
        tokens.extend(part for part in parts if part)

    excluded = set()
    for tok in tokens:
        norm = _normalize_exclude_token(tok)
        if norm:
            excluded.add(norm)

    return excluded


def list_ecg_records(
    folder: Union[str, Path],
    data_format: DataFormat = "auto",
    exclude_list: Optional[Union[str, Path]] = None,
) -> ECGRecordScanResult:
    """
    Scan folder and return ECG records with matching JSON metadata
    together with a small summary.

    Supported formats
    -----------------
    1) paired_decimal:
       ECG file:   1637176.830
       JSON file:  1637176.830.json

    2) paired_npy:
       ECG file:   1001_01.npy
       JSON file:  1001_01.json

    3) auto:
       Accept both formats in the same folder.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected directory, got: {folder}")

    allowed_formats = {"auto", "paired_decimal", "paired_npy"}
    if data_format not in allowed_formats:
        raise ValueError(
            f"Unsupported data_format='{data_format}'. Allowed: {sorted(allowed_formats)}"
        )

    excluded_basenames = _load_excluded_basenames(exclude_list)

    records: List[ECGRecordInfo] = []
    total_json = 0
    excluded_count = 0
    matched_count = 0
    unmatched_count = 0

    for json_path in sorted(folder.glob("*.json")):
        total_json += 1
        base = json_path.stem

        if base in excluded_basenames:
            excluded_count += 1
            continue

        ecg_path: Optional[Path] = None

        # Format 1: ECG file has no suffix, e.g. "1637176.830"
        if data_format in ("auto", "paired_decimal"):
            cand_decimal = folder / base
            if cand_decimal.exists() and cand_decimal.is_file():
                if cand_decimal.name != json_path.name:
                    ecg_path = cand_decimal

        # Format 2: ECG file has .npy suffix, e.g. "1001_01.npy"
        if ecg_path is None and data_format in ("auto", "paired_npy"):
            cand_npy = folder / f"{base}.npy"
            if cand_npy.exists() and cand_npy.is_file():
                ecg_path = cand_npy

        if ecg_path is not None:
            records.append(
                ECGRecordInfo(
                    basename=base,
                    ecg_path=ecg_path,
                    json_path=json_path,
                )
            )
            matched_count += 1
        else:
            unmatched_count += 1

    summary = ECGRecordScanSummary(
        total_json=total_json,
        excluded=excluded_count,
        matched=matched_count,
        unmatched_json=unmatched_count,
    )

    return ECGRecordScanResult(
        records=records,
        summary=summary,
    )
    
    
# lowcut_filter
def lowcut_filter(signal, fs=200, lowcut=0.5, method="butterworth", order=4):

    """
    signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method='butterworth', order=2, window_size='default', powerline=50, show=False)[source]
    Filter a signal using different methods such as “butterworth”, “fir”, “savgol” or “powerline” filters.
    Apply a lowpass (if “highcut” frequency is provided), highpass (if “lowcut” frequency is provided) or bandpass (if both are provided) filter to the signal.
    """

    filtered = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=lowcut,
        highcut=None,
        method=method,
        order=order
    )
    return np.asarray(filtered, dtype=float)



# bandpass_filter
def bandpass_filter(signal, fs=200, lowcut=0.5, highcut=40., method="butterworth", order=4):

    """
    signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method='butterworth', order=2, window_size='default', powerline=50, show=False)[source]
    Filter a signal using different methods such as “butterworth”, “fir”, “savgol” or “powerline” filters.
    Apply a lowpass (if “highcut” frequency is provided), highpass (if “lowcut” frequency is provided) or bandpass (if both are provided) filter to the signal.
    """

    filtered = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=lowcut,
        highcut=None,
        method=method,
        order=order
    )
    return np.asarray(filtered, dtype=float)


def get_rpeaks(ecg_signal: np.ndarray, fs: int = 200, correct_artifacts: bool = False
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks with NeuroKit2 and return:
      - rpeaks_samples: np.ndarray[int] of sample indices
      - rpeaks_seconds: np.ndarray[float] of times in seconds

    Raises:
        ValueError: if 'ECG_R_Peaks' is missing or empty.
    """
    _, rpeaks_info = nk.ecg_peaks(ecg_signal, sampling_rate=fs, correct_artifacts=correct_artifacts)

    # teach Pylance the right types
    rpeaks_info = cast(Dict[str, Any], rpeaks_info)
    rpeaks_samples = np.asarray(
        cast(list[int] | np.ndarray, rpeaks_info.get("ECG_R_Peaks", [])),
        dtype=int
    )

    if rpeaks_samples.size == 0:
        raise ValueError("No R-peaks found (ECG_R_Peaks is missing or empty).")

    rpeaks_seconds = rpeaks_samples / float(fs)
    return rpeaks_samples, rpeaks_seconds


# testavimui

def main() -> None:

    folder = "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test"
    exclude_list="/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test/exclude_list.txt"

    scan_result = list_ecg_records(
        folder=folder,
        data_format="auto",
        exclude_list=exclude_list,
    )

    records = scan_result.records
    summary = scan_result.summary

    print(f"total_json     : {summary.total_json}")
    print(f"excluded       : {summary.excluded}")
    print(f"matched        : {summary.matched}")
    print(f"unmatched_json : {summary.unmatched_json}")
    print(f"records returned: {len(records)}")

    # records = list_ecg_records(folder, data_format="auto")

    print(f"Found {len(records)} matched records")
    for rec in records[:5]:
        print(rec.basename, rec.ecg_path.name, rec.json_path.name)
        
        if rec.ecg_path is None:
            msg = f"No matching ECG file for JSON '{rec.json_path.name}'"
            continue

        metadata: Dict[str, Any] = {}
        metadata = read_json_file(rec.json_path)
        
        signal = load_ecg_npy(rec.ecg_path)
        n_samples = int(signal.shape[0])
        print(rec.basename, signal.shape)
    

if __name__ == "__main__":
    main()
