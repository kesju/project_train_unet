# Versija 1.0
# Autorius: Kęstutis Juškevičius
# Data: 2024-10-12
# Aprašymas: Šis modulis teikia funkcijas EKG filtravimui.

from typing import Literal, Optional, Sequence
import numpy as np
from dataclasses import dataclass

@dataclass
class FilterParams:
    """Configuration for filtering an ECG recording."""

    enabled: bool = True
    method: Literal["butterworth", "fir", "bessel", "savgol"] = "butterworth"
    type: Literal["highpass", "bandpass"] = "highpass"
    lowcut: Optional[float] = 0.5
    highcut: Optional[float] = 90.0
    order: int = 5
    # order (int) – Only used if method is "butterworth" or "savgol"

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable representation, useful for logging."""

        return {
            "enabled": self.enabled,
            "method": self.method,
            "type": self.type,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "order": self.order,
        }


FType = Literal["highpass", "bandpass"]
NkMethod = Literal["butterworth", "fir", "bessel", "savgol"]  # tighten as you like

def filter_ecg(
    x: Sequence[float] | np.ndarray,
    fs: int,
    method: NkMethod,
    type: FType,
    lowcut: float | None,
    highcut: float | None,
    order: int,
) -> np.ndarray:
    """Filter an ECG signal using NeuroKit2.

    Args:
        x: Input ECG samples (list/tuple/np.ndarray).
        fs: Sampling rate in Hz (positive int).
        method: One of 'butterworth', 'fir', 'bessel', 'savgol'.
        type: 'highpass' or 'bandpass'.
        lowcut: Hz; required for highpass/bandpass.
        highcut: Hz; required for bandpass.
        order: Positive int. For Butterworth/Bessel it is the filter order.
               For Savitzky–Golay it is the polynomial order (not window size).

    Returns:
        Filtered signal as float64 np.ndarray.
    """
    # --- basic shape/empty handling
    signal = np.asarray(x, dtype=float)
    if signal.size == 0:
        return signal.copy()

    # --- normalize and validate text params
    method = method.lower()  # type: ignore[assignment]
    type = type.lower()    # type: ignore[assignment]
    if type not in {"highpass", "bandpass"}:
        raise ValueError("ftype must be 'highpass' or 'bandpass'.")
    if method not in {"butterworth", "fir", "bessel", "savgol"}:
        raise ValueError("Unsupported method for NK2.")

    # --- numeric sanity
    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("fs must be a positive integer.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be an integer >= 1.")
    # Savitzky–Golay tip: very high polynomial orders are unstable; keep small.
    if method == "savgol" and order > 6:
        raise ValueError("For 'savgol', keep polynomial order reasonably small (<= 6).")

    # --- cutoff requirements by ftype
    if type == "highpass":
        if lowcut is None:
            raise ValueError("High-pass requires lowcut.")
        high = None
    else:  # bandpass
        if lowcut is None or highcut is None:
            raise ValueError("Band-pass requires both lowcut and highcut.")
        if not (0 < lowcut < highcut):
            raise ValueError("Require 0 < lowcut < highcut for band-pass.")
        high = float(highcut)

    # cast for NK2
    low = float(lowcut) if lowcut is not None else None

    # --- call NeuroKit2
    try:
        import neurokit2 as nk  # type: ignore
    except Exception as exc:
        raise RuntimeError("NeuroKit2 is required (`pip install neurokit2`).") from exc

    filtered = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=low,
        highcut=high,
        method=method,
        order=order,
    )
    return np.asarray(filtered, dtype=float)



def ecg_filter(
    signal: Sequence[float] | np.ndarray,
    fs: int,
    filter_params: Optional[FilterParams] = None,
    file_name: Optional[str] = None,
) -> np.ndarray:
    """Filter an ECG signal using the configured filter parameters.
    
    Args:
        signal: Input ECG samples (list/tuple/np.ndarray).
        fs: Sampling rate in Hz (positive int).
        filter_params: Filter configuration. If None, uses default settings.
        file_name: Optional filename for error reporting.
        
    Returns:
        Filtered signal as float64 np.ndarray, or original signal if filtering fails.
    """
    if filter_params is None:
        filter_params = FilterParams(enabled=True, type="highpass", lowcut=0.5, highcut=None, order=5)
    
    if not filter_params.enabled:
        return np.asarray(signal, dtype=float)
    
    try:
        processed = filter_ecg(
            signal,
            fs,
            filter_params.method,
            filter_params.type,
            filter_params.lowcut,
            filter_params.highcut,
            filter_params.order,
        )
        return processed
    except Exception as exc:
        error_msg = f"Filtering failed"
        if file_name:
            error_msg += f" for {file_name}"
        error_msg += f": {exc}. Using raw signal instead."
        print(error_msg)
        return np.asarray(signal, dtype=float)