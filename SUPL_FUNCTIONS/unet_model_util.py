# Versija 1.0
# Autorius: Kęstutis Juškevičius
# Data: 2024-10-12
# Aprašymas: Šis modulis teikia funkcijas unet modelio nuskaitymui ir validavimui.

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional, cast

import numpy as np
import tensorflow as tf
import keras  # Keras 3 API (standalone)

# ---------- Public return type ----------
@dataclass(frozen=True)
class UnetModelBundle:
    model: keras.Model
    expected_input_shape: Tuple[Optional[int], Optional[int]]
    model_path: Path
    logger: logging.Logger  # <- returned too

# ---------- Internal (cached) bundle ----------
@dataclass(frozen=True)
class _CachedBundle:
    model: keras.Model
    expected_input_shape: Tuple[Optional[int], Optional[int]]
    model_path: Path

def _default_logger() -> logging.Logger:
    """Create or return a singleton-ish console logger for the loader."""
    logger = logging.getLogger("unet_loader")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
        logger.propagate = False
    return logger

def _infer_input_shape(mdl: keras.Model, logger: logging.Logger) -> Tuple[Optional[int], Optional[int]]:
    logger.info("Verifying model input shape")
    input_shape = getattr(mdl, "input_shape", None)

    if input_shape is None:
        inputs = getattr(mdl, "inputs", None)
        if inputs:
            input_shape = tf.TensorShape(inputs[0].shape).as_list()
        else:
            msg = "Loaded model does not expose input shape; cannot determine input shape."
            logger.error(msg)
            raise ValueError(msg)

    if hasattr(input_shape, "as_list"):
        input_shape = input_shape.as_list()

    ts = tuple(int(d) if isinstance(d, (int, np.integer)) else None for d in input_shape)

    # Drop leading batch dimension if present (commonly None)
    dims = ts[1:] if (len(ts) >= 1 and ts[0] is None) else ts

    # Ensure we always return exactly (segment_length, channels)
    if len(dims) == 2:
        return cast(Tuple[Optional[int], Optional[int]], dims)

    if len(dims) > 2:
        # Heuristic: pick the last two dims (e.g., (H, W, C) -> (W, C); validation will catch mismatch)
        cand = dims[-2:]
        logger.warning(f"Input shape {ts} has >2 non-batch dims; using last two dims {cand}")
        return cast(Tuple[Optional[int], Optional[int]], cand)

    msg = f"Unsupported model input shape {ts}; expected (batch?, segment_length, channels)."
    logger.error(msg)
    raise ValueError(msg)

def _validate_expected_shape(expected: Tuple[Optional[int], Optional[int]],
                             segment_length: int,
                             logger: logging.Logger) -> None:
    desired = (segment_length, 1)
    if expected != desired:
        msg = f"Model expects input shape {expected}, but script uses {desired}"
        logger.error(msg)
        raise ValueError(msg)

@lru_cache(maxsize=1)
def _load_cached(model_path_str: str, segment_length: int) -> _CachedBundle:
    """
    Heavy loader keyed by (model_path_str, segment_length).
    Does not keep/carry the caller's logger to avoid odd cache effects.
    """
    logger = _default_logger()
    model_path = Path(model_path_str)
    logger.info(f"Loading U-Net model from {model_path}")

    try:
        mdl = keras.models.load_model(model_path)
        logger.debug("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    expected_input_shape = _infer_input_shape(mdl, logger)
    _validate_expected_shape(expected_input_shape, segment_length, logger)

    return _CachedBundle(model=mdl, expected_input_shape=expected_input_shape, model_path=model_path)

def get_unet_model(unet_model_dir: Path,
                   model_filename: str,
                   segment_length: int,
                   logger: Optional[logging.Logger] = None) -> UnetModelBundle:
    """
    Public API: returns UnetModelBundle(model, expected_input_shape, model_path, logger).

    - Reuses a cached load for identical (path, segment_length).
    - Uses the provided `logger` if given; otherwise a sensible default console logger.
    - Always returns the logger so callers can keep using the exact same one.
    """
    lg = logger or _default_logger()
    model_path = (unet_model_dir / model_filename).resolve()

    lg.info(f"Preparing to load U-Net model from {model_path}")
    cached = _load_cached(str(model_path), segment_length)
    lg.info(f"U-Net model ready: {cached.model_path.name}, expected_input_shape={cached.expected_input_shape}")

    # Build the public bundle with the caller's logger attached (not cached)
    return UnetModelBundle(
        model=cached.model,
        expected_input_shape=cached.expected_input_shape,
        model_path=cached.model_path,
        logger=lg,
    )
