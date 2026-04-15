from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..index_map import IndexMap, Interval


@dataclass
class DenoisingPipelineResult:
    """Pipeline-aware container describing every stage of the denoising run."""

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
    
    
    
    