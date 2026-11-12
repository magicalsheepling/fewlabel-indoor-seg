"""NYUv2 loading helpers.

The labelled NYUv2 file is a MATLAB v7.3 (HDF5) file. Read through h5py, arrays
come out with reversed axes: one RGB frame is (3, W, H) and one label map is
(W, H). The functions here convert single frames to the usual image layout.
"""

from __future__ import annotations

import numpy as np


def image_to_hwc(frame: np.ndarray) -> np.ndarray:
    """(3, W, H) as stored in the .mat file -> (H, W, 3)."""
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError(f"expected (3, W, H), got {frame.shape}")
    return np.ascontiguousarray(frame.transpose(2, 1, 0))


def label_to_hw(label: np.ndarray) -> np.ndarray:
    """(W, H) as stored in the .mat file -> (H, W)."""
    if label.ndim != 2:
        raise ValueError(f"expected (W, H), got {label.shape}")
    return np.ascontiguousarray(label.T)


def label_budget_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    """Pick a reproducible subset of training indices for a label budget."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, round(n * fraction))
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n)[:k])
