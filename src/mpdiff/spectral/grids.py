"""Grid construction utilities for spectral computations."""

from __future__ import annotations

import numpy as np


def make_linear_grid(x_min: float, x_max: float, num_points: int) -> np.ndarray:
    """Create a linearly spaced grid."""
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    return np.linspace(x_min, x_max, num_points)


def grid_from_samples(samples: np.ndarray, num_points: int = 512, pad_ratio: float = 0.05) -> np.ndarray:
    """Create a plotting/computation grid from sample support."""
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("samples must be non-empty")
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 1e-6)
    return make_linear_grid(max(0.0, low - pad_ratio * span), high + pad_ratio * span, num_points)
