"""Random number helper utilities."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random number generator.

    Parameters
    ----------
    seed:
        Random seed for reproducibility. If ``None``, entropy from OS is used.

    Returns
    -------
    np.random.Generator
        Random generator instance.
    """
    return np.random.default_rng(seed)
