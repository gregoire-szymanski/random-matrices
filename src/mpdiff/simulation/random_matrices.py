"""Random matrix generators used by covariance builders."""

from __future__ import annotations

import numpy as np


def generate_orthogonal_matrix(d: int, rng: np.random.Generator, method: str = "haar") -> np.ndarray:
    """Generate an orthogonal matrix.

    Parameters
    ----------
    d:
        Matrix dimension.
    rng:
        NumPy random generator.
    method:
        ``"haar"`` for a Haar-distributed orthogonal matrix, or ``"identity"``.
    """
    if method == "identity":
        return np.eye(d)
    if method != "haar":
        raise ValueError(f"Unsupported orthogonal method: {method}")

    gaussian = rng.normal(size=(d, d))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q @ np.diag(signs)


def generate_low_rank_factor(d: int, rank: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Generate a random low-rank factor matrix with normalized columns."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    factor = rng.normal(size=(d, rank))
    return scale * factor / np.sqrt(rank)
