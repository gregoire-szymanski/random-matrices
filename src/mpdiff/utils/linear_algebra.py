"""Linear algebra helpers used across simulation and spectral modules."""

from __future__ import annotations

import numpy as np


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    return 0.5 * (matrix + matrix.T)


def sqrt_psd(matrix: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """Compute a stable square root of a symmetric positive semi-definite matrix.

    Parameters
    ----------
    matrix:
        Symmetric PSD matrix.
    jitter:
        Lower clipping threshold for eigenvalues.

    Returns
    -------
    np.ndarray
        Matrix square root ``A`` such that ``A @ A.T`` approximates ``matrix``.
    """
    eigvals, eigvecs = np.linalg.eigh(symmetrize(matrix))
    eigvals = np.clip(eigvals, jitter, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
