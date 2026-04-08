"""Empirical spectral analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from .densities import GridDensity
from .grids import grid_from_samples


def realized_covariance(path: np.ndarray, total_time: float | None = None) -> np.ndarray:
    """Compute realized covariance from a simulated path.

    Parameters
    ----------
    path:
        Simulated path with shape ``(n_steps + 1, d)``.
    total_time:
        Total horizon. If ``None``, uses ``n_steps`` as normalizing factor.

    Returns
    -------
    np.ndarray
        Realized covariance matrix.
    """
    increments = np.diff(path, axis=0)
    scale = float(total_time if total_time is not None else increments.shape[0])
    if scale <= 0:
        raise ValueError("total_time must be positive")
    return (increments.T @ increments) / scale


def empirical_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of a symmetric matrix."""
    return np.linalg.eigvalsh(0.5 * (matrix + matrix.T))


def empirical_spectral_density(
    eigenvalues: np.ndarray,
    grid: np.ndarray | None = None,
    bandwidth: float | None = None,
) -> GridDensity:
    """Estimate empirical spectral density with Gaussian KDE."""
    eigs = np.asarray(eigenvalues, dtype=float)
    if grid is None:
        grid = grid_from_samples(eigs)
    if np.allclose(eigs, eigs[0]):
        # Degenerate case: use a narrow Gaussian around a single atom.
        sigma = max(1e-3, 0.02 * max(eigs[0], 1.0))
        density = np.exp(-0.5 * ((grid - eigs[0]) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        return GridDensity(grid=grid, density=density, name="empirical")

    kde = gaussian_kde(eigs, bw_method=bandwidth)
    density = kde(grid)
    return GridDensity(grid=grid, density=density, name="empirical")
