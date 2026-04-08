"""Empirical spectral analysis utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import gaussian_kde

from .densities import DiscreteSpectrum, GridDensity
from .grids import grid_from_samples


RealizedCovarianceNormalization = Literal["total_time", "n_steps", "n_steps_minus_one", "none"]


def compute_increments(path: np.ndarray) -> np.ndarray:
    """Compute path increments ``X_{t_{i+1}} - X_{t_i}``.

    Parameters
    ----------
    path:
        Simulated path with shape ``(n_steps + 1, d)``.
    """
    values = np.asarray(path, dtype=float)
    if values.ndim != 2:
        raise ValueError("path must be a 2D array with shape (n_steps + 1, d)")
    if values.shape[0] < 2:
        raise ValueError("path must contain at least two time points")
    return np.diff(values, axis=0)


def realized_covariance_from_increments(
    increments: np.ndarray,
    normalization: RealizedCovarianceNormalization = "total_time",
    total_time: float | None = None,
) -> np.ndarray:
    """Compute realized covariance from increments with configurable scaling.

    Parameters
    ----------
    increments:
        Increments with shape ``(n_steps, d)``.
    normalization:
        Scaling used in
        ``RCV = (ΔX^T ΔX) / scale``:
        - ``total_time``: ``scale = T`` (requires ``total_time``),
        - ``n_steps``: ``scale = n_steps``,
        - ``n_steps_minus_one``: ``scale = max(n_steps - 1, 1)``,
        - ``none``: ``scale = 1``.
    total_time:
        Horizon ``T`` used when ``normalization='total_time'``.

    Returns
    -------
    np.ndarray
        Realized covariance matrix.
    """
    deltas = np.asarray(increments, dtype=float)
    if deltas.ndim != 2:
        raise ValueError("increments must be a 2D array with shape (n_steps, d)")

    n_steps = deltas.shape[0]
    if n_steps <= 0:
        raise ValueError("increments must contain at least one row")

    if normalization == "total_time":
        if total_time is None:
            raise ValueError("total_time must be provided when normalization='total_time'")
        scale = float(total_time)
    elif normalization == "n_steps":
        scale = float(n_steps)
    elif normalization == "n_steps_minus_one":
        scale = float(max(n_steps - 1, 1))
    elif normalization == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if scale <= 0:
        raise ValueError("normalization scale must be positive")

    return (deltas.T @ deltas) / scale


def realized_covariance(
    path: np.ndarray,
    total_time: float | None = None,
    normalization: RealizedCovarianceNormalization = "total_time",
) -> np.ndarray:
    """Compute realized covariance directly from a path."""
    increments = compute_increments(path)
    return realized_covariance_from_increments(
        increments=increments,
        normalization=normalization,
        total_time=total_time,
    )


def empirical_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvalues of a symmetric matrix."""
    return np.linalg.eigvalsh(0.5 * (matrix + matrix.T))


def empirical_discrete_spectrum(eigenvalues: np.ndarray, name: str = "empirical_discrete") -> DiscreteSpectrum:
    """Build empirical discrete spectral law from eigenvalues."""
    vals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if vals.size == 0:
        raise ValueError("eigenvalues must be non-empty")
    return DiscreteSpectrum(atoms=vals, name=name)


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
