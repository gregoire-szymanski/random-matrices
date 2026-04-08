"""Comparison metrics for spectral densities and laws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .densities import DiscreteSpectrum, GridDensity


@dataclass(slots=True)
class SpectrumComparison:
    """Diagnostic metrics comparing a candidate law to a reference law."""

    l1: float
    l2: float
    wasserstein_1: float
    support_min_diff: float
    support_max_diff: float
    moment_abs_errors: list[float]


def _common_grid(reference: GridDensity, candidate: GridDensity, num_points: int = 1024) -> np.ndarray:
    low = min(float(reference.grid[0]), float(candidate.grid[0]))
    high = max(float(reference.grid[-1]), float(candidate.grid[-1]))
    if high <= low:
        high = low + 1e-6
    return np.linspace(low, high, num_points)


def resample_density(density: GridDensity, grid: np.ndarray) -> np.ndarray:
    """Interpolate a density on a new grid with non-negativity clipping."""
    values = np.interp(grid, density.grid, density.density, left=0.0, right=0.0)
    return np.clip(values, 0.0, None)


def support_interval(density: GridDensity, threshold: float = 1e-4) -> tuple[float, float]:
    """Approximate support interval where density exceeds a threshold."""
    mask = density.density > threshold
    if not np.any(mask):
        return float(density.grid[0]), float(density.grid[-1])
    idx = np.flatnonzero(mask)
    return float(density.grid[idx[0]]), float(density.grid[idx[-1]])


def l1_distance(reference: GridDensity, candidate: GridDensity, num_points: int = 1024) -> float:
    """Approximate L1 distance between densities."""
    grid = _common_grid(reference, candidate, num_points=num_points)
    ref_vals = resample_density(reference, grid)
    cand_vals = resample_density(candidate, grid)
    return float(np.trapz(np.abs(ref_vals - cand_vals), grid))


def l2_distance(reference: GridDensity, candidate: GridDensity, num_points: int = 1024) -> float:
    """Approximate L2 distance between densities."""
    grid = _common_grid(reference, candidate, num_points=num_points)
    ref_vals = resample_density(reference, grid)
    cand_vals = resample_density(candidate, grid)
    return float(np.sqrt(np.trapz((ref_vals - cand_vals) ** 2, grid)))


def wasserstein_1_distance(reference: GridDensity, candidate: GridDensity, n_quantiles: int = 1000) -> float:
    """1-Wasserstein distance via quantile integration."""
    probs = np.linspace(0.0, 1.0, n_quantiles)
    q_ref = reference.quantiles(probs)
    q_cand = candidate.quantiles(probs)
    return float(np.trapz(np.abs(q_ref - q_cand), probs))


def compare_grid_densities(
    reference: GridDensity,
    candidate: GridDensity,
    moment_orders: tuple[int, ...] = (1, 2, 3),
) -> SpectrumComparison:
    """Compute a compact set of spectral discrepancy metrics."""
    ref_support = support_interval(reference)
    cand_support = support_interval(candidate)

    moment_errors = [
        abs(reference.moment(order) - candidate.moment(order))
        for order in moment_orders
    ]

    return SpectrumComparison(
        l1=l1_distance(reference, candidate),
        l2=l2_distance(reference, candidate),
        wasserstein_1=wasserstein_1_distance(reference, candidate),
        support_min_diff=abs(ref_support[0] - cand_support[0]),
        support_max_diff=abs(ref_support[1] - cand_support[1]),
        moment_abs_errors=moment_errors,
    )


def discrete_to_grid(
    spectrum: DiscreteSpectrum,
    grid: np.ndarray,
    bandwidth: float | None = None,
) -> GridDensity:
    """Convert discrete law to smooth density representation."""
    return spectrum.to_grid_density(grid, bandwidth=bandwidth)
