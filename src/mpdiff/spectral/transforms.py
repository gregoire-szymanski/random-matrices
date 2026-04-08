"""Marcenko-Pastur forward transform and numerical helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .densities import DiscreteSpectrum, GridDensity


def mp_stieltjes_fixed_point(
    population: DiscreteSpectrum,
    z: complex,
    aspect_ratio: float,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    initial_m: complex | None = None,
) -> tuple[complex, int, bool]:
    """Solve Silverstein fixed-point equation for the Stieltjes transform.

    We solve:

    ``m(z) = ∫ 1 / (t (1 - c - c z m(z)) - z) dH(t)``

    by damped fixed-point iterations.
    """
    c = aspect_ratio
    m = initial_m if initial_m is not None else -1.0 / z
    for it in range(1, max_iter + 1):
        denom = population.atoms * (1.0 - c - c * z * m) - z
        rhs = np.sum(population.weights / denom)
        m_next = damping * rhs + (1.0 - damping) * m
        if abs(m_next - m) < tol:
            return m_next, it, True
        m = m_next
    return m, max_iter, False


def mp_forward_transform(
    population: DiscreteSpectrum,
    aspect_ratio: float,
    grid: np.ndarray,
    eta: float = 1e-3,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    return_diagnostics: bool = False,
) -> GridDensity | tuple[GridDensity, dict[str, Any]]:
    """Compute MP image density on ``grid`` via Stieltjes inversion."""
    grid = np.asarray(grid, dtype=float)
    m_values = np.zeros_like(grid, dtype=np.complex128)
    iterations = np.zeros_like(grid, dtype=int)
    converged = np.zeros_like(grid, dtype=bool)

    previous_m: complex | None = None
    for idx, x in enumerate(grid):
        z = complex(x, eta)
        m, it_count, ok = mp_stieltjes_fixed_point(
            population=population,
            z=z,
            aspect_ratio=aspect_ratio,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            initial_m=previous_m,
        )
        m_values[idx] = m
        iterations[idx] = it_count
        converged[idx] = ok
        previous_m = m

    density = np.clip(np.imag(m_values) / np.pi, 0.0, None)
    result = GridDensity(grid=grid, density=density, name="mp_forward")

    if return_diagnostics:
        diagnostics = {
            "mean_iterations": float(np.mean(iterations)),
            "max_iterations": int(np.max(iterations)),
            "convergence_rate": float(np.mean(converged)),
        }
        return result, diagnostics
    return result


def mp_dirac_density(grid: np.ndarray, variance: float, aspect_ratio: float) -> np.ndarray:
    """Continuous MP density for a single population eigenvalue.

    The law has support
    ``[variance * (1-sqrt(c))^2, variance * (1+sqrt(c))^2]``.
    """
    x = np.asarray(grid, dtype=float)
    c = max(aspect_ratio, 1e-12)
    if variance <= 0:
        return np.zeros_like(x)

    sqrt_c = np.sqrt(c)
    a = variance * (1.0 - sqrt_c) ** 2
    b = variance * (1.0 + sqrt_c) ** 2

    density = np.zeros_like(x)
    mask = (x > max(a, 0.0)) & (x < b)
    numerator = np.sqrt((b - x[mask]) * (x[mask] - a))
    denominator = 2.0 * np.pi * c * variance * np.maximum(x[mask], 1e-15)
    density[mask] = numerator / denominator
    return density
