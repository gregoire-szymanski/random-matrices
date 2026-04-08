"""Approximate fixed-point MP inverse using Richardson-Lucy iterations."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.transforms import mp_dirac_density


def _support_range(observed: GridDensity, aspect_ratio: float, settings: MPInverseConfig) -> tuple[float, float]:
    sqrt_c = np.sqrt(aspect_ratio)
    lower_scale = (max(1.0 - sqrt_c, 0.05)) ** 2
    upper_scale = (1.0 + sqrt_c) ** 2

    if settings.support_min is None:
        support_min = max(1e-8, observed.grid[0] / upper_scale)
    else:
        support_min = settings.support_min

    if settings.support_max is None:
        support_max = max(support_min * 1.1, observed.grid[-1] / lower_scale)
    else:
        support_max = settings.support_max

    if support_max <= support_min:
        support_max = support_min * 1.5

    return support_min, support_max


def invert_fixed_point(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> tuple[DiscreteSpectrum, GridDensity, dict[str, float | int | bool]]:
    """Estimate population law by deconvolving an MP kernel dictionary.

    This method approximates the MP inverse as a weighted mixture of
    single-atom MP densities and estimates mixture weights with a
    Richardson-Lucy fixed-point iteration.
    """
    support_min, support_max = _support_range(observed, aspect_ratio, inverse_settings)
    support = np.linspace(support_min, support_max, inverse_settings.n_support)

    kernels = np.column_stack([
        mp_dirac_density(observed.grid, variance=float(atom), aspect_ratio=aspect_ratio)
        for atom in support
    ])
    kernels = np.clip(kernels, 1e-16, None)

    # Normalize each column to avoid scaling bias in updates.
    col_norms = np.trapz(kernels, observed.grid, axis=0)
    col_norms = np.clip(col_norms, 1e-12, None)
    kernels = kernels / col_norms

    weights = np.full(inverse_settings.n_support, 1.0 / inverse_settings.n_support)
    history: list[float] = []
    converged = False

    obs_pdf = np.clip(observed.density, 1e-12, None)
    smooth_kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    smooth_strength = float(np.clip(inverse_settings.regularization, 0.0, 0.4))

    for _ in range(inverse_settings.max_iter):
        estimate = np.clip(kernels @ weights, 1e-12, None)
        ratio = obs_pdf / estimate
        updated = weights * (kernels.T @ ratio)
        updated = np.clip(updated, 1e-16, None)

        if smooth_strength > 0:
            smoothed = np.convolve(updated, smooth_kernel, mode="same")
            updated = (1.0 - smooth_strength) * updated + smooth_strength * smoothed

        updated = np.clip(updated, 1e-16, None)
        updated /= updated.sum()

        delta = float(np.linalg.norm(updated - weights, ord=1))
        history.append(delta)
        weights = updated
        if delta < inverse_settings.tol:
            converged = True
            break

    population = DiscreteSpectrum(atoms=support, weights=weights, name="inverse_fixed_point")
    reconstructed = GridDensity(grid=observed.grid, density=kernels @ weights, name="reconstructed")

    diagnostics = {
        "iterations": len(history),
        "converged": converged,
        "final_delta": history[-1] if history else 0.0,
        "mean_delta": float(np.mean(history)) if history else 0.0,
    }
    return population, reconstructed, diagnostics
