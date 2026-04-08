"""Moment-based MP inverse approximation."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.transforms import mp_forward_transform


def invert_moment_based(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> tuple[DiscreteSpectrum, GridDensity, dict[str, float | int | bool]]:
    """Estimate population law from low-order moment relations.

    We use
    ``m1(F)=m1(H)`` and ``m2(F)=m2(H)+c m1(H)^2``
    to infer mean and variance of ``H``, then fit a gamma law.
    """
    m1_f = observed.moment(1)
    m2_f = observed.moment(2)

    m1_h = max(m1_f, 1e-10)
    m2_h = max(m2_f - aspect_ratio * m1_h**2, 1e-10)
    var_h = max(m2_h - m1_h**2, 1e-10)

    if var_h < 1e-8:
        atoms = np.array([m1_h], dtype=float)
        weights = np.array([1.0], dtype=float)
        diagnostics = {"iterations": 1, "converged": True, "shape": float("inf"), "scale": 0.0}
    else:
        shape = m1_h**2 / var_h
        scale = var_h / m1_h
        probs = (np.arange(inverse_settings.n_support) + 0.5) / inverse_settings.n_support
        probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
        atoms = gamma_dist.ppf(probs, a=shape, scale=scale)
        weights = np.full(inverse_settings.n_support, 1.0 / inverse_settings.n_support)
        diagnostics = {
            "iterations": inverse_settings.n_support,
            "converged": True,
            "shape": float(shape),
            "scale": float(scale),
        }

    population = DiscreteSpectrum(atoms=atoms, weights=weights, name="inverse_moment_based")
    reconstructed = mp_forward_transform(
        population=population,
        aspect_ratio=aspect_ratio,
        grid=observed.grid,
        eta=inverse_settings.eta,
        tol=inverse_settings.forward_tol,
        max_iter=inverse_settings.forward_max_iter,
        damping=forward_settings.damping,
    )
    return population, reconstructed, diagnostics
