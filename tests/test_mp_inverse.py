"""Tests for MP inverse methods."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum
from mpdiff.spectral.inverse import invert_mp_density
from mpdiff.spectral.transforms import mp_forward_transform


def test_mp_inverse_optimization_recovers_mean_reasonably() -> None:
    population = DiscreteSpectrum(
        atoms=np.array([0.8, 1.8]),
        weights=np.array([0.55, 0.45]),
    )
    aspect_ratio = 0.35
    grid = np.linspace(0.02, 5.0, 220)

    observed = mp_forward_transform(
        population=population,
        aspect_ratio=aspect_ratio,
        grid=grid,
        eta=0.003,
        tol=1e-9,
        max_iter=450,
        damping=0.7,
    )

    inverse_settings = MPInverseConfig(
        method="optimization",
        n_support=30,
        support_min=0.2,
        support_max=3.2,
        eta=0.003,
        regularization=1e-3,
        optimizer_max_iter=45,
        forward_max_iter=180,
        forward_tol=1e-7,
    )
    forward_settings = MPForwardConfig(damping=0.7)

    result = invert_mp_density(
        observed=observed,
        aspect_ratio=aspect_ratio,
        inverse_settings=inverse_settings,
        forward_settings=forward_settings,
    )

    mean_error = abs(result.estimated_population.mean() - population.mean())
    recon_rmse = float(np.sqrt(np.mean((result.reconstructed_observed.density - observed.density) ** 2)))

    assert mean_error < 0.25
    assert recon_rmse < 0.15
