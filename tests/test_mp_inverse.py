"""Tests for MP inverse methods and method comparison."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import dirac_law, dirac_mixture_law
from mpdiff.spectral.inverse import compare_inverse_methods, invert_mp_density
from mpdiff.spectral.metrics import compare_grid_densities
from mpdiff.spectral.transforms import compute_mp_forward


def _default_forward_config() -> MPForwardConfig:
    return MPForwardConfig(
        aspect_ratio=0.35,
        grid_min=0.02,
        grid_max=5.0,
        num_points=240,
        eta=0.003,
        tol=1e-9,
        max_iter=500,
        damping=0.72,
    )


def test_mp_inverse_optimization_recovers_mean_reasonably_for_atomic_mixture() -> None:
    population = dirac_mixture_law(
        values=np.array([0.8, 1.8]),
        weights=np.array([0.55, 0.45]),
    ).to_discrete(2)
    aspect_ratio = 0.35
    grid = np.linspace(0.02, 5.0, 220)

    observed = compute_mp_forward(
        population=population,
        c=aspect_ratio,
        grid=grid,
        epsilon=0.003,
        tol=1e-9,
        max_iter=500,
        damping=0.72,
    ).transformed_density

    inverse_settings = MPInverseConfig(
        method="optimization",
        n_support=28,
        support_min=0.2,
        support_max=3.2,
        eta=0.003,
        regularization=8e-4,
        optimizer_max_iter=40,
        forward_max_iter=180,
        forward_tol=1e-7,
    )
    forward_settings = MPForwardConfig(damping=0.72)

    result = invert_mp_density(
        observed=observed,
        aspect_ratio=aspect_ratio,
        inverse_settings=inverse_settings,
        forward_settings=forward_settings,
    )

    mean_error = abs(result.estimated_population.mean() - population.mean())
    recon_rmse = float(np.sqrt(np.mean((result.reconstructed_observed.density - observed.density) ** 2)))

    assert mean_error < 0.3
    assert recon_rmse < 0.2


def test_compare_inverse_methods_runs_requested_methods() -> None:
    population = dirac_law(1.4).to_discrete(1)
    c = 0.25
    grid = np.linspace(0.01, 4.0, 180)

    observed = compute_mp_forward(
        population=population,
        c=c,
        grid=grid,
        epsilon=0.003,
        tol=1e-9,
        max_iter=450,
        damping=0.7,
    ).transformed_density

    inv_cfg = MPInverseConfig(
        method="optimization",
        compare_methods=["optimization", "moment_based", "fixed_point", "stieltjes_based"],
        n_support=32,
        support_min=0.2,
        support_max=2.5,
        eta=0.003,
        max_iter=200,
        optimizer_max_iter=35,
        forward_max_iter=140,
        forward_tol=1e-7,
    )

    results = compare_inverse_methods(
        observed=observed,
        aspect_ratio=c,
        inverse_settings=inv_cfg,
        forward_settings=_default_forward_config(),
    )

    assert set(results.keys()) == {"optimization", "moment_based", "fixed_point", "stieltjes_based"}
    for result in results.values():
        assert np.isfinite(result.estimated_population.mean())
        assert np.all(result.reconstructed_observed.density >= 0.0)


def test_forward_inverse_dirac_pipeline_is_reasonable_for_moment_method() -> None:
    population = dirac_law(1.2).to_discrete(1)
    c = 0.2
    grid = np.linspace(0.01, 3.8, 200)

    forward = compute_mp_forward(
        population=population,
        c=c,
        grid=grid,
        epsilon=0.003,
        tol=1e-9,
        max_iter=500,
        damping=0.72,
    )

    inv_cfg = MPInverseConfig(
        method="moment_based",
        n_support=30,
        support_min=0.1,
        support_max=2.5,
        eta=0.003,
        forward_max_iter=160,
        forward_tol=1e-7,
    )

    inverse = invert_mp_density(
        observed=forward.transformed_density,
        aspect_ratio=c,
        inverse_settings=inv_cfg,
        forward_settings=_default_forward_config(),
    )

    recovered_population_density = inverse.estimated_population.to_grid_density(grid)
    original_population_density = population.to_grid_density(grid)
    metrics = compare_grid_densities(original_population_density, recovered_population_density)

    assert metrics.wasserstein_1 < 0.5
