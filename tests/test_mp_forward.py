"""Tests for MP forward transform and spectral-law inputs."""

from __future__ import annotations

import numpy as np

from mpdiff.spectral.densities import (
    GridDensity,
    dirac_law,
    dirac_mixture_law,
    gamma_law,
)
from mpdiff.spectral.transforms import compute_mp_forward, mp_dirac_density, mp_forward_transform


def test_mp_forward_dirac_matches_closed_form_reasonably() -> None:
    grid = np.linspace(0.01, 6.0, 500)
    c = 0.4
    variance = 2.0

    result = compute_mp_forward(
        population=dirac_law(variance),
        c=c,
        grid=grid,
        epsilon=0.003,
        tol=1e-10,
        max_iter=600,
        damping=0.72,
    )

    closed_form = mp_dirac_density(grid, variance=variance, c=c)
    numerical = result.transformed_density.density

    # Numerical density is eta-regularized, so compare with moderate tolerance.
    rel_l2 = np.linalg.norm(numerical - closed_form) / max(np.linalg.norm(closed_form), 1e-12)
    assert rel_l2 < 0.35


def test_mp_forward_finite_atomic_mixture_preserves_first_moment() -> None:
    population = dirac_mixture_law(
        values=np.array([0.6, 1.4, 2.2]),
        weights=np.array([0.25, 0.45, 0.30]),
    ).to_discrete(3)
    grid = np.linspace(0.02, 6.0, 420)

    density = mp_forward_transform(
        population=population,
        aspect_ratio=0.3,
        grid=grid,
        eta=0.003,
        tol=1e-10,
        max_iter=550,
        damping=0.72,
    )

    assert np.all(density.density >= 0.0)
    assert abs(density.moment(1) - population.mean()) < 0.2


def test_mp_forward_gamma_population_input_and_density_normalization() -> None:
    law = gamma_law(shape=2.5, scale=0.5)
    grid = np.linspace(0.01, 8.0, 500)

    result = compute_mp_forward(
        population=law,
        c=0.25,
        grid=grid,
        epsilon=0.004,
        tol=1e-9,
        max_iter=600,
        damping=0.72,
        n_population_atoms=450,
    )

    area = np.trapz(result.transformed_density.density, result.transformed_density.grid)
    assert 0.8 <= area <= 1.2
    assert result.diagnostics["convergence_rate"] > 0.95


def test_mp_forward_accepts_grid_density_input() -> None:
    grid = np.linspace(0.2, 3.0, 200)
    base = np.exp(-0.5 * ((grid - 1.2) / 0.35) ** 2)
    population_grid = GridDensity(grid=grid, density=base, name="grid_input")

    density = mp_forward_transform(
        population=population_grid,
        aspect_ratio=0.2,
        grid=np.linspace(0.01, 4.5, 260),
        eta=0.003,
        tol=1e-9,
        max_iter=450,
        damping=0.7,
    )
    assert np.all(np.isfinite(density.density))

