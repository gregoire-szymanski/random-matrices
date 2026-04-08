"""Tests for inversion benchmark helper layer."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.experiments.inversion_benchmark import benchmark_inverse_methods_from_population, resolve_methods
from mpdiff.spectral.densities import dirac_law


def test_resolve_methods_compare_all_mode() -> None:
    cfg = MPInverseConfig(method="optimization", compare_all_methods=True)
    methods = resolve_methods(cfg)
    assert set(methods) >= {"optimization", "fixed_point", "stieltjes_based", "moment_based"}


def test_benchmark_inverse_methods_from_population_returns_summary() -> None:
    population = dirac_law(1.2).to_discrete(1)
    grid = np.linspace(0.01, 3.5, 160)

    forward_cfg = MPForwardConfig(
        aspect_ratio=0.2,
        grid_min=float(grid[0]),
        grid_max=float(grid[-1]),
        num_points=grid.size,
        eta=0.003,
        tol=1e-9,
        max_iter=300,
        damping=0.72,
    )
    inverse_cfg = MPInverseConfig(
        method="moment_based",
        compare_methods=["moment_based", "optimization"],
        n_support=20,
        support_min=0.05,
        support_max=2.5,
        eta=0.003,
        max_iter=150,
        optimizer_max_iter=30,
        forward_max_iter=140,
        forward_tol=1e-7,
    )

    result = benchmark_inverse_methods_from_population(
        population=population,
        aspect_ratio=0.2,
        grid=grid,
        forward_settings=forward_cfg,
        inverse_settings=inverse_cfg,
        methods=None,
    )

    assert result.summary_table.shape[0] == 2
    assert set(result.summary_table["method"]) == {"moment_based", "optimization"}
