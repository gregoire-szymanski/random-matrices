"""Helpers to benchmark MP inversion methods side by side."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.inverse import invert_mp_density, resolve_inverse_methods
from mpdiff.spectral.metrics import compare_grid_densities
from mpdiff.spectral.transforms import compute_mp_forward


@dataclass(slots=True)
class InversionBenchmarkResult:
    """Container for side-by-side inversion method benchmark outputs."""

    methods: list[str]
    observed_density: GridDensity
    method_results: dict[str, Any]
    summary_table: pd.DataFrame


def resolve_methods(inverse_settings: MPInverseConfig, methods: list[str] | None = None) -> list[str]:
    """Compatibility wrapper for method resolution in benchmark workflows."""
    return resolve_inverse_methods(inverse_settings, methods=methods)


def _clone_inverse_settings(inverse_settings: MPInverseConfig, method_name: str) -> MPInverseConfig:
    return MPInverseConfig(
        method=method_name,
        compare_all_methods=False,
        compare_methods=list(inverse_settings.compare_methods),
        n_support=inverse_settings.n_support,
        support_min=inverse_settings.support_min,
        support_max=inverse_settings.support_max,
        eta=inverse_settings.eta,
        tol=inverse_settings.tol,
        max_iter=inverse_settings.max_iter,
        regularization=inverse_settings.regularization,
        optimizer_max_iter=inverse_settings.optimizer_max_iter,
        forward_max_iter=inverse_settings.forward_max_iter,
        forward_tol=inverse_settings.forward_tol,
        fixed_point=inverse_settings.fixed_point,
        optimization=inverse_settings.optimization,
        stieltjes_based=inverse_settings.stieltjes_based,
        moment_based=inverse_settings.moment_based,
    )


def benchmark_inverse_methods_on_observed(
    observed_density: GridDensity,
    reference_population: DiscreteSpectrum,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
    methods: list[str] | None = None,
    density_bandwidth: float | None = None,
) -> InversionBenchmarkResult:
    """Benchmark inverse methods for a fixed observed density."""
    method_names = resolve_methods(inverse_settings, methods=methods)

    reference_density = reference_population.to_grid_density(observed_density.grid, bandwidth=density_bandwidth)

    method_results: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    for method_name in method_names:
        local_settings = _clone_inverse_settings(inverse_settings, method_name)

        t0 = perf_counter()
        inversion = invert_mp_density(
            observed=observed_density,
            aspect_ratio=aspect_ratio,
            inverse_settings=local_settings,
            forward_settings=forward_settings,
        )
        runtime = perf_counter() - t0

        estimated_density = inversion.estimated_population.to_grid_density(observed_density.grid, bandwidth=density_bandwidth)

        population_metrics = compare_grid_densities(reference_density, estimated_density)
        reconstruction_metrics = compare_grid_densities(observed_density, inversion.reconstructed_observed)

        row = {
            "method": method_name,
            "runtime_seconds": float(runtime),
            "estimated_population_mean": float(inversion.estimated_population.mean()),
            "population_l1": float(population_metrics.l1),
            "population_l2": float(population_metrics.l2),
            "population_wasserstein_1": float(population_metrics.wasserstein_1),
            "reconstruction_l1": float(reconstruction_metrics.l1),
            "reconstruction_l2": float(reconstruction_metrics.l2),
            "reconstruction_wasserstein_1": float(reconstruction_metrics.wasserstein_1),
        }
        rows.append(row)

        method_results[method_name] = {
            "inversion": inversion,
            "estimated_density": estimated_density,
            "population_metrics": population_metrics,
            "reconstruction_metrics": reconstruction_metrics,
            "runtime_seconds": float(runtime),
        }

    summary = pd.DataFrame(rows).sort_values(by=["population_wasserstein_1", "runtime_seconds"], ascending=[True, True])

    return InversionBenchmarkResult(
        methods=method_names,
        observed_density=observed_density,
        method_results=method_results,
        summary_table=summary,
    )


def benchmark_inverse_methods_from_population(
    population: DiscreteSpectrum,
    aspect_ratio: float,
    grid,
    forward_settings: MPForwardConfig,
    inverse_settings: MPInverseConfig,
    methods: list[str] | None = None,
    density_bandwidth: float | None = None,
) -> InversionBenchmarkResult:
    """Benchmark inverse methods from synthetic benchmark pipeline H -> MP(H)."""
    forward_result = compute_mp_forward(
        population=population,
        c=aspect_ratio,
        grid=grid,
        epsilon=forward_settings.eta,
        tol=forward_settings.tol,
        max_iter=forward_settings.max_iter,
        damping=forward_settings.damping,
    )

    return benchmark_inverse_methods_on_observed(
        observed_density=forward_result.transformed_density,
        reference_population=population,
        aspect_ratio=aspect_ratio,
        inverse_settings=inverse_settings,
        forward_settings=forward_settings,
        methods=methods,
        density_bandwidth=density_bandwidth,
    )
