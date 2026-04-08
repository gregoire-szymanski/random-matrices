"""Experiment runner: simulation -> realized spectrum -> MP inverse."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison
from mpdiff.simulation.diffusion import simulate_diffusion
from mpdiff.simulation.volatility_segments import build_volatility_schedule
from mpdiff.spectral.empirical import empirical_eigenvalues, empirical_spectral_density, realized_covariance
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.inverse import compare_inverse_methods
from mpdiff.spectral.metrics import compare_grid_densities, discrete_to_grid
from mpdiff.spectral.transforms import compute_mp_forward
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import ensure_output_dir, integrated_population_spectrum, log_summary, resolve_aspect_ratio, save_density


def run_full_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run full end-to-end pipeline with simulated paths."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_full_pipeline")
    out_dir = ensure_output_dir(cfg)

    rng = make_rng(cfg.global_settings.seed)

    with timed_block("build_schedule", logger if cfg.benchmark.enabled else None):
        schedule = build_volatility_schedule(cfg, rng)

    with timed_block("simulate_diffusion", logger if cfg.benchmark.enabled else None):
        sim_result = simulate_diffusion(cfg.simulation, schedule, rng, logger=logger)

    with timed_block("realized_covariance", logger if cfg.benchmark.enabled else None):
        rcov = realized_covariance(sim_result.path, total_time=cfg.simulation.T)
        realized_eigs = empirical_eigenvalues(rcov)

    aspect_ratio = resolve_aspect_ratio(cfg)
    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)
    empirical_density = empirical_spectral_density(realized_eigs, grid=grid)

    with timed_block("mp_inverse_methods", logger if cfg.benchmark.enabled else None):
        methods = cfg.mp_inverse.compare_methods or [cfg.mp_inverse.method]
        inverse_results = compare_inverse_methods(
            observed=empirical_density,
            aspect_ratio=aspect_ratio,
            inverse_settings=cfg.mp_inverse,
            forward_settings=cfg.mp_forward,
            methods=methods,
        )

    reference_population = integrated_population_spectrum(schedule)
    reference_population_density = reference_population.to_grid_density(grid)

    reference_forward_result = compute_mp_forward(
        population=reference_population,
        c=aspect_ratio,
        grid=grid,
        epsilon=cfg.mp_forward.eta,
        tol=cfg.mp_forward.tol,
        max_iter=cfg.mp_forward.max_iter,
        damping=cfg.mp_forward.damping,
    )
    reference_forward = reference_forward_result.transformed_density

    method_metrics: dict[str, Any] = {}
    for method_name, result in inverse_results.items():
        est_density = discrete_to_grid(result.estimated_population, grid)
        pop_metrics = compare_grid_densities(reference_population_density, est_density)
        recon_metrics = compare_grid_densities(empirical_density, result.reconstructed_observed)
        method_metrics[method_name] = {
            "population_wasserstein_1": pop_metrics.wasserstein_1,
            "population_l2": pop_metrics.l2,
            "forward_reconstruction_l2": recon_metrics.l2,
            "estimated_population_mean": float(result.estimated_population.mean()),
            "diagnostics": result.diagnostics,
        }

    if cfg.global_settings.save_arrays:
        np.save(out_dir / "full_pipeline_realized_eigenvalues.npy", realized_eigs)
        save_density(out_dir / "full_pipeline_empirical_density.npz", empirical_density)
        save_density(out_dir / "full_pipeline_reference_population_density.npz", reference_population_density)
        save_density(out_dir / "full_pipeline_reference_forward_density.npz", reference_forward)
        for method_name, result in inverse_results.items():
            save_density(out_dir / f"full_pipeline_reconstructed_density_{method_name}.npz", result.reconstructed_observed)
            np.save(out_dir / f"full_pipeline_estimated_population_atoms_{method_name}.npy", result.estimated_population.atoms)
            np.save(out_dir / f"full_pipeline_estimated_population_weights_{method_name}.npy", result.estimated_population.weights)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)

        fig, _ = plot_density_comparison(
            densities=[empirical_density, reference_forward]
            + [result.reconstructed_observed for result in inverse_results.values()],
            labels=["empirical from path", "forward(reference population)"]
            + [f"inverse reconstructed ({name})" for name in inverse_results.keys()],
            title="Full Pipeline: Realized vs Reference Forward vs Inverse Reconstruction",
            figsize=cfg.plotting.figsize,
        )
        fig.savefig(out_dir / "full_pipeline_density_comparison.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        fig2, _ = plot_density_comparison(
            densities=[reference_population_density]
            + [discrete_to_grid(result.estimated_population, grid) for result in inverse_results.values()],
            labels=["reference population"] + [f"estimated population ({name})" for name in inverse_results.keys()],
            title="Full Pipeline: Population Recovery by Method",
            figsize=cfg.plotting.figsize,
        )
        fig2.savefig(out_dir / "full_pipeline_population_comparison.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig2)

    metadata = {
        "config_path": str(config_path),
        "aspect_ratio_c": aspect_ratio,
        "methods": methods,
        "method_metrics": method_metrics,
    }
    if cfg.global_settings.save_metadata:
        with (out_dir / "full_pipeline_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    primary_method = cfg.mp_inverse.method if cfg.mp_inverse.method in method_metrics else methods[0]
    primary_metrics = method_metrics[primary_method]

    summary = {
        "method": primary_method,
        "aspect_ratio": aspect_ratio,
        "reference_population_mean": float(reference_population.mean()),
        "estimated_population_mean": float(primary_metrics["estimated_population_mean"]),
        "population_wasserstein_1": float(primary_metrics["population_wasserstein_1"]),
        "forward_reconstruction_l2": float(primary_metrics["forward_reconstruction_l2"]),
    }
    log_summary(logger, "Full pipeline summary", summary)
    return summary
