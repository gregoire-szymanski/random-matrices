"""Experiment runner: MP inverse from synthetic observed spectrum."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.diagnostics import plot_convergence_curve
from mpdiff.plotting.spectra import plot_density_comparison, plot_population_forward_recovered
from mpdiff.spectral.inverse import compare_inverse_methods
from mpdiff.spectral.metrics import compare_grid_densities, discrete_to_grid
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.transforms import compute_mp_forward
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import build_population_spectrum, ensure_output_dir, log_summary, resolve_aspect_ratio, save_density


def _compact_metrics(metrics) -> dict[str, float | list[float]]:
    return {
        "l1": metrics.l1,
        "l2": metrics.l2,
        "wasserstein_1": metrics.wasserstein_1,
        "support_min_diff": metrics.support_min_diff,
        "support_max_diff": metrics.support_max_diff,
        "moment_abs_errors": metrics.moment_abs_errors,
    }


def run_mp_inverse(config_path: str | Path) -> dict[str, Any]:
    """Run inverse recovery benchmark: population -> forward -> inverse."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_mp_inverse")
    out_dir = ensure_output_dir(cfg)

    rng = make_rng(cfg.global_settings.seed)
    population = build_population_spectrum(cfg, rng)
    aspect_ratio = resolve_aspect_ratio(cfg)
    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)

    with timed_block("mp_forward_for_inverse", logger if cfg.benchmark.enabled else None):
        observed_result = compute_mp_forward(
            population=population,
            c=aspect_ratio,
            grid=grid,
            epsilon=cfg.mp_forward.eta,
            tol=cfg.mp_forward.tol,
            max_iter=cfg.mp_forward.max_iter,
            damping=cfg.mp_forward.damping,
        )

    observed = observed_result.transformed_density
    methods = cfg.mp_inverse.compare_methods or [cfg.mp_inverse.method]

    with timed_block("mp_inverse_methods", logger if cfg.benchmark.enabled else None):
        inverse_results = compare_inverse_methods(
            observed=observed,
            aspect_ratio=aspect_ratio,
            inverse_settings=cfg.mp_inverse,
            forward_settings=cfg.mp_forward,
            methods=methods,
        )

    population_density = population.to_grid_density(grid)

    method_summaries: dict[str, dict[str, Any]] = {}
    for method_name, result in inverse_results.items():
        estimated_density = discrete_to_grid(result.estimated_population, grid)
        recovered_metrics = compare_grid_densities(population_density, estimated_density)
        reconstruction_metrics = compare_grid_densities(observed, result.reconstructed_observed)

        method_summaries[method_name] = {
            "population_recovery": _compact_metrics(recovered_metrics),
            "forward_reconstruction": _compact_metrics(reconstruction_metrics),
            "diagnostics": result.diagnostics,
            "estimated_mean": float(result.estimated_population.mean()),
        }

        if cfg.global_settings.save_arrays:
            np.save(out_dir / f"inverse_population_atoms_{method_name}.npy", result.estimated_population.atoms)
            np.save(out_dir / f"inverse_population_weights_{method_name}.npy", result.estimated_population.weights)
            save_density(out_dir / f"inverse_reconstructed_density_{method_name}.npz", result.reconstructed_observed)
            save_density(out_dir / f"inverse_estimated_population_density_{method_name}.npz", estimated_density)

        if cfg.global_settings.save_figures:
            objective_history = result.diagnostics.get("objective_history")
            if isinstance(objective_history, list) and objective_history:
                fig_conv, ax_conv = plt.subplots(figsize=cfg.plotting.figsize)
                plot_convergence_curve(
                    objective_history,
                    ylabel="objective",
                    title=f"Inverse Convergence ({method_name})",
                    ax=ax_conv,
                )
                fig_conv.tight_layout()
                fig_conv.savefig(out_dir / f"mp_inverse_convergence_{method_name}.png", dpi=cfg.plotting.dpi)
                plt.close(fig_conv)

    if cfg.global_settings.save_arrays:
        save_density(out_dir / "inverse_observed_density.npz", observed)
        save_density(out_dir / "inverse_population_density_true.npz", population_density)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)

        fig, ax = plt.subplots(figsize=cfg.plotting.figsize)
        ax.plot(observed.grid, observed.density, linewidth=2.2, label="observed MP density")
        for method_name, result in inverse_results.items():
            ax.plot(
                result.reconstructed_observed.grid,
                result.reconstructed_observed.density,
                linewidth=1.8,
                label=f"reconstructed ({method_name})",
            )
        ax.set_title("MP Inverse: Observed vs Reconstructed")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "mp_inverse_density_reconstruction.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        primary = inverse_results[cfg.mp_inverse.method] if cfg.mp_inverse.method in inverse_results else next(iter(inverse_results.values()))
        primary_estimated_density = discrete_to_grid(primary.estimated_population, grid)
        fig2, _ = plot_population_forward_recovered(
            population_density=population_density,
            forward_density=observed,
            recovered_density=primary_estimated_density,
            figsize=cfg.plotting.figsize,
            title=f"Population -> MP Forward -> Inverse ({primary.method})",
        )
        fig2.savefig(out_dir / "mp_inverse_population_vs_recovered.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig2)

        fig3, _ = plot_density_comparison(
            densities=[population_density]
            + [discrete_to_grid(res.estimated_population, grid) for res in inverse_results.values()],
            labels=["true population"] + [f"estimated ({name})" for name in inverse_results.keys()],
            title="Population Recovery Across Inverse Methods",
            figsize=cfg.plotting.figsize,
        )
        fig3.savefig(out_dir / "mp_inverse_method_comparison.png", dpi=cfg.plotting.dpi)
        plt.close(fig3)

    metadata = {
        "config_path": str(config_path),
        "aspect_ratio_c": aspect_ratio,
        "methods": methods,
        "method_summaries": method_summaries,
    }
    if cfg.global_settings.save_metadata:
        with (out_dir / "mp_inverse_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    primary_method = cfg.mp_inverse.method if cfg.mp_inverse.method in method_summaries else methods[0]
    primary_summary = method_summaries[primary_method]

    summary = {
        "method": primary_method,
        "aspect_ratio": aspect_ratio,
        "population_mean": float(population.mean()),
        "estimated_mean": float(primary_summary["estimated_mean"]),
        "population_wasserstein_1": float(primary_summary["population_recovery"]["wasserstein_1"]),
        "forward_reconstruction_l2": float(primary_summary["forward_reconstruction"]["l2"]),
    }
    log_summary(logger, "MP inverse summary", summary)
    return summary
