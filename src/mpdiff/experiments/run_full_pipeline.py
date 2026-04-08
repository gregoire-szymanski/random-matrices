"""Experiment runner: simulation -> realized spectrum -> MP inverse.

Reference population law used for comparison is the eigenvalue law of the
time-averaged covariance (integrated over segments). For constant volatility,
this reduces to the model covariance itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison, plot_eigen_histogram
from mpdiff.simulation.diffusion import simulate_diffusion
from mpdiff.simulation.volatility_segments import build_volatility_schedule
from mpdiff.spectral.empirical import (
    compute_increments,
    empirical_discrete_spectrum,
    empirical_eigenvalues,
    empirical_spectral_density,
    realized_covariance_from_increments,
)
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.inverse import compare_inverse_methods, resolve_inverse_methods
from mpdiff.spectral.metrics import compare_grid_densities, discrete_to_grid
from mpdiff.spectral.transforms import compute_mp_forward
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import ensure_output_dir, integrated_population_spectrum, log_summary, resolve_aspect_ratio, save_density


def _timer_store(timer_dict: dict[str, float], label: str, elapsed_seconds: float) -> None:
    timer_dict[label] = float(elapsed_seconds)


def _report_text(
    config_path: str | Path,
    aspect_ratio: float,
    normalization: str,
    df_summary: pd.DataFrame,
    primary_method: str,
) -> str:
    lines = [
        "mpdiff full pipeline report",
        "",
        f"config_path: {config_path}",
        f"aspect_ratio_c: {aspect_ratio:.6g}",
        f"realized_covariance_normalization: {normalization}",
        f"primary_method: {primary_method}",
        "",
        "method summary (sorted by population_wasserstein_1):",
        df_summary.to_string(index=False),
    ]
    return "\n".join(lines)


def _float(value: Any) -> float:
    return float(np.asarray(value).item()) if np.asarray(value).shape == () else float(value)


def run_full_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Run full end-to-end pipeline with simulated paths and MP recovery.

    Pipeline:
    1. simulate diffusion paths,
    2. build realized covariance from increments,
    3. compute realized empirical spectrum,
    4. run MP inverse methods,
    5. compare recovered population law(s) against reference integrated law.
    """
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_full_pipeline")
    out_dir = ensure_output_dir(cfg)

    rng = make_rng(cfg.global_settings.seed)
    timers: dict[str, float] = {}

    with timed_block("build_schedule", logger if cfg.benchmark.enabled else None) as timer:
        schedule = build_volatility_schedule(cfg, rng)
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    with timed_block("simulation", logger if cfg.benchmark.enabled else None) as timer:
        sim_result = simulate_diffusion(cfg.simulation, schedule, rng, logger=logger)
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    with timed_block("compute_increments", logger if cfg.benchmark.enabled else None) as timer:
        increments = compute_increments(sim_result.path)
        increments_consistency_l2 = float(np.linalg.norm(increments - sim_result.increments))
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    normalization = cfg.analysis.realized_covariance_normalization
    with timed_block("realized_covariance", logger if cfg.benchmark.enabled else None) as timer:
        rcov = realized_covariance_from_increments(
            increments,
            normalization=normalization,
            total_time=cfg.simulation.T,
        )
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    with timed_block("eigendecomposition", logger if cfg.benchmark.enabled else None) as timer:
        realized_eigs = empirical_eigenvalues(rcov)
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    aspect_ratio = resolve_aspect_ratio(cfg)
    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)
    bandwidth = cfg.analysis.empirical_density_bandwidth or cfg.plotting.density_bandwidth

    with timed_block("empirical_density_estimation", logger if cfg.benchmark.enabled else None) as timer:
        empirical_discrete = empirical_discrete_spectrum(realized_eigs)
        empirical_density = empirical_spectral_density(
            realized_eigs,
            grid=grid,
            bandwidth=bandwidth,
        )
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    with timed_block("mp_inverse", logger if cfg.benchmark.enabled else None) as timer:
        methods = resolve_inverse_methods(cfg.mp_inverse)
        inverse_results = compare_inverse_methods(
            observed=empirical_density,
            aspect_ratio=aspect_ratio,
            inverse_settings=cfg.mp_inverse,
            forward_settings=cfg.mp_forward,
            methods=methods,
        )
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    with timed_block("reference_population", logger if cfg.benchmark.enabled else None) as timer:
        # Reference "true" law is the integrated covariance over the full horizon.
        reference_population = integrated_population_spectrum(schedule)
        reference_population_density = reference_population.to_grid_density(
            grid,
            bandwidth=bandwidth,
        )
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
    _timer_store(timers, timer.label, timer.elapsed_seconds)

    summary_rows: list[dict[str, Any]] = []
    method_metrics: dict[str, Any] = {}

    for method_name, result in inverse_results.items():
        estimated_density = discrete_to_grid(
            result.estimated_population,
            grid,
            bandwidth=bandwidth,
        )

        pop_metrics = compare_grid_densities(reference_population_density, estimated_density)
        reconstruction_metrics = compare_grid_densities(empirical_density, result.reconstructed_observed)

        row = {
            "method": method_name,
            "estimated_population_mean": float(result.estimated_population.mean()),
            "population_l1": float(pop_metrics.l1),
            "population_l2": float(pop_metrics.l2),
            "population_wasserstein_1": float(pop_metrics.wasserstein_1),
            "population_support_min_diff": float(pop_metrics.support_min_diff),
            "population_support_max_diff": float(pop_metrics.support_max_diff),
            "population_moment1_abs_error": float(pop_metrics.moment_abs_errors[0]),
            "population_moment2_abs_error": float(pop_metrics.moment_abs_errors[1]),
            "population_moment3_abs_error": float(pop_metrics.moment_abs_errors[2]),
            "reconstruction_l1": float(reconstruction_metrics.l1),
            "reconstruction_l2": float(reconstruction_metrics.l2),
            "reconstruction_wasserstein_1": float(reconstruction_metrics.wasserstein_1),
        }
        summary_rows.append(row)

        method_metrics[method_name] = {
            "population_recovery": {
                "l1": row["population_l1"],
                "l2": row["population_l2"],
                "wasserstein_1": row["population_wasserstein_1"],
                "support_min_diff": row["population_support_min_diff"],
                "support_max_diff": row["population_support_max_diff"],
                "moment_abs_errors": [
                    row["population_moment1_abs_error"],
                    row["population_moment2_abs_error"],
                    row["population_moment3_abs_error"],
                ],
            },
            "reconstruction": {
                "l1": row["reconstruction_l1"],
                "l2": row["reconstruction_l2"],
                "wasserstein_1": row["reconstruction_wasserstein_1"],
            },
            "diagnostics": result.diagnostics,
        }

    df_summary = pd.DataFrame(summary_rows).sort_values(by="population_wasserstein_1", ascending=True)

    primary_method = cfg.mp_inverse.method if cfg.mp_inverse.method in inverse_results else str(df_summary.iloc[0]["method"])
    primary_result = inverse_results[primary_method]
    primary_estimated_density = discrete_to_grid(
        primary_result.estimated_population,
        grid,
        bandwidth=bandwidth,
    )

    if cfg.global_settings.save_arrays:
        np.save(out_dir / "full_pipeline_times.npy", sim_result.times)
        np.save(out_dir / "full_pipeline_paths.npy", sim_result.path)
        np.save(out_dir / "full_pipeline_increments.npy", increments)
        np.save(out_dir / "full_pipeline_realized_covariance.npy", rcov)
        np.save(out_dir / "full_pipeline_realized_eigenvalues.npy", realized_eigs)
        np.save(out_dir / "full_pipeline_reference_population_eigenvalues.npy", reference_population.atoms)

        np.save(out_dir / "full_pipeline_empirical_discrete_atoms.npy", empirical_discrete.atoms)
        np.save(out_dir / "full_pipeline_empirical_discrete_weights.npy", empirical_discrete.weights)

        save_density(out_dir / "full_pipeline_empirical_density.npz", empirical_density)
        save_density(out_dir / "full_pipeline_reference_population_density.npz", reference_population_density)
        save_density(out_dir / "full_pipeline_reference_forward_density.npz", reference_forward)
        save_density(out_dir / f"full_pipeline_estimated_population_density_{primary_method}.npz", primary_estimated_density)

        for method_name, result in inverse_results.items():
            save_density(out_dir / f"full_pipeline_reconstructed_density_{method_name}.npz", result.reconstructed_observed)
            np.save(out_dir / f"full_pipeline_estimated_population_atoms_{method_name}.npy", result.estimated_population.atoms)
            np.save(out_dir / f"full_pipeline_estimated_population_weights_{method_name}.npy", result.estimated_population.weights)

    if cfg.global_settings.save_arrays or cfg.global_settings.save_metadata:
        df_summary.to_csv(out_dir / "full_pipeline_method_summary.csv", index=False)
        df_summary.to_json(out_dir / "full_pipeline_method_summary.json", orient="records", indent=2)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)

        fig1, _ = plot_density_comparison(
            densities=[reference_population_density, empirical_density, primary_estimated_density],
            labels=[
                "original population (reference)",
                "empirical realized covariance spectrum",
                f"recovered population ({primary_method})",
            ],
            title="Population vs Empirical Realized vs Recovered Population",
            figsize=cfg.plotting.figsize,
        )
        fig1.savefig(out_dir / "full_pipeline_overlay_population_empirical_recovered.png", dpi=cfg.plotting.dpi)
        plt.close(fig1)

        fig2, _ = plot_density_comparison(
            densities=[empirical_density, primary_result.reconstructed_observed, reference_forward],
            labels=[
                "empirical realized covariance density",
                f"forward(recovered population) [{primary_method}]",
                "forward(original population)",
            ],
            title="Observed vs MP-Reconstructed vs Reference Forward",
            figsize=cfg.plotting.figsize,
        )
        fig2.savefig(out_dir / "full_pipeline_overlay_observed_reconstructed_forward.png", dpi=cfg.plotting.dpi)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=cfg.plotting.figsize)
        bins = cfg.analysis.empirical_histogram_bins
        plot_eigen_histogram(realized_eigs, bins=bins, density=True, ax=ax3, label="realized covariance eigs", alpha=0.45)
        plot_eigen_histogram(reference_population.atoms, bins=bins, density=True, ax=ax3, label="reference population eigs", alpha=0.35)
        ax3.hist(
            primary_result.estimated_population.atoms,
            bins=bins,
            weights=primary_result.estimated_population.weights,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=f"recovered population atoms ({primary_method})",
        )
        ax3.set_title("Eigenvalue Histograms: Realized vs Reference vs Recovered")
        ax3.grid(alpha=0.2)
        ax3.legend(frameon=False)
        fig3.tight_layout()
        fig3.savefig(out_dir / "full_pipeline_eigen_histograms.png", dpi=cfg.plotting.dpi)
        plt.close(fig3)

        fig4, _ = plot_density_comparison(
            densities=[reference_population_density]
            + [
                discrete_to_grid(
                    inverse_results[name].estimated_population,
                    grid,
                    bandwidth=bandwidth,
                )
                for name in df_summary["method"].tolist()
            ],
            labels=["reference population"] + [f"recovered ({name})" for name in df_summary["method"].tolist()],
            title="Recovered Population Spectra by Inverse Method",
            figsize=cfg.plotting.figsize,
        )
        fig4.savefig(out_dir / "full_pipeline_method_population_comparison.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig4)

    report_text = _report_text(
        config_path=config_path,
        aspect_ratio=aspect_ratio,
        normalization=normalization,
        df_summary=df_summary,
        primary_method=primary_method,
    )

    metadata = {
        "runner": "full-pipeline",
        "config_path": str(config_path),
        "seed": cfg.global_settings.seed,
        "aspect_ratio_c": float(aspect_ratio),
        "methods": methods,
        "primary_method": primary_method,
        "timers_seconds": timers,
        "realized_covariance_normalization": normalization,
        "increments_consistency_l2": increments_consistency_l2,
        "method_metrics": method_metrics,
    }

    if cfg.global_settings.save_metadata:
        with (out_dir / "full_pipeline_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        with (out_dir / "full_pipeline_report.txt").open("w", encoding="utf-8") as handle:
            handle.write(report_text)

    primary_row = df_summary[df_summary["method"] == primary_method].iloc[0]
    summary = {
        "method": primary_method,
        "aspect_ratio": float(aspect_ratio),
        "reference_population_mean": float(reference_population.mean()),
        "estimated_population_mean": float(primary_result.estimated_population.mean()),
        "population_wasserstein_1": _float(primary_row["population_wasserstein_1"]),
        "reconstruction_l2": _float(primary_row["reconstruction_l2"]),
        "output_dir": str(out_dir),
    }

    if summary["reconstruction_l2"] > 0.15:
        logger.warning(
            "High reconstruction L2 (%.3f). Inverse recovery may be numerically unstable for this sample.",
            summary["reconstruction_l2"],
        )
    if df_summary.shape[0] >= 2:
        w1_spread = float(df_summary["population_wasserstein_1"].max() - df_summary["population_wasserstein_1"].min())
        if w1_spread > 0.25:
            logger.warning(
                "Inverse methods disagree on recovered population law (W1 spread=%.3f).",
                w1_spread,
            )

    log_summary(logger, "Full pipeline summary", summary)
    return summary
