"""Experiment runner focused on diffusion simulation workflows."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.paths import plot_diffusion_paths
from mpdiff.plotting.spectra import plot_density_comparison, plot_eigen_histogram
from mpdiff.simulation.diffusion import simulate_diffusion
from mpdiff.simulation.volatility_segments import build_volatility_schedule
from mpdiff.spectral.empirical import empirical_eigenvalues, empirical_spectral_density, realized_covariance
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import ensure_output_dir, log_summary, save_density


def _save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def run_simulation(config_path: str | Path) -> dict[str, Any]:
    """Run diffusion simulation and save arrays/plots/metadata."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_simulation")

    out_dir = ensure_output_dir(cfg)
    rng = make_rng(cfg.global_settings.seed)
    timers: dict[str, float] = {}

    with timed_block("build_volatility_schedule", logger if cfg.benchmark.enabled else None) as timer:
        schedule = build_volatility_schedule(cfg, rng)
    timers[timer.label] = timer.elapsed_seconds

    with timed_block("simulate_diffusion", logger if cfg.benchmark.enabled else None) as timer:
        sim_result = simulate_diffusion(cfg.simulation, schedule, rng, logger=logger)
    timers[timer.label] = timer.elapsed_seconds

    with timed_block("realized_covariance", logger if cfg.benchmark.enabled else None) as timer:
        rcov = realized_covariance(sim_result.path, total_time=cfg.simulation.T)
        realized_eigs = empirical_eigenvalues(rcov)
        target_cov = schedule.integrated_covariance()
        target_eigs = np.linalg.eigvalsh(target_cov)
    timers[timer.label] = timer.elapsed_seconds

    with timed_block("density_estimation", logger if cfg.benchmark.enabled else None) as timer:
        data_min = min(float(np.min(realized_eigs)), float(np.min(target_eigs)))
        data_max = max(float(np.max(realized_eigs)), float(np.max(target_eigs)))
        span = max(data_max - data_min, 1e-6)
        grid_min = min(cfg.mp_forward.grid_min, max(0.0, data_min - 0.05 * span))
        grid_max = max(cfg.mp_forward.grid_max, data_max + 0.05 * span)
        grid = make_linear_grid(grid_min, grid_max, cfg.mp_forward.num_points)
        realized_density = empirical_spectral_density(
            realized_eigs,
            grid=grid,
            bandwidth=cfg.plotting.density_bandwidth,
        )
        target_density = empirical_spectral_density(
            target_eigs,
            grid=grid,
            bandwidth=cfg.plotting.density_bandwidth,
        )
    timers[timer.label] = timer.elapsed_seconds

    if cfg.global_settings.save_arrays:
        np.save(out_dir / "times.npy", sim_result.times)
        np.save(out_dir / "paths.npy", sim_result.path)
        np.save(out_dir / "increments.npy", sim_result.increments)
        np.save(out_dir / "segment_indices.npy", sim_result.segment_indices)
        np.save(out_dir / "realized_covariance.npy", rcov)
        np.save(out_dir / "realized_eigenvalues.npy", realized_eigs)
        np.save(out_dir / "target_population_eigenvalues.npy", target_eigs)
        np.save(out_dir / "target_integrated_covariance.npy", target_cov)
        save_density(out_dir / "realized_density.npz", realized_density)
        save_density(out_dir / "target_density.npz", target_density)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)

        if cfg.plotting.plot_paths:
            fig, _ = plot_diffusion_paths(
                sim_result.times,
                sim_result.path,
                max_dims=min(cfg.plotting.max_path_dims, cfg.simulation.d),
                figsize=cfg.plotting.figsize,
            )
            fig.savefig(out_dir / "paths_sample.png", dpi=cfg.plotting.dpi)
            plt.close(fig)

        if cfg.plotting.plot_eigen_hist:
            fig, ax = plt.subplots(figsize=cfg.plotting.figsize)
            plot_eigen_histogram(
                realized_eigs,
                bins=cfg.plotting.eigen_hist_bins,
                density=True,
                ax=ax,
                label="realized eigenvalues",
                alpha=0.5,
            )
            plot_eigen_histogram(
                target_eigs,
                bins=cfg.plotting.eigen_hist_bins,
                density=True,
                ax=ax,
                label="target population eigenvalues",
                alpha=0.35,
            )
            ax.set_title("Eigenvalue Histogram: Realized vs Target")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_dir / "eigen_histogram.png", dpi=cfg.plotting.dpi)
            plt.close(fig)

        if cfg.plotting.plot_eigen_density:
            fig, _ = plot_density_comparison(
                densities=[realized_density, target_density],
                labels=["realized (KDE)", "target population (KDE)"],
                title="Eigenvalue Density: Realized vs Target",
                figsize=cfg.plotting.figsize,
            )
            fig.savefig(out_dir / "eigen_density_comparison.png", dpi=cfg.plotting.dpi)
            plt.close(fig)

        if cfg.plotting.show:
            plt.show()

    metadata: dict[str, Any] = {
        "runner": "simulation",
        "config_path": str(config_path),
        "seed": cfg.global_settings.seed,
        "dimension": cfg.simulation.d,
        "n_steps": cfg.simulation.n_steps,
        "horizon_T": cfg.simulation.T,
        "volatility_mode": cfg.volatility.mode,
        "drift_model": asdict(cfg.simulation.drift_model),
        "timers_seconds": timers,
        "realized_eigenvalue_mean": float(np.mean(realized_eigs)),
        "target_eigenvalue_mean": float(np.mean(target_eigs)),
    }

    if cfg.global_settings.save_metadata:
        _save_metadata(out_dir / "metadata.json", metadata)

    summary = {
        "dimension": cfg.simulation.d,
        "n_steps": cfg.simulation.n_steps,
        "volatility_mode": cfg.volatility.mode,
        "drift_kind": cfg.simulation.drift_model.kind,
        "mean_realized_eigenvalue": float(np.mean(realized_eigs)),
        "mean_target_eigenvalue": float(np.mean(target_eigs)),
        "output_dir": str(out_dir),
    }
    log_summary(logger, "Simulation summary", summary)
    return summary
