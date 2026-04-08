"""Experiment runner: diffusion simulation and empirical spectrum."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.paths import plot_diffusion_paths
from mpdiff.plotting.spectra import plot_grid_density
from mpdiff.simulation.diffusion import simulate_diffusion
from mpdiff.simulation.volatility_segments import build_volatility_schedule
from mpdiff.spectral.empirical import empirical_eigenvalues, empirical_spectral_density, realized_covariance
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import ensure_output_dir, log_summary, save_density


def run_simulation(config_path: str | Path) -> dict[str, Any]:
    """Run simulation from config and persist core artifacts."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_simulation")

    out_dir = ensure_output_dir(cfg)
    rng = make_rng(cfg.global_settings.seed)

    with timed_block("build_volatility_schedule", logger if cfg.benchmark.enabled else None):
        schedule = build_volatility_schedule(cfg, rng)

    with timed_block("simulate_diffusion", logger if cfg.benchmark.enabled else None):
        sim_result = simulate_diffusion(cfg.simulation, schedule, rng, logger=logger)

    with timed_block("realized_covariance", logger if cfg.benchmark.enabled else None):
        rcov = realized_covariance(sim_result.path, total_time=cfg.simulation.T)
        eigs = empirical_eigenvalues(rcov)

    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)
    esd = empirical_spectral_density(eigs, grid=grid)

    if cfg.global_settings.save_arrays:
        np.save(out_dir / "simulated_path.npy", sim_result.path)
        np.save(out_dir / "realized_covariance.npy", rcov)
        np.save(out_dir / "realized_eigenvalues.npy", eigs)
        save_density(out_dir / "empirical_density.npz", esd)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)
        fig_path, _ = plot_diffusion_paths(sim_result.times, sim_result.path, max_dims=min(5, cfg.simulation.d), figsize=cfg.plotting.figsize)
        fig_path.savefig(out_dir / "diffusion_paths.png", dpi=cfg.plotting.dpi)
        plt.close(fig_path)

        fig, ax = plt.subplots(figsize=cfg.plotting.figsize)
        plot_grid_density(esd, ax=ax, label="empirical")
        ax.set_title("Empirical Spectral Density")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "empirical_spectrum.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig)

    summary = {
        "dimension": cfg.simulation.d,
        "n_steps": cfg.simulation.n_steps,
        "mean_realized_eigenvalue": float(np.mean(eigs)),
        "max_realized_eigenvalue": float(np.max(eigs)),
    }
    log_summary(logger, "Simulation summary", summary)
    return summary
