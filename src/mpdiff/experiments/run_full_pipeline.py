"""Experiment runner: simulation -> realized spectrum -> MP inverse."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison, plot_discrete_spectrum
from mpdiff.simulation.diffusion import simulate_diffusion
from mpdiff.simulation.volatility_segments import build_volatility_schedule
from mpdiff.spectral.empirical import empirical_eigenvalues, empirical_spectral_density, realized_covariance
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.inverse import invert_mp_density
from mpdiff.spectral.transforms import mp_forward_transform
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

    with timed_block("mp_inverse", logger if cfg.benchmark.enabled else None):
        inverse_result = invert_mp_density(
            observed=empirical_density,
            aspect_ratio=aspect_ratio,
            inverse_settings=cfg.mp_inverse,
            forward_settings=cfg.mp_forward,
        )

    reference_population = integrated_population_spectrum(schedule)
    reference_forward = mp_forward_transform(
        population=reference_population,
        aspect_ratio=aspect_ratio,
        grid=grid,
        eta=cfg.mp_forward.eta,
        tol=cfg.mp_forward.tol,
        max_iter=cfg.mp_forward.max_iter,
        damping=cfg.mp_forward.damping,
    )

    if cfg.global_settings.save_arrays:
        np.save(out_dir / "full_pipeline_realized_eigenvalues.npy", realized_eigs)
        save_density(out_dir / "full_pipeline_empirical_density.npz", empirical_density)
        save_density(out_dir / "full_pipeline_reconstructed_density.npz", inverse_result.reconstructed_observed)
        np.save(out_dir / "full_pipeline_estimated_population_atoms.npy", inverse_result.estimated_population.atoms)
        np.save(out_dir / "full_pipeline_estimated_population_weights.npy", inverse_result.estimated_population.weights)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)
        fig, _ = plot_density_comparison(
            densities=[empirical_density, inverse_result.reconstructed_observed, reference_forward],
            labels=["empirical from path", "inverse reconstructed", "forward(ref population)"],
            title="Full Pipeline: Realized vs Inverse vs Reference Forward",
            figsize=cfg.plotting.figsize,
        )
        fig.savefig(out_dir / "full_pipeline_density_comparison.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=cfg.plotting.figsize)
        plot_discrete_spectrum(reference_population, ax=ax2, label="reference population")
        plot_discrete_spectrum(inverse_result.estimated_population, ax=ax2, label="estimated population")
        ax2.set_title("Population Law: Reference vs Estimated")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(out_dir / "full_pipeline_population_comparison.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig2)

    mean_err = abs(inverse_result.estimated_population.mean() - reference_population.mean())
    summary = {
        "method": cfg.mp_inverse.method,
        "aspect_ratio": aspect_ratio,
        "reference_population_mean": float(reference_population.mean()),
        "estimated_population_mean": float(inverse_result.estimated_population.mean()),
        "population_mean_error": float(mean_err),
        "inverse_diagnostics": str(inverse_result.diagnostics),
    }
    log_summary(logger, "Full pipeline summary", summary)
    return summary
