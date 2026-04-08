"""Experiment runner: MP inverse from synthetic observed spectrum."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison, plot_discrete_spectrum
from mpdiff.spectral.inverse import invert_mp_density
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.transforms import mp_forward_transform
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import build_population_spectrum, ensure_output_dir, log_summary, resolve_aspect_ratio, save_density


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

    observed = mp_forward_transform(
        population=population,
        aspect_ratio=aspect_ratio,
        grid=grid,
        eta=cfg.mp_forward.eta,
        tol=cfg.mp_forward.tol,
        max_iter=cfg.mp_forward.max_iter,
        damping=cfg.mp_forward.damping,
    )

    with timed_block("mp_inverse", logger if cfg.benchmark.enabled else None):
        inverse_result = invert_mp_density(
            observed=observed,
            aspect_ratio=aspect_ratio,
            inverse_settings=cfg.mp_inverse,
            forward_settings=cfg.mp_forward,
        )

    mean_err = abs(inverse_result.estimated_population.mean() - population.mean())
    recon_err = float(np.sqrt(np.mean((inverse_result.reconstructed_observed.density - observed.density) ** 2)))

    if cfg.global_settings.save_arrays:
        save_density(out_dir / "inverse_observed_density.npz", observed)
        save_density(out_dir / "inverse_reconstructed_density.npz", inverse_result.reconstructed_observed)
        np.save(out_dir / "inverse_population_atoms.npy", inverse_result.estimated_population.atoms)
        np.save(out_dir / "inverse_population_weights.npy", inverse_result.estimated_population.weights)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)
        fig, _ = plot_density_comparison(
            densities=[observed, inverse_result.reconstructed_observed],
            labels=["observed", "reconstructed"],
            title=f"MP Inverse Reconstruction ({cfg.mp_inverse.method})",
            figsize=cfg.plotting.figsize,
        )
        fig.savefig(out_dir / "mp_inverse_density_reconstruction.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=cfg.plotting.figsize)
        plot_discrete_spectrum(population, ax=ax2, label="true population")
        plot_discrete_spectrum(inverse_result.estimated_population, ax=ax2, label="estimated population")
        ax2.set_title("Population Law Recovery")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(out_dir / "mp_inverse_population_recovery.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig2)

    summary = {
        "method": cfg.mp_inverse.method,
        "aspect_ratio": aspect_ratio,
        "population_mean": float(population.mean()),
        "estimated_mean": float(inverse_result.estimated_population.mean()),
        "mean_error": float(mean_err),
        "reconstruction_rmse": recon_err,
        "diagnostics": str(inverse_result.diagnostics),
    }
    log_summary(logger, "MP inverse summary", summary)
    return summary
