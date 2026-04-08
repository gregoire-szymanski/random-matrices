"""Experiment runner: MP forward transform from population spectrum."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_density_comparison, plot_discrete_spectrum
from mpdiff.spectral.empirical import empirical_spectral_density
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.transforms import mp_forward_transform
from mpdiff.utils.logging_utils import setup_logging
from mpdiff.utils.random import make_rng
from mpdiff.utils.timers import timed_block

from .common import build_population_spectrum, ensure_output_dir, log_summary, resolve_aspect_ratio, save_density


def run_mp_forward(config_path: str | Path) -> dict[str, Any]:
    """Run MP forward transform and create diagnostic plots."""
    cfg = load_config(config_path)
    setup_logging(cfg.global_settings.log_level)
    logger = logging.getLogger("mpdiff.experiments.run_mp_forward")
    out_dir = ensure_output_dir(cfg)

    rng = make_rng(cfg.global_settings.seed)
    population = build_population_spectrum(cfg, rng)
    aspect_ratio = resolve_aspect_ratio(cfg)
    grid = make_linear_grid(cfg.mp_forward.grid_min, cfg.mp_forward.grid_max, cfg.mp_forward.num_points)

    with timed_block("mp_forward_transform", logger if cfg.benchmark.enabled else None):
        mp_density, diagnostics = mp_forward_transform(
            population=population,
            aspect_ratio=aspect_ratio,
            grid=grid,
            eta=cfg.mp_forward.eta,
            tol=cfg.mp_forward.tol,
            max_iter=cfg.mp_forward.max_iter,
            damping=cfg.mp_forward.damping,
            return_diagnostics=True,
        )

    pop_density = empirical_spectral_density(population.atoms, grid=grid)

    if cfg.global_settings.save_arrays:
        save_density(out_dir / "mp_forward_density.npz", mp_density)
        np.save(out_dir / "population_atoms.npy", population.atoms)
        np.save(out_dir / "population_weights.npy", population.weights)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)
        fig, _ = plot_density_comparison(
            densities=[pop_density, mp_density],
            labels=["population (smoothed)", "MP forward"],
            title="Population vs MP Forward Density",
            figsize=cfg.plotting.figsize,
        )
        fig.savefig(out_dir / "mp_forward_comparison.png", dpi=cfg.plotting.dpi)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=cfg.plotting.figsize)
        plot_discrete_spectrum(population, ax=ax2, label="population atoms")
        ax2.set_title("Population Discrete Spectrum")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(out_dir / "population_discrete_spectrum.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig2)

    summary = {
        "aspect_ratio": aspect_ratio,
        "population_mean": float(population.mean()),
        "forward_mean": float(mp_density.moment(1)),
        "convergence_rate": diagnostics["convergence_rate"],
        "mean_fixed_point_iterations": diagnostics["mean_iterations"],
    }
    log_summary(logger, "MP forward summary", summary)
    return summary
