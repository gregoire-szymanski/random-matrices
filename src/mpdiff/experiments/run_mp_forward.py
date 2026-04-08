"""Experiment runner: MP forward transform from population spectrum."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.config.loader import load_config
from mpdiff.plotting.spectra import plot_population_forward_recovered
from mpdiff.spectral.grids import make_linear_grid
from mpdiff.spectral.transforms import compute_mp_forward
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
        forward_result = compute_mp_forward(
            population=population,
            c=aspect_ratio,
            grid=grid,
            epsilon=cfg.mp_forward.eta,
            tol=cfg.mp_forward.tol,
            max_iter=cfg.mp_forward.max_iter,
            damping=cfg.mp_forward.damping,
        )

    mp_density = forward_result.transformed_density
    pop_density = population.to_grid_density(grid)

    if cfg.global_settings.save_arrays:
        save_density(out_dir / "mp_forward_density.npz", mp_density)
        save_density(out_dir / "population_density.npz", pop_density)
        np.save(out_dir / "population_atoms.npy", population.atoms)
        np.save(out_dir / "population_weights.npy", population.weights)
        np.save(out_dir / "mp_forward_stieltjes.npy", forward_result.stieltjes_values)

    if cfg.global_settings.save_figures:
        plt.style.use(cfg.plotting.style)
        fig, _ = plot_population_forward_recovered(
            population_density=pop_density,
            forward_density=mp_density,
            figsize=cfg.plotting.figsize,
            title="Population vs MP Forward Density",
        )
        fig.savefig(out_dir / "mp_forward_comparison.png", dpi=cfg.plotting.dpi)
        if cfg.plotting.show:
            plt.show()
        plt.close(fig)

    metadata = {
        "config_path": str(config_path),
        "aspect_ratio_c": aspect_ratio,
        "diagnostics": {
            key: value
            for key, value in forward_result.diagnostics.items()
            if key not in {"iterations", "residuals", "converged_mask", "used_newton_fallback"}
        },
    }

    if cfg.global_settings.save_metadata:
        with (out_dir / "mp_forward_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    summary = {
        "aspect_ratio": aspect_ratio,
        "population_mean": float(population.mean()),
        "forward_mean": float(mp_density.moment(1)),
        "convergence_rate": forward_result.diagnostics["convergence_rate"],
        "mean_fixed_point_iterations": forward_result.diagnostics["mean_iterations"],
        "newton_fallback_rate": forward_result.diagnostics["newton_fallback_rate"],
    }
    log_summary(logger, "MP forward summary", summary)
    return summary
