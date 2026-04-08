"""Shared helpers for experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mpdiff.config.schemas import ProjectConfig
from mpdiff.simulation.covariance_builders import build_covariance_matrix
from mpdiff.simulation.volatility_segments import VolatilitySchedule
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity


def ensure_output_dir(cfg: ProjectConfig) -> Path:
    """Create and return output directory from global config."""
    out_dir = Path(cfg.global_settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_aspect_ratio(cfg: ProjectConfig) -> float:
    """Aspect ratio c=d/n used for MP transforms."""
    if cfg.mp_forward.aspect_ratio is not None:
        return cfg.mp_forward.aspect_ratio
    return cfg.simulation.d / cfg.simulation.n_steps


def build_population_spectrum(cfg: ProjectConfig, rng: np.random.Generator) -> DiscreteSpectrum:
    """Build reference population spectral law from config hierarchy."""
    d = cfg.simulation.d
    if cfg.analysis.population_model is not None:
        cov = build_covariance_matrix(cfg.analysis.population_model, d=d, rng=rng).covariance
    elif cfg.volatility.mode == "constant":
        cov = build_covariance_matrix(cfg.volatility.constant_model, d=d, rng=rng).covariance
    elif cfg.volatility.mode == "piecewise" and cfg.volatility.segments:
        segment_model = cfg.volatility.segments[cfg.analysis.reference_segment_index].model
        if segment_model is None:
            raise ValueError("reference segment model is missing")
        cov = build_covariance_matrix(segment_model, d=d, rng=rng).covariance
    else:
        cov = build_covariance_matrix(cfg.volatility.scaled_base.base_model, d=d, rng=rng).covariance

    eigs = np.linalg.eigvalsh(cov)
    return DiscreteSpectrum(atoms=eigs, name="population")


def integrated_population_spectrum(schedule: VolatilitySchedule) -> DiscreteSpectrum:
    """Build spectral law of the time-averaged covariance matrix."""
    cov = schedule.integrated_covariance()
    eigs = np.linalg.eigvalsh(cov)
    return DiscreteSpectrum(atoms=eigs, name="integrated_population")


def save_density(path: Path, density: GridDensity) -> None:
    """Save a grid density as a NumPy archive."""
    np.savez(path, grid=density.grid, density=density.density)


def log_summary(logger: logging.Logger, title: str, summary: dict[str, float | int | str | bool]) -> None:
    """Emit sorted summary key-value pairs."""
    logger.info("%s", title)
    for key in sorted(summary.keys()):
        logger.info("  %s: %s", key, summary[key])
