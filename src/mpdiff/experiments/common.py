"""Shared helpers for experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mpdiff.config.schemas import ProjectConfig
from mpdiff.simulation.covariance_builders import build_covariance_matrix
from mpdiff.simulation.volatility_segments import VolatilitySchedule
from mpdiff.spectral.densities import (
    DiscreteSpectrum,
    GridDensity,
    ParametricSpectrumLaw,
    empirical_discrete_law,
)


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


def _build_population_from_spectral_config(cfg: ProjectConfig) -> DiscreteSpectrum | None:
    spectral_cfg = cfg.analysis.population_spectrum
    if spectral_cfg is None:
        return None

    source = spectral_cfg.source
    if source == "from_covariance_model":
        return None

    if source == "parametric":
        law = ParametricSpectrumLaw(
            kind=spectral_cfg.kind,
            params=dict(spectral_cfg.params),
            name=f"population_{spectral_cfg.kind}",
        )
        return law.to_discrete(n_atoms=spectral_cfg.n_atoms)

    if source == "atomic":
        return DiscreteSpectrum(
            atoms=np.asarray(spectral_cfg.atoms, dtype=float),
            weights=np.asarray(spectral_cfg.weights, dtype=float) if spectral_cfg.weights else None,
            name="population_atomic",
        )

    if source == "grid":
        density = GridDensity(
            grid=np.asarray(spectral_cfg.grid, dtype=float),
            density=np.asarray(spectral_cfg.density, dtype=float),
            name="population_grid",
        )
        return density.to_discrete(n_atoms=spectral_cfg.n_atoms)

    if source == "empirical":
        if spectral_cfg.eigenvalues:
            values = np.asarray(spectral_cfg.eigenvalues, dtype=float)
        elif spectral_cfg.eigenvalues_path is not None:
            values = np.load(spectral_cfg.eigenvalues_path)
        else:
            raise ValueError("empirical spectral source requires eigenvalues or eigenvalues_path")
        return empirical_discrete_law(values, name="population_empirical")

    raise ValueError(f"Unsupported analysis.population_spectrum.source: {source}")


def build_population_spectrum(cfg: ProjectConfig, rng: np.random.Generator) -> DiscreteSpectrum:
    """Build reference population spectral law from config hierarchy."""
    spectral_population = _build_population_from_spectral_config(cfg)
    if spectral_population is not None:
        return spectral_population

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
