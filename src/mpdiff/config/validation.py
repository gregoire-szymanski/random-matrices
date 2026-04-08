"""Configuration validation utilities."""

from __future__ import annotations

from typing import Iterable

from .schemas import (
    CovarianceModelConfig,
    EigenDistributionConfig,
    ProjectConfig,
)


class ConfigValidationError(ValueError):
    """Raised when a project configuration is invalid."""


def _validate_distribution(cfg: EigenDistributionConfig, prefix: str) -> None:
    allowed = {"dirac", "dirac_mixture", "uniform", "gamma", "rescaled_beta"}
    if cfg.kind not in allowed:
        raise ConfigValidationError(f"{prefix}: unsupported distribution kind '{cfg.kind}'")
    if cfg.kind == "dirac" and cfg.value < 0.0:
        raise ConfigValidationError(f"{prefix}: dirac value must be non-negative")
    if cfg.kind == "dirac_mixture":
        if len(cfg.values) == 0:
            raise ConfigValidationError(f"{prefix}: dirac_mixture needs non-empty values")
        if cfg.weights and len(cfg.weights) != len(cfg.values):
            raise ConfigValidationError(f"{prefix}: dirac_mixture weights length mismatch")
    if cfg.kind == "uniform" and not (cfg.high > cfg.low >= 0):
        raise ConfigValidationError(f"{prefix}: uniform requires 0 <= low < high")
    if cfg.kind == "gamma" and (cfg.shape <= 0 or cfg.scale <= 0):
        raise ConfigValidationError(f"{prefix}: gamma requires positive shape and scale")
    if cfg.kind == "rescaled_beta" and (cfg.alpha <= 0 or cfg.beta <= 0 or cfg.a <= 0):
        raise ConfigValidationError(f"{prefix}: rescaled_beta requires alpha,beta,a > 0")


def _validate_covariance_model(cfg: CovarianceModelConfig, prefix: str) -> None:
    allowed = {"diag_scalar", "diag_distribution", "orthogonal_diag", "low_rank_plus_diag"}
    if cfg.kind not in allowed:
        raise ConfigValidationError(f"{prefix}: unsupported covariance kind '{cfg.kind}'")
    if cfg.kind == "diag_scalar" and cfg.scalar < 0:
        raise ConfigValidationError(f"{prefix}: scalar must be non-negative")
    if cfg.kind in {"diag_distribution", "orthogonal_diag"}:
        _validate_distribution(cfg.eigen_distribution, f"{prefix}.eigen_distribution")
    if cfg.kind == "low_rank_plus_diag":
        if cfg.low_rank.rank <= 0:
            raise ConfigValidationError(f"{prefix}.low_rank.rank must be positive")
        _validate_distribution(cfg.low_rank.latent_eigen_distribution, f"{prefix}.low_rank.latent_eigen_distribution")
        _validate_distribution(cfg.low_rank.diag_eigen_distribution, f"{prefix}.low_rank.diag_eigen_distribution")


def _validate_segments_cover_horizon(segments: Iterable, horizon: float) -> None:
    segments_list = list(segments)
    if not segments_list:
        raise ConfigValidationError("volatility.segments must not be empty")
    prev_end = 0.0
    for idx, segment in enumerate(segments_list):
        if segment.start >= segment.end:
            raise ConfigValidationError(f"segment {idx}: start must be < end")
        if abs(segment.start - prev_end) > 1e-10:
            raise ConfigValidationError("segments must be contiguous and ordered")
        prev_end = segment.end
    if abs(prev_end - horizon) > 1e-10:
        raise ConfigValidationError("segments must end exactly at simulation.T")


def validate_config(cfg: ProjectConfig) -> None:
    """Validate the project configuration in-place.

    Raises
    ------
    ConfigValidationError
        If one or more consistency checks fail.
    """
    if cfg.simulation.d <= 0:
        raise ConfigValidationError("simulation.d must be positive")
    if cfg.simulation.n_steps <= 0:
        raise ConfigValidationError("simulation.n_steps must be positive")
    if cfg.simulation.T <= 0:
        raise ConfigValidationError("simulation.T must be positive")

    if isinstance(cfg.simulation.drift, list) and len(cfg.simulation.drift) not in {1, cfg.simulation.d}:
        raise ConfigValidationError("simulation.drift list length must be 1 or d")
    if isinstance(cfg.simulation.initial_state, list) and len(cfg.simulation.initial_state) not in {1, cfg.simulation.d}:
        raise ConfigValidationError("simulation.initial_state list length must be 1 or d")

    if cfg.simulation.time_grid is not None:
        if len(cfg.simulation.time_grid) != cfg.simulation.n_steps + 1:
            raise ConfigValidationError("simulation.time_grid must have n_steps + 1 points")
        if abs(cfg.simulation.time_grid[0]) > 1e-12:
            raise ConfigValidationError("simulation.time_grid must start at 0")
        if abs(cfg.simulation.time_grid[-1] - cfg.simulation.T) > 1e-12:
            raise ConfigValidationError("simulation.time_grid must end at T")

    if cfg.volatility.mode == "constant":
        _validate_covariance_model(cfg.volatility.constant_model, "volatility.constant_model")
    elif cfg.volatility.mode == "piecewise":
        _validate_segments_cover_horizon(cfg.volatility.segments, cfg.simulation.T)
        for idx, segment in enumerate(cfg.volatility.segments):
            if segment.model is None:
                raise ConfigValidationError(f"segment {idx}: model is required in piecewise mode")
            _validate_covariance_model(segment.model, f"volatility.segments[{idx}].model")
            if segment.scalar <= 0:
                raise ConfigValidationError(f"segment {idx}: scalar must be positive")
    elif cfg.volatility.mode == "piecewise_scaled_base":
        _validate_segments_cover_horizon(cfg.volatility.segments, cfg.simulation.T)
        _validate_covariance_model(cfg.volatility.scaled_base.base_model, "volatility.scaled_base.base_model")
        if cfg.volatility.scaled_base.base_matrix_policy not in {"fixed_once", "redraw_per_segment"}:
            raise ConfigValidationError("volatility.scaled_base.base_matrix_policy must be fixed_once or redraw_per_segment")
        for idx, segment in enumerate(cfg.volatility.segments):
            if segment.scalar <= 0:
                raise ConfigValidationError(f"segment {idx}: scalar must be positive")
            if segment.model is not None and not cfg.volatility.scaled_base.share_matrix_law_across_segments:
                _validate_covariance_model(segment.model, f"volatility.segments[{idx}].model")
    else:
        raise ConfigValidationError("volatility.mode must be one of constant, piecewise, piecewise_scaled_base")

    if cfg.mp_forward.num_points < 50:
        raise ConfigValidationError("mp_forward.num_points should be at least 50")
    if cfg.mp_forward.grid_max <= cfg.mp_forward.grid_min:
        raise ConfigValidationError("mp_forward.grid_max must be > grid_min")
    if cfg.mp_forward.eta <= 0:
        raise ConfigValidationError("mp_forward.eta must be positive")

    if cfg.mp_inverse.method not in {"fixed_point", "optimization", "stieltjes_based", "moment_based"}:
        raise ConfigValidationError("mp_inverse.method is invalid")
    if cfg.mp_inverse.n_support <= 2:
        raise ConfigValidationError("mp_inverse.n_support must be > 2")
    if cfg.mp_inverse.max_iter <= 0 or cfg.mp_inverse.optimizer_max_iter <= 0:
        raise ConfigValidationError("mp_inverse iteration settings must be positive")
