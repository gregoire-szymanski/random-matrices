"""Configuration validation utilities."""

from __future__ import annotations

from typing import Iterable

from .schemas import CovarianceModelConfig, DriftConfig, EigenDistributionConfig, ProjectConfig


class ConfigValidationError(ValueError):
    """Raised when a project configuration is invalid."""


def _validate_distribution(cfg: EigenDistributionConfig, prefix: str) -> None:
    allowed = {"dirac", "dirac_mixture", "uniform", "gamma", "rescaled_beta"}
    if cfg.kind not in allowed:
        raise ConfigValidationError(f"{prefix}: unsupported distribution kind '{cfg.kind}'")

    if cfg.kind == "dirac":
        if cfg.value < 0:
            raise ConfigValidationError(f"{prefix}: dirac value must be non-negative")
        return

    if cfg.kind == "dirac_mixture":
        if not cfg.values:
            raise ConfigValidationError(f"{prefix}: dirac_mixture requires non-empty values")
        if any(v < 0 for v in cfg.values):
            raise ConfigValidationError(f"{prefix}: dirac_mixture values must be non-negative")
        if cfg.weights and len(cfg.weights) != len(cfg.values):
            raise ConfigValidationError(f"{prefix}: weights length must match values length")
        if cfg.weights and sum(cfg.weights) <= 0:
            raise ConfigValidationError(f"{prefix}: weights must have positive sum")
        return

    if cfg.kind == "uniform":
        if cfg.low < 0 or cfg.high <= cfg.low:
            raise ConfigValidationError(f"{prefix}: uniform requires 0 <= low < high")
        return

    if cfg.kind == "gamma":
        if cfg.shape <= 0 or cfg.scale <= 0:
            raise ConfigValidationError(f"{prefix}: gamma requires shape > 0 and scale > 0")
        return

    if cfg.kind == "rescaled_beta":
        if cfg.alpha <= 0 or cfg.beta <= 0:
            raise ConfigValidationError(f"{prefix}: rescaled_beta requires alpha > 0 and beta > 0")
        if cfg.beta_scale < 0:
            raise ConfigValidationError(f"{prefix}: beta_scale must be non-negative")
        if cfg.beta_shift < 0:
            raise ConfigValidationError(f"{prefix}: beta_shift must be non-negative")


def _validate_drift(cfg: DriftConfig, d: int) -> None:
    allowed = {"zero", "constant", "linear_mean_reversion", "time_sine", "callable"}
    if cfg.kind not in allowed:
        raise ConfigValidationError(f"simulation.drift_model.kind unsupported: {cfg.kind}")

    if cfg.kind == "zero":
        return

    if cfg.kind == "constant":
        if not cfg.vector:
            raise ConfigValidationError("simulation.drift_model.vector is required for constant drift")
        if len(cfg.vector) not in {1, d}:
            raise ConfigValidationError("simulation.drift_model.vector length must be 1 or d")
        return

    if cfg.kind == "linear_mean_reversion":
        if cfg.theta < 0:
            raise ConfigValidationError("simulation.drift_model.theta must be non-negative")
        if isinstance(cfg.target, list) and len(cfg.target) not in {1, d}:
            raise ConfigValidationError("simulation.drift_model.target length must be 1 or d")
        return

    if cfg.kind == "time_sine":
        if cfg.frequency <= 0:
            raise ConfigValidationError("simulation.drift_model.frequency must be positive")
        if cfg.direction and len(cfg.direction) not in {1, d}:
            raise ConfigValidationError("simulation.drift_model.direction length must be 1 or d")
        return

    if cfg.kind == "callable":
        if not cfg.callable_path:
            raise ConfigValidationError("simulation.drift_model.callable_path is required for callable drift")


def _validate_covariance_model(cfg: CovarianceModelConfig, prefix: str, d: int) -> None:
    allowed = {"diag_scalar", "diag_distribution", "orthogonal_diag", "low_rank_plus_diag"}
    if cfg.kind not in allowed:
        raise ConfigValidationError(f"{prefix}: unsupported covariance kind '{cfg.kind}'")
    if cfg.jitter < 0:
        raise ConfigValidationError(f"{prefix}: jitter must be non-negative")
    if cfg.sampling_policy not in {"draw_once", "redraw_per_segment"}:
        raise ConfigValidationError(f"{prefix}: sampling_policy must be draw_once or redraw_per_segment")

    if cfg.kind == "diag_scalar":
        if cfg.scalar < 0:
            raise ConfigValidationError(f"{prefix}: scalar must be non-negative")
        return

    if cfg.kind in {"diag_distribution", "orthogonal_diag"}:
        _validate_distribution(cfg.eigen_distribution, f"{prefix}.eigen_distribution")
        if cfg.kind == "orthogonal_diag" and cfg.orthogonal.method not in {"haar", "identity"}:
            raise ConfigValidationError(f"{prefix}.orthogonal.method must be haar or identity")
        return

    # low_rank_plus_diag
    if cfg.low_rank.rank <= 0:
        raise ConfigValidationError(f"{prefix}.low_rank.rank must be positive")
    if cfg.low_rank.rank >= d:
        raise ConfigValidationError(f"{prefix}.low_rank.rank must be strictly smaller than d")

    _validate_distribution(cfg.low_rank.latent_eigen_distribution, f"{prefix}.low_rank.latent_eigen_distribution")

    factor = cfg.low_rank.factor
    if factor.method not in {"gaussian", "identity_block", "from_file"}:
        raise ConfigValidationError(f"{prefix}.low_rank.factor.method must be gaussian, identity_block, or from_file")
    if factor.scale <= 0:
        raise ConfigValidationError(f"{prefix}.low_rank.factor.scale must be positive")
    if factor.method == "from_file" and not factor.matrix_path:
        raise ConfigValidationError(f"{prefix}.low_rank.factor.matrix_path is required for method=from_file")

    noise = cfg.low_rank.diagonal_noise
    if noise.kind not in {"scalar_identity", "distribution"}:
        raise ConfigValidationError(f"{prefix}.low_rank.diagonal_noise.kind must be scalar_identity or distribution")
    if noise.kind == "scalar_identity" and noise.scalar < 0:
        raise ConfigValidationError(f"{prefix}.low_rank.diagonal_noise.scalar must be non-negative")
    if noise.kind == "distribution":
        _validate_distribution(noise.distribution, f"{prefix}.low_rank.diagonal_noise.distribution")


def _validate_segments_cover_horizon(segments: Iterable, horizon: float) -> None:
    segments_list = list(segments)
    if not segments_list:
        raise ConfigValidationError("volatility.segments must not be empty")

    prev_end = 0.0
    for idx, segment in enumerate(segments_list):
        if segment.start >= segment.end:
            raise ConfigValidationError(f"volatility.segments[{idx}]: start must be < end")
        if abs(segment.start - prev_end) > 1e-10:
            raise ConfigValidationError("volatility.segments must be contiguous and ordered")
        prev_end = segment.end

    if abs(prev_end - horizon) > 1e-10:
        raise ConfigValidationError("volatility.segments must end exactly at simulation.T")


def validate_config(cfg: ProjectConfig) -> None:
    """Validate a project configuration in-place.

    Raises
    ------
    ConfigValidationError
        If one or more consistency checks fail.
    """
    d = cfg.simulation.d
    if d <= 0:
        raise ConfigValidationError("simulation.d must be positive")
    if cfg.simulation.n_steps <= 0:
        raise ConfigValidationError("simulation.n_steps must be positive")
    if cfg.simulation.T <= 0:
        raise ConfigValidationError("simulation.T must be positive")

    if isinstance(cfg.simulation.initial_state, list) and len(cfg.simulation.initial_state) not in {1, d}:
        raise ConfigValidationError("simulation.initial_state list length must be 1 or d")

    if cfg.simulation.time_grid is not None:
        if len(cfg.simulation.time_grid) != cfg.simulation.n_steps + 1:
            raise ConfigValidationError("simulation.time_grid must have n_steps + 1 entries")
        if abs(cfg.simulation.time_grid[0]) > 1e-12:
            raise ConfigValidationError("simulation.time_grid must start at 0")
        if abs(cfg.simulation.time_grid[-1] - cfg.simulation.T) > 1e-12:
            raise ConfigValidationError("simulation.time_grid must end at T")
        if any(b <= a for a, b in zip(cfg.simulation.time_grid[:-1], cfg.simulation.time_grid[1:])):
            raise ConfigValidationError("simulation.time_grid must be strictly increasing")

    _validate_drift(cfg.simulation.drift_model, d=d)

    if cfg.volatility.mode == "constant":
        _validate_covariance_model(cfg.volatility.constant_model, "volatility.constant_model", d=d)

    elif cfg.volatility.mode == "piecewise":
        _validate_segments_cover_horizon(cfg.volatility.segments, cfg.simulation.T)
        for idx, segment in enumerate(cfg.volatility.segments):
            if segment.scalar <= 0:
                raise ConfigValidationError(f"volatility.segments[{idx}].scalar must be positive")
            if segment.model is None:
                raise ConfigValidationError(f"volatility.segments[{idx}].model is required in piecewise mode")
            _validate_covariance_model(segment.model, f"volatility.segments[{idx}].model", d=d)

    elif cfg.volatility.mode == "piecewise_scaled_base":
        _validate_segments_cover_horizon(cfg.volatility.segments, cfg.simulation.T)
        _validate_covariance_model(cfg.volatility.scaled_base.base_model, "volatility.scaled_base.base_model", d=d)

        policy = cfg.volatility.scaled_base.base_matrix_policy
        if policy not in {"common_fixed", "redraw_per_segment"}:
            raise ConfigValidationError(
                "volatility.scaled_base.base_matrix_policy must be common_fixed or redraw_per_segment"
            )

        for idx, segment in enumerate(cfg.volatility.segments):
            if segment.scalar <= 0:
                raise ConfigValidationError(f"volatility.segments[{idx}].scalar must be positive")
            if segment.model is not None and not cfg.volatility.scaled_base.share_matrix_law_across_segments:
                _validate_covariance_model(segment.model, f"volatility.segments[{idx}].model", d=d)

    else:
        raise ConfigValidationError("volatility.mode must be one of constant, piecewise, piecewise_scaled_base")

    if cfg.plotting.max_path_dims <= 0:
        raise ConfigValidationError("plotting.max_path_dims must be positive")
    if cfg.plotting.eigen_hist_bins <= 0:
        raise ConfigValidationError("plotting.eigen_hist_bins must be positive")

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
