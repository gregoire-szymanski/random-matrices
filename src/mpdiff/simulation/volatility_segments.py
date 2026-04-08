"""Piecewise-constant volatility schedule construction and caching."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpdiff.config.schemas import CovarianceModelConfig, PiecewiseSegmentConfig, ProjectConfig
from mpdiff.simulation.covariance_builders import (
    CovarianceBuildResult,
    build_covariance_matrix,
    covariance_model_cache_key,
)


@dataclass(slots=True)
class VolatilitySegment:
    """Single time segment with fixed covariance and volatility matrices."""

    start: float
    end: float
    covariance: np.ndarray
    volatility: np.ndarray
    eigenvalues: np.ndarray
    label: str = ""
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)


@dataclass(slots=True)
class VolatilitySchedule:
    """Piecewise-constant volatility schedule over a time horizon."""

    segments: list[VolatilitySegment]

    def covariance_at(self, t: float) -> np.ndarray:
        """Return covariance matrix active at time ``t``."""
        return self.segments[self.segment_index_at(t)].covariance

    def volatility_at(self, t: float) -> np.ndarray:
        """Return volatility matrix active at time ``t``."""
        return self.segments[self.segment_index_at(t)].volatility

    def segment_index_at(self, t: float) -> int:
        """Return segment index active at time ``t``."""
        for idx, segment in enumerate(self.segments):
            if segment.start <= t < segment.end:
                return idx
        return len(self.segments) - 1

    def segment_indices_for_times(self, times: np.ndarray) -> np.ndarray:
        """Map each left-endpoint time to its segment index."""
        ends = np.asarray([segment.end for segment in self.segments])
        indices = np.searchsorted(ends, times, side="right")
        return np.clip(indices, 0, len(self.segments) - 1)

    def integrated_covariance(self) -> np.ndarray:
        """Time-averaged covariance matrix across the full horizon."""
        total_time = self.segments[-1].end - self.segments[0].start
        weighted = np.zeros_like(self.segments[0].covariance)
        for segment in self.segments:
            weighted += (segment.end - segment.start) * segment.covariance
        return weighted / total_time

    def integrated_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the integrated covariance matrix."""
        return np.linalg.eigvalsh(self.integrated_covariance())


def _resolve_model_realization(
    model_cfg: CovarianceModelConfig,
    d: int,
    rng: np.random.Generator,
    cache: dict[str, CovarianceBuildResult],
    force_redraw: bool = False,
    force_cache: bool = False,
) -> CovarianceBuildResult:
    use_cache = force_cache or (model_cfg.sampling_policy == "draw_once" and not force_redraw)
    key = covariance_model_cache_key(model_cfg, d)

    if use_cache and key in cache:
        return cache[key]

    result = build_covariance_matrix(model_cfg=model_cfg, d=d, rng=rng)
    if use_cache:
        cache[key] = result
    return result


def _scale_segment(
    segment_cfg: PiecewiseSegmentConfig,
    base: CovarianceBuildResult,
    default_label: str,
) -> VolatilitySegment:
    scalar = float(segment_cfg.scalar)
    if scalar <= 0:
        raise ValueError("segment scalar must be positive")

    label = segment_cfg.name or default_label
    covariance = scalar * base.covariance
    volatility = np.sqrt(scalar) * base.volatility
    eigenvalues = scalar * base.eigenvalues

    metadata = dict(base.metadata)
    metadata["segment_scalar"] = scalar
    return VolatilitySegment(
        start=segment_cfg.start,
        end=segment_cfg.end,
        covariance=covariance,
        volatility=volatility,
        eigenvalues=eigenvalues,
        label=label,
        metadata=metadata,
    )


def _resolve_scaled_base_model(cfg: ProjectConfig, segment_cfg: PiecewiseSegmentConfig) -> CovarianceModelConfig:
    if cfg.volatility.scaled_base.share_matrix_law_across_segments:
        return cfg.volatility.scaled_base.base_model
    return segment_cfg.model or cfg.volatility.scaled_base.base_model


def build_volatility_schedule(cfg: ProjectConfig, rng: np.random.Generator) -> VolatilitySchedule:
    """Build piecewise-constant volatility schedule from project config."""
    d = cfg.simulation.d
    horizon = cfg.simulation.T
    vol_cfg = cfg.volatility
    cache: dict[str, CovarianceBuildResult] = {}

    if vol_cfg.mode == "constant":
        base = _resolve_model_realization(vol_cfg.constant_model, d=d, rng=rng, cache=cache)
        segment_cfg = PiecewiseSegmentConfig(start=0.0, end=horizon, scalar=1.0, name="constant")
        return VolatilitySchedule(segments=[_scale_segment(segment_cfg, base=base, default_label="constant")])

    if vol_cfg.mode == "piecewise":
        segments: list[VolatilitySegment] = []
        for idx, segment_cfg in enumerate(vol_cfg.segments):
            if segment_cfg.model is None:
                raise ValueError(f"volatility.segments[{idx}].model is required in piecewise mode")

            force_redraw = segment_cfg.model.sampling_policy == "redraw_per_segment"
            base = _resolve_model_realization(segment_cfg.model, d=d, rng=rng, cache=cache, force_redraw=force_redraw)
            segments.append(_scale_segment(segment_cfg, base=base, default_label=f"segment_{idx}"))
        return VolatilitySchedule(segments=segments)

    if vol_cfg.mode == "piecewise_scaled_base":
        segments: list[VolatilitySegment] = []
        policy = vol_cfg.scaled_base.base_matrix_policy
        for idx, segment_cfg in enumerate(vol_cfg.segments):
            model_cfg = _resolve_scaled_base_model(cfg, segment_cfg)
            force_redraw = policy == "redraw_per_segment"
            force_cache = policy == "common_fixed"
            base = _resolve_model_realization(
                model_cfg,
                d=d,
                rng=rng,
                cache=cache,
                force_redraw=force_redraw,
                force_cache=force_cache,
            )
            segments.append(_scale_segment(segment_cfg, base=base, default_label=f"scaled_segment_{idx}"))
        return VolatilitySchedule(segments=segments)

    raise ValueError(f"Unsupported volatility mode: {vol_cfg.mode}")
