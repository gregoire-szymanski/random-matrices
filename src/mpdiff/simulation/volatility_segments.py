"""Piecewise-constant volatility schedule construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpdiff.config.schemas import ProjectConfig
from mpdiff.simulation.covariance_builders import build_covariance_matrix


@dataclass(slots=True)
class VolatilitySegment:
    """Single time segment with fixed covariance matrix."""

    start: float
    end: float
    covariance: np.ndarray
    label: str = ""


@dataclass(slots=True)
class VolatilitySchedule:
    """Piecewise-constant covariance schedule over a time horizon."""

    segments: list[VolatilitySegment]

    def covariance_at(self, t: float) -> np.ndarray:
        """Return covariance matrix active at time ``t``."""
        for segment in self.segments:
            if segment.start <= t < segment.end:
                return segment.covariance
        return self.segments[-1].covariance

    def segment_indices_for_times(self, times: np.ndarray) -> np.ndarray:
        """Map each left-endpoint time to a segment index."""
        ends = np.asarray([segment.end for segment in self.segments])
        indices = np.searchsorted(ends, times, side="right")
        return np.clip(indices, 0, len(self.segments) - 1)

    def integrated_covariance(self) -> np.ndarray:
        """Time-averaged covariance across the full horizon."""
        total_time = self.segments[-1].end - self.segments[0].start
        weighted = np.zeros_like(self.segments[0].covariance)
        for segment in self.segments:
            weighted += (segment.end - segment.start) * segment.covariance
        return weighted / total_time


def build_volatility_schedule(cfg: ProjectConfig, rng: np.random.Generator) -> VolatilitySchedule:
    """Build a piecewise constant volatility schedule from project config."""
    d = cfg.simulation.d
    horizon = cfg.simulation.T
    vol_cfg = cfg.volatility

    if vol_cfg.mode == "constant":
        cov = build_covariance_matrix(vol_cfg.constant_model, d=d, rng=rng).covariance
        return VolatilitySchedule(segments=[VolatilitySegment(start=0.0, end=horizon, covariance=cov, label="constant")])

    if vol_cfg.mode == "piecewise":
        segments: list[VolatilitySegment] = []
        for idx, segment_cfg in enumerate(vol_cfg.segments):
            if segment_cfg.model is None:
                raise ValueError("piecewise mode requires each segment to define model")
            base_cov = build_covariance_matrix(segment_cfg.model, d=d, rng=rng).covariance
            segments.append(
                VolatilitySegment(
                    start=segment_cfg.start,
                    end=segment_cfg.end,
                    covariance=segment_cfg.scalar * base_cov,
                    label=f"segment_{idx}",
                )
            )
        return VolatilitySchedule(segments=segments)

    if vol_cfg.mode == "piecewise_scaled_base":
        segments: list[VolatilitySegment] = []
        shared_cov: np.ndarray | None = None
        if vol_cfg.scaled_base.base_matrix_policy == "fixed_once":
            shared_cov = build_covariance_matrix(vol_cfg.scaled_base.base_model, d=d, rng=rng).covariance

        for idx, segment_cfg in enumerate(vol_cfg.segments):
            if vol_cfg.scaled_base.share_matrix_law_across_segments:
                model_cfg = vol_cfg.scaled_base.base_model
            else:
                model_cfg = segment_cfg.model or vol_cfg.scaled_base.base_model

            if shared_cov is None:
                base_cov = build_covariance_matrix(model_cfg, d=d, rng=rng).covariance
            else:
                base_cov = shared_cov

            segments.append(
                VolatilitySegment(
                    start=segment_cfg.start,
                    end=segment_cfg.end,
                    covariance=segment_cfg.scalar * base_cov,
                    label=f"scaled_segment_{idx}",
                )
            )
        return VolatilitySchedule(segments=segments)

    raise ValueError(f"Unsupported volatility mode: {vol_cfg.mode}")
