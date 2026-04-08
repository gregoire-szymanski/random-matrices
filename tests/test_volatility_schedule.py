"""Tests for volatility schedule policies and scaling behavior."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import (
    CovarianceModelConfig,
    EigenDistributionConfig,
    PiecewiseSegmentConfig,
    ProjectConfig,
    ScaledBaseVolatilityConfig,
    SimulationConfig,
    VolatilityConfig,
)
from mpdiff.simulation.volatility_segments import build_volatility_schedule


def test_piecewise_scaled_base_common_fixed_reuses_same_base_matrix() -> None:
    base_model = CovarianceModelConfig(
        kind="diag_distribution",
        eigen_distribution=EigenDistributionConfig(kind="uniform", low=0.5, high=1.2),
        sampling_policy="draw_once",
    )
    cfg = ProjectConfig(
        simulation=SimulationConfig(d=25, T=1.0, n_steps=100),
        volatility=VolatilityConfig(
            mode="piecewise_scaled_base",
            scaled_base=ScaledBaseVolatilityConfig(
                base_model=base_model,
                base_matrix_policy="common_fixed",
                share_matrix_law_across_segments=True,
            ),
            segments=[
                PiecewiseSegmentConfig(start=0.0, end=0.4, scalar=0.5),
                PiecewiseSegmentConfig(start=0.4, end=1.0, scalar=2.0),
            ],
        ),
    )

    rng = np.random.default_rng(123)
    schedule = build_volatility_schedule(cfg, rng)

    seg0 = schedule.segments[0]
    seg1 = schedule.segments[1]
    expected_ratio = 2.0 / 0.5
    assert np.allclose(seg1.covariance, expected_ratio * seg0.covariance)
    assert np.allclose(seg1.volatility, np.sqrt(expected_ratio) * seg0.volatility)


def test_piecewise_scaled_base_redraw_per_segment_draws_independent_matrices() -> None:
    base_model = CovarianceModelConfig(
        kind="diag_distribution",
        eigen_distribution=EigenDistributionConfig(kind="uniform", low=0.5, high=1.2),
        sampling_policy="draw_once",
    )
    cfg = ProjectConfig(
        simulation=SimulationConfig(d=25, T=1.0, n_steps=100),
        volatility=VolatilityConfig(
            mode="piecewise_scaled_base",
            scaled_base=ScaledBaseVolatilityConfig(
                base_model=base_model,
                base_matrix_policy="redraw_per_segment",
                share_matrix_law_across_segments=True,
            ),
            segments=[
                PiecewiseSegmentConfig(start=0.0, end=0.4, scalar=1.0),
                PiecewiseSegmentConfig(start=0.4, end=1.0, scalar=1.0),
            ],
        ),
    )

    rng = np.random.default_rng(123)
    schedule = build_volatility_schedule(cfg, rng)

    seg0 = schedule.segments[0]
    seg1 = schedule.segments[1]
    assert not np.allclose(seg0.covariance, seg1.covariance)
