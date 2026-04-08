"""Tests for covariance matrix constructors."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import CovarianceModelConfig, EigenDistributionConfig, LowRankConfig, OrthogonalConfig
from mpdiff.simulation.covariance_builders import build_covariance_matrix


def test_orthogonal_diag_covariance_is_psd() -> None:
    rng = np.random.default_rng(0)
    model = CovarianceModelConfig(
        kind="orthogonal_diag",
        eigen_distribution=EigenDistributionConfig(kind="uniform", low=0.3, high=2.0),
        orthogonal=OrthogonalConfig(method="haar"),
    )

    result = build_covariance_matrix(model_cfg=model, d=40, rng=rng)
    eigvals = np.linalg.eigvalsh(result.covariance)

    assert result.covariance.shape == (40, 40)
    assert eigvals.min() > -1e-8


def test_low_rank_plus_diag_covariance_is_psd() -> None:
    rng = np.random.default_rng(1)
    low_rank = LowRankConfig(
        rank=6,
        factor_scale=1.2,
        latent_eigen_distribution=EigenDistributionConfig(kind="gamma", shape=2.0, scale=0.6),
        diag_eigen_distribution=EigenDistributionConfig(kind="dirac", value=0.1),
    )
    model = CovarianceModelConfig(kind="low_rank_plus_diag", low_rank=low_rank)

    result = build_covariance_matrix(model_cfg=model, d=35, rng=rng)
    eigvals = np.linalg.eigvalsh(result.covariance)

    assert result.covariance.shape == (35, 35)
    assert eigvals.min() > -1e-8
