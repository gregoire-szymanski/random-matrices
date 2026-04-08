"""Covariance matrix model builders and eigenvalue samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpdiff.config.schemas import CovarianceModelConfig, EigenDistributionConfig
from mpdiff.simulation.random_matrices import generate_low_rank_factor, generate_orthogonal_matrix
from mpdiff.utils.linear_algebra import symmetrize


@dataclass(slots=True)
class CovarianceBuildResult:
    """Output of covariance construction."""

    covariance: np.ndarray
    eigenvalues: np.ndarray


def sample_eigenvalues(dist_cfg: EigenDistributionConfig, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample positive eigenvalues from configured law."""
    kind = dist_cfg.kind
    if kind == "dirac":
        eigs = np.full(size, dist_cfg.value)
    elif kind == "dirac_mixture":
        if not dist_cfg.values:
            raise ValueError("dirac_mixture requires non-empty values")
        weights = np.asarray(dist_cfg.weights, dtype=float)
        if weights.size == 0:
            weights = np.ones(len(dist_cfg.values), dtype=float) / len(dist_cfg.values)
        else:
            weights = weights / weights.sum()
        eigs = rng.choice(np.asarray(dist_cfg.values, dtype=float), size=size, p=weights)
    elif kind == "uniform":
        eigs = rng.uniform(dist_cfg.low, dist_cfg.high, size=size)
    elif kind == "gamma":
        eigs = rng.gamma(shape=dist_cfg.shape, scale=dist_cfg.scale, size=size)
    elif kind == "rescaled_beta":
        eigs = dist_cfg.a * rng.beta(a=dist_cfg.alpha, b=dist_cfg.beta, size=size) + dist_cfg.b
    else:
        raise ValueError(f"Unsupported eigenvalue distribution kind: {kind}")
    return np.clip(eigs, 0.0, None)


def build_covariance_matrix(model_cfg: CovarianceModelConfig, d: int, rng: np.random.Generator) -> CovarianceBuildResult:
    """Construct covariance matrix from configuration.

    Supported model kinds
    ---------------------
    - ``diag_scalar``
    - ``diag_distribution``
    - ``orthogonal_diag``
    - ``low_rank_plus_diag``
    """
    kind = model_cfg.kind

    if kind == "diag_scalar":
        eigs = np.full(d, model_cfg.scalar)
        cov = np.diag(eigs)

    elif kind == "diag_distribution":
        eigs = sample_eigenvalues(model_cfg.eigen_distribution, d, rng)
        cov = np.diag(eigs)

    elif kind == "orthogonal_diag":
        eigs = sample_eigenvalues(model_cfg.eigen_distribution, d, rng)
        u = generate_orthogonal_matrix(d, rng=rng, method=model_cfg.orthogonal.method)
        cov = u @ np.diag(eigs) @ u.T

    elif kind == "low_rank_plus_diag":
        rank = min(model_cfg.low_rank.rank, d)
        latent_eigs = sample_eigenvalues(model_cfg.low_rank.latent_eigen_distribution, rank, rng)
        noise_eigs = sample_eigenvalues(model_cfg.low_rank.diag_eigen_distribution, d, rng)
        factor = generate_low_rank_factor(d, rank, rng=rng, scale=model_cfg.low_rank.factor_scale)
        cov = factor @ np.diag(latent_eigs) @ factor.T + np.diag(noise_eigs)
        eigs = np.linalg.eigvalsh(symmetrize(cov))

    else:
        raise ValueError(f"Unsupported covariance model kind: {kind}")

    cov = symmetrize(cov) + model_cfg.jitter * np.eye(d)
    eigs = np.linalg.eigvalsh(cov)
    return CovarianceBuildResult(covariance=cov, eigenvalues=eigs)
