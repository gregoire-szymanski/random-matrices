"""Covariance/volatility matrix builders and eigenvalue samplers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np

from mpdiff.config.schemas import CovarianceModelConfig, DiagonalNoiseConfig, EigenDistributionConfig
from mpdiff.simulation.random_matrices import generate_low_rank_factor, generate_orthogonal_matrix
from mpdiff.utils.linear_algebra import symmetrize


@dataclass(slots=True)
class CovarianceBuildResult:
    """Output of covariance construction."""

    covariance: np.ndarray
    volatility: np.ndarray
    eigenvalues: np.ndarray
    sqrt_method: str
    sqrt_jitter_used: float
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)


def covariance_model_cache_key(model_cfg: CovarianceModelConfig, d: int) -> str:
    """Stable cache key for covariance model realizations."""
    payload = asdict(model_cfg)
    payload["dimension"] = d
    return json.dumps(payload, sort_keys=True)


def sample_eigenvalues(dist_cfg: EigenDistributionConfig, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample non-negative eigenvalues from configured law."""
    kind = dist_cfg.kind
    if kind == "dirac":
        eigs = np.full(size, dist_cfg.value)

    elif kind == "dirac_mixture":
        values = np.asarray(dist_cfg.values, dtype=float)
        if values.size == 0:
            raise ValueError("dirac_mixture requires non-empty values")
        if dist_cfg.weights:
            weights = np.asarray(dist_cfg.weights, dtype=float)
            weights = weights / np.sum(weights)
        else:
            weights = np.full(values.size, 1.0 / values.size)
        eigs = rng.choice(values, size=size, p=weights)

    elif kind == "uniform":
        eigs = rng.uniform(dist_cfg.low, dist_cfg.high, size=size)

    elif kind == "gamma":
        eigs = rng.gamma(shape=dist_cfg.shape, scale=dist_cfg.scale, size=size)

    elif kind == "rescaled_beta":
        eigs = dist_cfg.beta_scale * rng.beta(a=dist_cfg.alpha, b=dist_cfg.beta, size=size) + dist_cfg.beta_shift

    else:
        raise ValueError(f"Unsupported eigenvalue distribution kind: {kind}")

    return np.clip(eigs, 0.0, None)


def _build_diagonal_noise(noise_cfg: DiagonalNoiseConfig, d: int, rng: np.random.Generator) -> np.ndarray:
    if noise_cfg.kind == "scalar_identity":
        return noise_cfg.scalar * np.eye(d)
    diag_eigs = sample_eigenvalues(noise_cfg.distribution, d, rng)
    return np.diag(diag_eigs)


def _enforce_psd(covariance: np.ndarray, base_jitter: float) -> tuple[np.ndarray, np.ndarray]:
    cov = symmetrize(covariance)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.min(eigvals) < 0:
        eigvals = np.clip(eigvals, 0.0, None)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    if base_jitter > 0:
        cov = cov + base_jitter * np.eye(cov.shape[0])

    cov = symmetrize(cov)
    eigvals = np.linalg.eigvalsh(cov)
    return cov, eigvals


def covariance_to_volatility(
    covariance: np.ndarray,
    jitter: float = 1e-10,
    max_cholesky_attempts: int = 6,
) -> tuple[np.ndarray, str, float]:
    """Compute volatility matrix ``sigma`` with ``sigma @ sigma.T = covariance``.

    Tries Cholesky first (with escalating diagonal jitter), then falls back to
    symmetric eigendecomposition-based square root.
    """
    cov = symmetrize(covariance)
    d = cov.shape[0]
    eye = np.eye(d)

    for k in range(max_cholesky_attempts):
        extra_jitter = 0.0 if k == 0 else jitter * (10.0 ** (k - 1))
        try:
            chol = np.linalg.cholesky(cov + extra_jitter * eye)
            return chol, "cholesky", float(extra_jitter)
        except np.linalg.LinAlgError:
            continue

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return sqrt_cov, "eigh", 0.0


def build_covariance_matrix(model_cfg: CovarianceModelConfig, d: int, rng: np.random.Generator) -> CovarianceBuildResult:
    """Construct covariance and volatility matrices from a model config.

    Supported ``kind`` values:
    - ``diag_scalar``
    - ``diag_distribution``
    - ``orthogonal_diag``
    - ``low_rank_plus_diag``
    """
    kind = model_cfg.kind

    if kind == "diag_scalar":
        covariance = model_cfg.scalar * np.eye(d)

    elif kind == "diag_distribution":
        eigs = sample_eigenvalues(model_cfg.eigen_distribution, d, rng)
        covariance = np.diag(eigs)

    elif kind == "orthogonal_diag":
        eigs = sample_eigenvalues(model_cfg.eigen_distribution, d, rng)
        u = generate_orthogonal_matrix(d=d, rng=rng, method=model_cfg.orthogonal.method)
        covariance = u @ np.diag(eigs) @ u.T

    elif kind == "low_rank_plus_diag":
        rank = min(model_cfg.low_rank.rank, d - 1)
        latent_eigs = sample_eigenvalues(model_cfg.low_rank.latent_eigen_distribution, rank, rng)
        factor = generate_low_rank_factor(d=d, rank=rank, rng=rng, factor_cfg=model_cfg.low_rank.factor)
        diagonal_noise = _build_diagonal_noise(model_cfg.low_rank.diagonal_noise, d=d, rng=rng)
        covariance = factor @ np.diag(latent_eigs) @ factor.T + diagonal_noise

    else:
        raise ValueError(f"Unsupported covariance model kind: {kind}")

    covariance, eigvals = _enforce_psd(covariance, base_jitter=model_cfg.jitter)
    volatility, sqrt_method, sqrt_jitter = covariance_to_volatility(covariance, jitter=max(model_cfg.jitter, 1e-14))

    if sqrt_method == "cholesky" and sqrt_jitter > 0:
        covariance = covariance + sqrt_jitter * np.eye(d)
        eigvals = np.linalg.eigvalsh(covariance)

    metadata: dict[str, str | float | int | bool] = {
        "kind": kind,
        "sqrt_method": sqrt_method,
        "sqrt_jitter_used": float(sqrt_jitter),
    }
    return CovarianceBuildResult(
        covariance=covariance,
        volatility=volatility,
        eigenvalues=eigvals,
        sqrt_method=sqrt_method,
        sqrt_jitter_used=sqrt_jitter,
        metadata=metadata,
    )


def build_covariance_and_volatility(model_cfg: CovarianceModelConfig, d: int, rng: np.random.Generator) -> CovarianceBuildResult:
    """Convenience alias for :func:`build_covariance_matrix`."""
    return build_covariance_matrix(model_cfg=model_cfg, d=d, rng=rng)
