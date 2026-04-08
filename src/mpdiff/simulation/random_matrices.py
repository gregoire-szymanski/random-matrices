"""Random and deterministic matrix generators for covariance models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpdiff.config.schemas import LowRankFactorConfig


def generate_orthogonal_matrix(d: int, rng: np.random.Generator, method: str = "haar") -> np.ndarray:
    """Generate an orthogonal matrix.

    Parameters
    ----------
    d:
        Matrix dimension.
    rng:
        NumPy random generator.
    method:
        - ``"haar"``: Haar-distributed orthogonal matrix via QR.
        - ``"identity"``: deterministic identity matrix.
    """
    if method == "identity":
        return np.eye(d)
    if method != "haar":
        raise ValueError(f"Unsupported orthogonal method: {method}")

    gaussian = rng.normal(size=(d, d))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q @ np.diag(signs)


def _load_matrix_from_file(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Low-rank factor file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(file_path)
    elif suffix == ".npz":
        archive = np.load(file_path)
        if "matrix" in archive:
            matrix = archive["matrix"]
        else:
            first_key = sorted(archive.files)[0]
            matrix = archive[first_key]
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(file_path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported factor file format: {suffix}")

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape == expected_shape:
        return matrix
    if matrix.shape == expected_shape[::-1]:
        return matrix.T
    raise ValueError(f"Factor matrix shape mismatch: expected {expected_shape}, got {matrix.shape}")


def _normalize_columns(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return matrix / norms


def generate_low_rank_factor(
    d: int,
    rank: int,
    rng: np.random.Generator,
    factor_cfg: LowRankFactorConfig,
) -> np.ndarray:
    """Generate low-rank factor matrix ``B`` with configurable method."""
    if rank <= 0:
        raise ValueError("rank must be positive")

    if factor_cfg.method == "gaussian":
        factor = rng.normal(size=(d, rank))

    elif factor_cfg.method == "identity_block":
        factor = np.zeros((d, rank), dtype=float)
        diag_len = min(d, rank)
        factor[np.arange(diag_len), np.arange(diag_len)] = 1.0

    elif factor_cfg.method == "from_file":
        if factor_cfg.matrix_path is None:
            raise ValueError("factor.matrix_path is required when method='from_file'")
        factor = _load_matrix_from_file(factor_cfg.matrix_path, expected_shape=(d, rank))

    else:
        raise ValueError(f"Unsupported low-rank factor generation method: {factor_cfg.method}")

    if factor_cfg.normalize_columns:
        factor = _normalize_columns(factor)

    return factor_cfg.scale * factor
