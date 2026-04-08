"""Optimization-based MP inverse solver."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.transforms import mp_forward_transform


def _support_range(observed: GridDensity, aspect_ratio: float, settings: MPInverseConfig) -> tuple[float, float]:
    sqrt_c = np.sqrt(aspect_ratio)
    lower_scale = (max(1.0 - sqrt_c, 0.05)) ** 2
    upper_scale = (1.0 + sqrt_c) ** 2

    support_min = settings.support_min if settings.support_min is not None else max(1e-8, observed.grid[0] / upper_scale)
    support_max = settings.support_max if settings.support_max is not None else max(support_min * 1.1, observed.grid[-1] / lower_scale)
    if support_max <= support_min:
        support_max = support_min * 1.5
    return support_min, support_max


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits)
    exp = np.exp(centered)
    return exp / np.sum(exp)


def invert_optimization(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> tuple[DiscreteSpectrum, GridDensity, dict[str, float | int | bool | str]]:
    """Estimate population law by minimizing forward mismatch."""
    support_min, support_max = _support_range(observed, aspect_ratio, inverse_settings)
    support = np.linspace(support_min, support_max, inverse_settings.n_support)

    obs_pdf = observed.density

    def objective(logits: np.ndarray) -> float:
        weights = _softmax(logits)
        candidate = DiscreteSpectrum(atoms=support, weights=weights)
        predicted = mp_forward_transform(
            population=candidate,
            aspect_ratio=aspect_ratio,
            grid=observed.grid,
            eta=inverse_settings.eta,
            tol=inverse_settings.forward_tol,
            max_iter=inverse_settings.forward_max_iter,
            damping=forward_settings.damping,
        )
        misfit = float(np.mean((predicted.density - obs_pdf) ** 2))

        if inverse_settings.regularization > 0:
            smooth_penalty = float(np.mean(np.diff(weights, n=2) ** 2))
            misfit += inverse_settings.regularization * smooth_penalty
        return misfit

    x0 = np.zeros(inverse_settings.n_support)
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": inverse_settings.optimizer_max_iter},
    )

    weights = _softmax(result.x)
    population = DiscreteSpectrum(atoms=support, weights=weights, name="inverse_optimization")
    reconstructed = mp_forward_transform(
        population=population,
        aspect_ratio=aspect_ratio,
        grid=observed.grid,
        eta=inverse_settings.eta,
        tol=inverse_settings.forward_tol,
        max_iter=inverse_settings.forward_max_iter,
        damping=forward_settings.damping,
    )

    diagnostics = {
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "objective": float(result.fun),
        "status": str(result.message),
    }
    return population, reconstructed, diagnostics
