"""Stieltjes-transform-based MP inverse estimator."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.transforms import mp_forward_transform

from .base import MethodResult


class StieltjesBasedInverseMethod:
    """Inverse method using nonlinear shrinkage of sample eigenvalue quantiles."""

    name = "stieltjes_based"

    def invert(
        self,
        observed: GridDensity,
        aspect_ratio: float,
        inverse_settings: MPInverseConfig,
        forward_settings: MPForwardConfig,
    ) -> MethodResult:
        q_min = inverse_settings.stieltjes_based.quantile_min
        q_max = inverse_settings.stieltjes_based.quantile_max
        probs = np.linspace(q_min, q_max, inverse_settings.n_support)
        sample_atoms = observed.quantiles(probs)

        estimated_population_atoms = np.empty_like(sample_atoms)
        denoms = np.empty_like(sample_atoms)

        for idx, sample_lambda in enumerate(sample_atoms):
            z = complex(float(sample_lambda), inverse_settings.eta)
            m_val = observed.stieltjes(z)
            denom = abs(1.0 - aspect_ratio - aspect_ratio * sample_lambda * m_val) ** 2
            denoms[idx] = max(denom, 1e-10)
            estimated_population_atoms[idx] = max(sample_lambda / denoms[idx], 1e-10)

        estimated_population_atoms.sort()
        weights = np.full(inverse_settings.n_support, 1.0 / inverse_settings.n_support)

        population = DiscreteSpectrum(
            atoms=estimated_population_atoms,
            weights=weights,
            name="inverse_stieltjes_based",
        )
        reconstructed = mp_forward_transform(
            population=population,
            aspect_ratio=aspect_ratio,
            grid=observed.grid,
            eta=inverse_settings.eta,
            tol=inverse_settings.forward_tol,
            max_iter=inverse_settings.forward_max_iter,
            damping=forward_settings.damping,
            use_newton_fallback=True,
        )

        diagnostics = {
            "iterations": int(inverse_settings.n_support),
            "converged": True,
            "mean_denom": float(np.mean(denoms)),
            "min_denom": float(np.min(denoms)),
            "max_denom": float(np.max(denoms)),
        }
        return MethodResult(
            estimated_population=population,
            reconstructed_observed=reconstructed,
            diagnostics=diagnostics,
        )


def invert_stieltjes_based(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> tuple[DiscreteSpectrum, GridDensity, dict[str, float | int | bool]]:
    """Backward-compatible functional wrapper for stieltjes-based inverse."""
    result = StieltjesBasedInverseMethod().invert(observed, aspect_ratio, inverse_settings, forward_settings)
    return result.estimated_population, result.reconstructed_observed, result.diagnostics
