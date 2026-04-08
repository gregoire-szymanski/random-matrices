"""MP inverse dispatch and result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.inversion_methods import (
    invert_fixed_point,
    invert_moment_based,
    invert_optimization,
    invert_stieltjes_based,
)


@dataclass(slots=True)
class InversionResult:
    """Result of a Marcenko-Pastur inverse computation."""

    method: str
    estimated_population: DiscreteSpectrum
    reconstructed_observed: GridDensity
    diagnostics: dict[str, Any]


def invert_mp_density(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> InversionResult:
    """Dispatch MP inverse estimation according to configured method."""
    method = inverse_settings.method
    if method == "fixed_point":
        population, reconstructed, diagnostics = invert_fixed_point(observed, aspect_ratio, inverse_settings, forward_settings)
    elif method == "optimization":
        population, reconstructed, diagnostics = invert_optimization(observed, aspect_ratio, inverse_settings, forward_settings)
    elif method == "stieltjes_based":
        population, reconstructed, diagnostics = invert_stieltjes_based(observed, aspect_ratio, inverse_settings, forward_settings)
    elif method == "moment_based":
        population, reconstructed, diagnostics = invert_moment_based(observed, aspect_ratio, inverse_settings, forward_settings)
    else:
        raise ValueError(f"Unsupported MP inverse method: {method}")

    return InversionResult(
        method=method,
        estimated_population=population,
        reconstructed_observed=reconstructed,
        diagnostics=diagnostics,
    )
