"""Common interfaces for MP inverse methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity


@dataclass(slots=True)
class MethodResult:
    """Standard output for an inverse method."""

    estimated_population: DiscreteSpectrum
    reconstructed_observed: GridDensity
    diagnostics: dict[str, Any]


class MPInverseMethod(Protocol):
    """Protocol implemented by MP inverse algorithms."""

    name: str

    def invert(
        self,
        observed: GridDensity,
        aspect_ratio: float,
        inverse_settings: MPInverseConfig,
        forward_settings: MPForwardConfig,
    ) -> MethodResult:
        """Run inverse estimation and return method result."""
