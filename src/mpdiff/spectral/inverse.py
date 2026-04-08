"""MP inverse dispatch, comparison helpers, and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mpdiff.config.schemas import MPForwardConfig, MPInverseConfig
from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity
from mpdiff.spectral.inversion_methods import build_method_registry
from mpdiff.spectral.inversion_methods.base import MethodResult


@dataclass(slots=True)
class InversionResult:
    """Result of a Marcenko-Pastur inverse computation."""

    method: str
    estimated_population: DiscreteSpectrum
    reconstructed_observed: GridDensity
    diagnostics: dict[str, Any]


def _resolve_method(method_name: str):
    registry = build_method_registry()
    if method_name not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported MP inverse method: {method_name}. Supported: {supported}")
    return registry[method_name]


def _to_inversion_result(method_name: str, method_result: MethodResult) -> InversionResult:
    return InversionResult(
        method=method_name,
        estimated_population=method_result.estimated_population,
        reconstructed_observed=method_result.reconstructed_observed,
        diagnostics=method_result.diagnostics,
    )


def invert_mp_density(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
) -> InversionResult:
    """Dispatch MP inverse estimation according to configured method."""
    method_name = inverse_settings.method
    method = _resolve_method(method_name)
    method_result = method.invert(observed, aspect_ratio, inverse_settings, forward_settings)
    return _to_inversion_result(method_name, method_result)


def compare_inverse_methods(
    observed: GridDensity,
    aspect_ratio: float,
    inverse_settings: MPInverseConfig,
    forward_settings: MPForwardConfig,
    methods: list[str] | None = None,
) -> dict[str, InversionResult]:
    """Run several inverse methods on the same observed density."""
    method_names = resolve_inverse_methods(inverse_settings, methods=methods)

    results: dict[str, InversionResult] = {}
    for method_name in method_names:
        local_settings = MPInverseConfig(
            method=method_name,
            compare_all_methods=False,
            compare_methods=list(inverse_settings.compare_methods),
            n_support=inverse_settings.n_support,
            support_min=inverse_settings.support_min,
            support_max=inverse_settings.support_max,
            eta=inverse_settings.eta,
            tol=inverse_settings.tol,
            max_iter=inverse_settings.max_iter,
            regularization=inverse_settings.regularization,
            optimizer_max_iter=inverse_settings.optimizer_max_iter,
            forward_max_iter=inverse_settings.forward_max_iter,
            forward_tol=inverse_settings.forward_tol,
            fixed_point=inverse_settings.fixed_point,
            optimization=inverse_settings.optimization,
            stieltjes_based=inverse_settings.stieltjes_based,
            moment_based=inverse_settings.moment_based,
        )
        result = invert_mp_density(observed, aspect_ratio, local_settings, forward_settings)
        results[method_name] = result

    return results


def available_inverse_methods() -> list[str]:
    """Return sorted list of registered inverse method names."""
    return sorted(build_method_registry().keys())


def resolve_inverse_methods(
    inverse_settings: MPInverseConfig,
    methods: list[str] | None = None,
) -> list[str]:
    """Resolve method list from explicit input or config comparison switches."""
    if methods is not None:
        return list(methods)
    if inverse_settings.compare_all_methods:
        return available_inverse_methods()
    if inverse_settings.compare_methods:
        return list(inverse_settings.compare_methods)
    return [inverse_settings.method]
