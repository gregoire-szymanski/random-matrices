"""Implementations and registry of MP inverse estimators."""

from __future__ import annotations

from .base import MPInverseMethod, MethodResult
from .fixed_point import FixedPointInverseMethod, invert_fixed_point
from .moment_based import MomentBasedInverseMethod, invert_moment_based
from .optimization import OptimizationInverseMethod, invert_optimization
from .stieltjes_based import StieltjesBasedInverseMethod, invert_stieltjes_based


def build_method_registry() -> dict[str, MPInverseMethod]:
    """Create a fresh registry of available inverse methods."""
    return {
        FixedPointInverseMethod.name: FixedPointInverseMethod(),
        OptimizationInverseMethod.name: OptimizationInverseMethod(),
        StieltjesBasedInverseMethod.name: StieltjesBasedInverseMethod(),
        MomentBasedInverseMethod.name: MomentBasedInverseMethod(),
    }


__all__ = [
    "MethodResult",
    "MPInverseMethod",
    "FixedPointInverseMethod",
    "OptimizationInverseMethod",
    "StieltjesBasedInverseMethod",
    "MomentBasedInverseMethod",
    "invert_fixed_point",
    "invert_optimization",
    "invert_stieltjes_based",
    "invert_moment_based",
    "build_method_registry",
]
