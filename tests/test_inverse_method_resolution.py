"""Tests for inverse-method resolution logic."""

from __future__ import annotations

from mpdiff.config.schemas import MPInverseConfig
from mpdiff.spectral.inverse import available_inverse_methods, resolve_inverse_methods


def test_resolve_inverse_methods_explicit_override() -> None:
    cfg = MPInverseConfig(method="optimization", compare_all_methods=True)
    methods = resolve_inverse_methods(cfg, methods=["moment_based"])
    assert methods == ["moment_based"]


def test_resolve_inverse_methods_compare_all() -> None:
    cfg = MPInverseConfig(method="optimization", compare_all_methods=True)
    methods = resolve_inverse_methods(cfg)
    assert methods == available_inverse_methods()


def test_resolve_inverse_methods_compare_subset() -> None:
    cfg = MPInverseConfig(method="optimization", compare_methods=["fixed_point", "moment_based"])
    methods = resolve_inverse_methods(cfg)
    assert methods == ["fixed_point", "moment_based"]


def test_resolve_inverse_methods_default_primary_method() -> None:
    cfg = MPInverseConfig(method="stieltjes_based")
    methods = resolve_inverse_methods(cfg)
    assert methods == ["stieltjes_based"]
