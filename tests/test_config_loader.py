"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

from mpdiff.config.loader import load_config


def test_load_sample_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "constant_diag_dirac.yaml"
    cfg = load_config(config_path)

    assert cfg.simulation.d == 80
    assert cfg.volatility.mode == "constant"
    assert cfg.volatility.constant_model.kind == "diag_scalar"
    assert cfg.mp_inverse.method == "optimization"
