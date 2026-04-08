"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

from mpdiff.config.loader import load_config


def test_load_sample_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation_examples" / "01_constant_isotropic.yaml"
    cfg = load_config(config_path)

    assert cfg.simulation.d == 80
    assert cfg.simulation.drift_model.kind == "zero"
    assert cfg.volatility.mode == "constant"
    assert cfg.volatility.constant_model.kind == "diag_scalar"
    assert cfg.volatility.constant_model.scalar == 1.25
