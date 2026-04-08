"""Smoke tests for documented simulation example configs."""

from __future__ import annotations

from pathlib import Path

from mpdiff.config.loader import load_config


def test_all_simulation_example_configs_load() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "simulation_examples"
    config_files = sorted(config_dir.glob("*.yaml"))
    assert len(config_files) >= 6

    for config_path in config_files:
        cfg = load_config(config_path)
        assert cfg.simulation.d > 0
        assert cfg.simulation.n_steps > 0
        assert cfg.volatility.mode in {"constant", "piecewise", "piecewise_scaled_base"}
