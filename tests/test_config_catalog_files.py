"""Smoke tests for top-level catalog config files."""

from __future__ import annotations

from pathlib import Path

from mpdiff.config.loader import load_config


def test_all_catalog_configs_load() -> None:
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    cfg_files = sorted(cfg_dir.glob("config_*.yaml"))
    assert len(cfg_files) >= 10

    for path in cfg_files:
        cfg = load_config(path)
        assert cfg.simulation.d > 0
        assert cfg.simulation.n_steps > 0
        assert cfg.mp_forward.num_points >= 50
