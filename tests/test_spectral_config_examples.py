"""Tests for spectral example configs and population-spectrum parsing."""

from __future__ import annotations

from pathlib import Path

from mpdiff.config.loader import load_config
from mpdiff.experiments.common import build_population_spectrum
from mpdiff.utils.random import make_rng


def test_load_spectral_example_configs_and_build_population_spectra() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs" / "spectral_examples"
    yaml_paths = sorted(cfg_dir.glob("*.yaml"))
    assert yaml_paths, "no spectral example config files found"

    for cfg_path in yaml_paths:
        cfg = load_config(cfg_path)
        population = build_population_spectrum(cfg, make_rng(cfg.global_settings.seed))
        assert population.atoms.size > 0
        assert population.weights.size == population.atoms.size

