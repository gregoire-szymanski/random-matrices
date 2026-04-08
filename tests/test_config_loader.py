"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mpdiff.config.loader import load_config


def test_load_sample_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation_examples" / "01_constant_isotropic.yaml"
    cfg = load_config(config_path)

    assert cfg.simulation.d == 80
    assert cfg.simulation.drift_model.kind == "zero"
    assert cfg.volatility.mode == "constant"
    assert cfg.volatility.constant_model.kind == "diag_scalar"
    assert cfg.volatility.constant_model.scalar == 1.25


def test_load_spectral_config_with_population_spectrum() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "spectral_examples" / "03_gamma_population.yaml"
    cfg = load_config(config_path)

    assert cfg.analysis.population_spectrum is not None
    assert cfg.analysis.population_spectrum.source == "parametric"
    assert cfg.analysis.population_spectrum.kind == "gamma"
    assert cfg.mp_inverse.compare_methods == ["optimization", "moment_based"]


def test_load_catalog_config_with_compare_all() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config_diagonal_gamma.yaml"
    cfg = load_config(config_path)

    assert cfg.mp_inverse.compare_all_methods is True
    assert cfg.mp_inverse.compare_methods == []


def test_compare_all_and_compare_methods_conflict_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
simulation:
  d: 10
  T: 1.0
  n_steps: 40
volatility:
  mode: constant
  constant_model:
    kind: diag_scalar
    scalar: 1.0
mp_inverse:
  method: optimization
  compare_all_methods: true
  compare_methods: [optimization, fixed_point]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(config_path)
