"""Tests for diffusion simulation."""

from __future__ import annotations

import numpy as np

from mpdiff.config.schemas import (
    CovarianceModelConfig,
    GlobalConfig,
    ProjectConfig,
    SimulationConfig,
    VolatilityConfig,
)
from mpdiff.simulation.diffusion import simulate_from_config


def test_simulation_reproducibility_with_seed() -> None:
    cfg = ProjectConfig(
        global_settings=GlobalConfig(seed=42),
        simulation=SimulationConfig(d=6, T=1.0, n_steps=120, drift=0.0, initial_state=0.0),
        volatility=VolatilityConfig(
            mode="constant",
            constant_model=CovarianceModelConfig(kind="diag_scalar", scalar=1.0),
        ),
    )

    result_1 = simulate_from_config(cfg)
    result_2 = simulate_from_config(cfg)

    assert result_1.path.shape == (121, 6)
    assert np.allclose(result_1.path, result_2.path)
