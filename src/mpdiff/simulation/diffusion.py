"""Simulation of high-dimensional diffusion trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mpdiff.config.schemas import ProjectConfig, SimulationConfig
from mpdiff.simulation.drift import build_drift_function, expand_to_dimension
from mpdiff.simulation.volatility_segments import VolatilitySchedule, build_volatility_schedule
from mpdiff.utils.random import make_rng


@dataclass(slots=True)
class SimulationResult:
    """Result of a diffusion simulation run."""

    times: np.ndarray
    path: np.ndarray
    increments: np.ndarray
    schedule: VolatilitySchedule
    segment_indices: np.ndarray


def build_time_grid(sim_cfg: SimulationConfig) -> np.ndarray:
    """Build simulation time grid with ``n_steps + 1`` points on ``[0, T]``."""
    if sim_cfg.time_grid is not None:
        return np.asarray(sim_cfg.time_grid, dtype=float)
    return np.linspace(0.0, sim_cfg.T, sim_cfg.n_steps + 1)


def simulate_diffusion(
    sim_cfg: SimulationConfig,
    schedule: VolatilitySchedule,
    rng: np.random.Generator,
    logger: logging.Logger | None = None,
) -> SimulationResult:
    """Simulate diffusion by Euler scheme with piecewise-constant volatility.

    The simulator applies
    ``X_{t_{i+1}} = X_{t_i} + b_i * dt + sigma_i * sqrt(dt) * Z_i``
    where ``Z_i ~ N(0, I_d)`` and ``sigma_i`` is segment-wise constant.
    """
    d = sim_cfg.d
    times = build_time_grid(sim_cfg)
    n_steps = len(times) - 1

    initial_state = expand_to_dimension(sim_cfg.initial_state, d)
    drift_fn = build_drift_function(sim_cfg)

    path = np.zeros((n_steps + 1, d), dtype=float)
    path[0] = initial_state
    increments = np.zeros((n_steps, d), dtype=float)

    segment_indices = schedule.segment_indices_for_times(times[:-1])

    for step in range(n_steps):
        dt = times[step + 1] - times[step]
        if dt <= 0:
            raise ValueError("Time grid must be strictly increasing")

        t_i = float(times[step])
        x_i = path[step]
        b_i = np.asarray(drift_fn(t_i, x_i), dtype=float).reshape(-1)
        if b_i.size == 1:
            b_i = np.full(d, float(b_i[0]), dtype=float)
        elif b_i.size != d:
            raise ValueError(f"Drift function must return length 1 or d={d}, got {b_i.size}")

        seg_idx = int(segment_indices[step])
        sigma_i = schedule.segments[seg_idx].volatility
        z_i = rng.standard_normal(d)

        increment = b_i * dt + sigma_i @ (np.sqrt(dt) * z_i)
        increments[step] = increment
        path[step + 1] = x_i + increment

    if logger is not None:
        logger.info("Simulated diffusion: d=%d, steps=%d, horizon=%.6f", d, n_steps, sim_cfg.T)

    return SimulationResult(
        times=times,
        path=path,
        increments=increments,
        schedule=schedule,
        segment_indices=segment_indices,
    )


def simulate_from_config(cfg: ProjectConfig) -> SimulationResult:
    """Run one full simulation from project config."""
    logger = logging.getLogger("mpdiff.simulation")
    rng = make_rng(cfg.global_settings.seed)
    schedule = build_volatility_schedule(cfg=cfg, rng=rng)
    return simulate_diffusion(sim_cfg=cfg.simulation, schedule=schedule, rng=rng, logger=logger)
