"""Simulation of high-dimensional diffusion trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mpdiff.config.schemas import ProjectConfig, SimulationConfig
from mpdiff.simulation.volatility_segments import VolatilitySchedule, build_volatility_schedule
from mpdiff.utils.linear_algebra import sqrt_psd
from mpdiff.utils.random import make_rng


@dataclass(slots=True)
class SimulationResult:
    """Result of a diffusion simulation run."""

    times: np.ndarray
    path: np.ndarray
    schedule: VolatilitySchedule
    segment_indices: np.ndarray


def _expand_vector(value: float | list[float], d: int) -> np.ndarray:
    if isinstance(value, (float, int)):
        return np.full(d, float(value))
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return np.full(d, float(arr[0]))
    if arr.size != d:
        raise ValueError(f"Vector-valued parameter must have size {d}")
    return arr


def build_time_grid(sim_cfg: SimulationConfig) -> np.ndarray:
    """Build simulation time grid."""
    if sim_cfg.time_grid is not None:
        return np.asarray(sim_cfg.time_grid, dtype=float)
    return np.linspace(0.0, sim_cfg.T, sim_cfg.n_steps + 1)


def simulate_diffusion(
    sim_cfg: SimulationConfig,
    schedule: VolatilitySchedule,
    rng: np.random.Generator,
    logger: logging.Logger | None = None,
) -> SimulationResult:
    """Simulate a diffusion with piecewise-constant covariance.

    The discretization follows Euler increments:

    ``X_{k+1} = X_k + b * dt + sqrt(C_t) * sqrt(dt) * ξ_k``.
    """
    d = sim_cfg.d
    times = build_time_grid(sim_cfg)
    n_steps = len(times) - 1

    drift = _expand_vector(sim_cfg.drift, d)
    x0 = _expand_vector(sim_cfg.initial_state, d)

    path = np.zeros((n_steps + 1, d), dtype=float)
    path[0] = x0

    segment_indices = schedule.segment_indices_for_times(times[:-1])
    unique_segments = np.unique(segment_indices)
    cov_sqrts = {
        int(idx): sqrt_psd(schedule.segments[int(idx)].covariance)
        for idx in unique_segments
    }

    for step in range(n_steps):
        dt = times[step + 1] - times[step]
        if dt <= 0:
            raise ValueError("Time grid must be strictly increasing")
        seg_idx = int(segment_indices[step])
        diffusion = cov_sqrts[seg_idx] @ (np.sqrt(dt) * rng.standard_normal(d))
        path[step + 1] = path[step] + drift * dt + diffusion

    if logger is not None:
        logger.info("Simulated diffusion with d=%d, steps=%d", d, n_steps)

    return SimulationResult(times=times, path=path, schedule=schedule, segment_indices=segment_indices)


def simulate_from_config(cfg: ProjectConfig) -> SimulationResult:
    """Run one full simulation from project config."""
    logger = logging.getLogger("mpdiff.simulation")
    rng = make_rng(cfg.global_settings.seed)
    schedule = build_volatility_schedule(cfg=cfg, rng=rng)
    return simulate_diffusion(sim_cfg=cfg.simulation, schedule=schedule, rng=rng, logger=logger)
