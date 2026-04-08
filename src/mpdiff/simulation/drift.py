"""Drift model construction for diffusion simulation."""

from __future__ import annotations

from collections.abc import Callable
import importlib

import numpy as np

from mpdiff.config.schemas import DriftConfig, SimulationConfig


DriftFunction = Callable[[float, np.ndarray], np.ndarray]


def expand_to_dimension(value: float | list[float], d: int) -> np.ndarray:
    """Expand scalar or length-1 vector to dimension ``d``."""
    if isinstance(value, (float, int)):
        return np.full(d, float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return np.full(d, float(arr[0]), dtype=float)
    if arr.size != d:
        raise ValueError(f"Vector-valued input must have length 1 or d={d}, got {arr.size}")
    return arr


def _load_callable(path: str) -> Callable[..., np.ndarray]:
    if ":" not in path:
        raise ValueError("callable_path must use 'module:function' format")
    module_name, fn_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Unable to import callable '{path}'")
    return fn


def build_drift_function(sim_cfg: SimulationConfig) -> DriftFunction:
    """Create a callable drift function ``b(t, x)`` from simulation config."""
    d = sim_cfg.d
    drift_cfg: DriftConfig = sim_cfg.drift_model

    if drift_cfg.kind == "zero":
        return lambda t, x: np.zeros(d, dtype=float)

    if drift_cfg.kind == "constant":
        vector = expand_to_dimension(drift_cfg.vector, d)
        return lambda t, x: vector

    if drift_cfg.kind == "linear_mean_reversion":
        target = expand_to_dimension(drift_cfg.target, d)
        theta = float(drift_cfg.theta)
        return lambda t, x: -theta * (x - target)

    if drift_cfg.kind == "time_sine":
        if drift_cfg.direction:
            direction = expand_to_dimension(drift_cfg.direction, d)
        else:
            direction = np.ones(d, dtype=float)
        amplitude = float(drift_cfg.amplitude)
        frequency = float(drift_cfg.frequency)
        phase = float(drift_cfg.phase)
        two_pi = 2.0 * np.pi
        return lambda t, x: amplitude * np.sin(two_pi * frequency * t + phase) * direction

    if drift_cfg.kind == "callable":
        if drift_cfg.callable_path is None:
            raise ValueError("callable drift requires callable_path")
        fn = _load_callable(drift_cfg.callable_path)
        kwargs = dict(drift_cfg.callable_kwargs)

        def _callable_drift(t: float, x: np.ndarray) -> np.ndarray:
            try:
                raw = fn(t, x, **kwargs)
            except TypeError:
                raw = fn(t, x, kwargs)
            return expand_to_dimension(raw.tolist() if isinstance(raw, np.ndarray) and raw.ndim == 1 else raw, d)

        return _callable_drift

    raise ValueError(f"Unsupported drift kind: {drift_cfg.kind}")
