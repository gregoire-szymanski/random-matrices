"""Spectral law and density objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DiscreteSpectrum:
    """Discrete spectral law represented by atoms and weights."""

    atoms: np.ndarray
    weights: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        atoms = np.asarray(self.atoms, dtype=float)
        if atoms.ndim != 1:
            raise ValueError("atoms must be a 1D array")
        if atoms.size == 0:
            raise ValueError("atoms must be non-empty")

        if self.weights is None:
            weights = np.full(atoms.size, 1.0 / atoms.size)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != atoms.shape:
                raise ValueError("weights must have same shape as atoms")
            total = weights.sum()
            if total <= 0:
                raise ValueError("weights must have strictly positive sum")
            weights = weights / total

        order = np.argsort(atoms)
        self.atoms = atoms[order]
        self.weights = weights[order]

    def mean(self) -> float:
        """First moment of the discrete law."""
        return float(np.dot(self.atoms, self.weights))

    def moment(self, order: int) -> float:
        """Moment of specified non-negative order."""
        if order < 0:
            raise ValueError("order must be non-negative")
        return float(np.dot(np.power(self.atoms, order), self.weights))


@dataclass(slots=True)
class GridDensity:
    """Density sampled on a numerical grid."""

    grid: np.ndarray
    density: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=float)
        density = np.asarray(self.density, dtype=float)
        if grid.ndim != 1 or density.ndim != 1:
            raise ValueError("grid and density must be 1D arrays")
        if grid.size != density.size:
            raise ValueError("grid and density must have the same length")
        if grid.size < 2:
            raise ValueError("grid must have at least 2 points")

        order = np.argsort(grid)
        grid = grid[order]
        density = np.clip(density[order], 0.0, None)

        area = np.trapz(density, grid)
        if area > 0:
            density = density / area

        self.grid = grid
        self.density = density

    def cdf(self) -> np.ndarray:
        """Numerical cumulative distribution function on ``grid``."""
        increments = 0.5 * (self.density[1:] + self.density[:-1]) * np.diff(self.grid)
        cdf_values = np.concatenate([[0.0], np.cumsum(increments)])
        if cdf_values[-1] > 0:
            cdf_values /= cdf_values[-1]
        return cdf_values

    def moment(self, order: int) -> float:
        """Numerical moment from the grid density."""
        if order < 0:
            raise ValueError("order must be non-negative")
        return float(np.trapz(np.power(self.grid, order) * self.density, self.grid))

    def quantiles(self, probs: np.ndarray) -> np.ndarray:
        """Quantiles via linear interpolation on the numerical CDF."""
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        return np.interp(probs, self.cdf(), self.grid)
