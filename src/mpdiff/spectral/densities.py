"""Spectral law objects for population and observed distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist


class PopulationLaw(Protocol):
    """Protocol for population spectral laws used by MP forward map."""

    name: str

    def to_discrete(self, n_atoms: int = 400) -> "DiscreteSpectrum":
        """Convert the law to a discrete atomic approximation."""


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return weights / total


@dataclass(slots=True)
class DiscreteSpectrum:
    """Discrete spectral law represented by atoms and probability weights."""

    atoms: np.ndarray
    weights: np.ndarray | None = None
    name: str = "discrete"

    def __post_init__(self) -> None:
        atoms = np.asarray(self.atoms, dtype=float).reshape(-1)
        if atoms.size == 0:
            raise ValueError("atoms must be non-empty")

        if self.weights is None:
            weights = np.full(atoms.size, 1.0 / atoms.size)
        else:
            weights = np.asarray(self.weights, dtype=float).reshape(-1)
            if weights.shape != atoms.shape:
                raise ValueError("weights must have same shape as atoms")
            weights = _normalize_weights(weights)

        order = np.argsort(atoms)
        self.atoms = atoms[order]
        self.weights = weights[order]

    def to_discrete(self, n_atoms: int = 400) -> "DiscreteSpectrum":
        """Return itself (compatibility with :class:`PopulationLaw`)."""
        return self

    def mean(self) -> float:
        """First moment of the law."""
        return float(np.dot(self.atoms, self.weights))

    def moment(self, order: int) -> float:
        """Moment of non-negative integer order."""
        if order < 0:
            raise ValueError("order must be non-negative")
        return float(np.dot(np.power(self.atoms, order), self.weights))

    def stieltjes(self, z: complex | np.ndarray) -> complex | np.ndarray:
        """Evaluate Stieltjes transform ``m(z)=∫(t-z)^(-1)dH(t)``."""
        z_arr = np.asarray(z, dtype=np.complex128)
        values = np.sum(self.weights[None, :] / (self.atoms[None, :] - z_arr[..., None]), axis=-1)
        if np.isscalar(z):
            return complex(values.item())
        return values

    def quantiles(self, probs: np.ndarray) -> np.ndarray:
        """Quantiles by interpolation of weighted empirical CDF."""
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        return np.interp(probs, cdf, self.atoms)

    def to_grid_density(self, grid: np.ndarray, bandwidth: float | None = None) -> "GridDensity":
        """Smooth discrete law on a grid with Gaussian kernels."""
        x = np.asarray(grid, dtype=float)
        span = max(float(np.max(x) - np.min(x)), 1e-6)
        bw = bandwidth if bandwidth is not None else 0.02 * span
        bw = max(float(bw), 1e-6)

        z = (x[:, None] - self.atoms[None, :]) / bw
        kernel = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * bw)
        density = kernel @ self.weights
        return GridDensity(grid=x, density=density, name=f"{self.name}_smoothed")


@dataclass(slots=True)
class GridDensity:
    """Density sampled on a numerical grid."""

    grid: np.ndarray
    density: np.ndarray
    name: str = "grid_density"

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=float).reshape(-1)
        density = np.asarray(self.density, dtype=float).reshape(-1)
        if grid.size != density.size:
            raise ValueError("grid and density must have same length")
        if grid.size < 2:
            raise ValueError("grid must contain at least 2 points")

        order = np.argsort(grid)
        grid = grid[order]
        density = np.clip(density[order], 0.0, None)

        area = float(np.trapz(density, grid))
        if area > 0:
            density = density / area

        self.grid = grid
        self.density = density

    def cdf(self) -> np.ndarray:
        """Cumulative distribution on the grid."""
        dx = np.diff(self.grid)
        trapezoids = 0.5 * (self.density[1:] + self.density[:-1]) * dx
        cdf = np.concatenate([[0.0], np.cumsum(trapezoids)])
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return cdf

    def moment(self, order: int) -> float:
        """Moment computed by numerical integration."""
        if order < 0:
            raise ValueError("order must be non-negative")
        return float(np.trapz(np.power(self.grid, order) * self.density, self.grid))

    def stieltjes(self, z: complex | np.ndarray) -> complex | np.ndarray:
        """Evaluate Stieltjes transform ``m(z)=∫(t-z)^(-1)f(t)dt`` numerically."""
        z_arr = np.asarray(z, dtype=np.complex128)
        integrand = self.density[None, :] / (self.grid[None, :] - z_arr[..., None])
        values = np.trapz(integrand, self.grid, axis=-1)
        if np.isscalar(z):
            return complex(values.item())
        return values

    def quantiles(self, probs: np.ndarray) -> np.ndarray:
        """Quantiles via interpolation of numerical CDF."""
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        return np.interp(probs, self.cdf(), self.grid)

    def to_discrete(self, n_atoms: int = 400) -> DiscreteSpectrum:
        """Convert grid density to discrete atomic approximation."""
        if n_atoms <= 0:
            raise ValueError("n_atoms must be positive")

        if n_atoms >= self.grid.size:
            dx = np.diff(self.grid)
            cell = np.empty_like(self.grid)
            cell[0] = 0.5 * dx[0]
            cell[-1] = 0.5 * dx[-1]
            cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])
            weights = self.density * cell
            return DiscreteSpectrum(atoms=self.grid, weights=weights, name=f"{self.name}_discrete")

        probs = (np.arange(n_atoms) + 0.5) / n_atoms
        atoms = self.quantiles(probs)
        weights = np.full(n_atoms, 1.0 / n_atoms)
        return DiscreteSpectrum(atoms=atoms, weights=weights, name=f"{self.name}_quantized")


@dataclass(slots=True)
class ParametricSpectrumLaw:
    """Parametric population spectral law.

    Supported ``kind`` values:
    - ``dirac``
    - ``dirac_mixture``
    - ``uniform``
    - ``gamma``
    - ``rescaled_beta``
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    name: str = "parametric"

    def _continuous_quantiles(self, n_atoms: int) -> np.ndarray:
        probs = (np.arange(n_atoms) + 0.5) / n_atoms

        if self.kind == "uniform":
            low = float(self.params["low"])
            high = float(self.params["high"])
            return low + (high - low) * probs

        if self.kind == "gamma":
            shape = float(self.params["shape"])
            scale = float(self.params["scale"])
            return gamma_dist.ppf(probs, a=shape, scale=scale)

        if self.kind == "rescaled_beta":
            alpha = float(self.params["alpha"])
            beta = float(self.params["beta"])
            scale = float(self.params.get("scale", self.params.get("beta_scale", 1.0)))
            shift = float(self.params.get("shift", self.params.get("beta_shift", 0.0)))
            return shift + scale * beta_dist.ppf(probs, a=alpha, b=beta)

        raise ValueError(f"Continuous quantiles not available for kind={self.kind}")

    def to_discrete(self, n_atoms: int = 400) -> DiscreteSpectrum:
        """Discretize the parametric law to atomic measure."""
        if n_atoms <= 0:
            raise ValueError("n_atoms must be positive")

        if self.kind == "dirac":
            value = float(self.params["value"])
            return DiscreteSpectrum(atoms=np.array([value]), weights=np.array([1.0]), name=self.name or "dirac")

        if self.kind == "dirac_mixture":
            values = np.asarray(self.params["values"], dtype=float)
            weights = np.asarray(self.params.get("weights", np.ones(values.size)), dtype=float)
            return DiscreteSpectrum(atoms=values, weights=weights, name=self.name or "dirac_mixture")

        atoms = self._continuous_quantiles(n_atoms=n_atoms)
        weights = np.full(n_atoms, 1.0 / n_atoms)
        return DiscreteSpectrum(atoms=atoms, weights=weights, name=self.name or self.kind)

    def density(self, grid: np.ndarray) -> GridDensity:
        """Evaluate law density on grid (dirac laws are smoothed)."""
        x = np.asarray(grid, dtype=float)

        if self.kind == "dirac":
            return self.to_discrete(1).to_grid_density(x)

        if self.kind == "dirac_mixture":
            return self.to_discrete(len(self.params.get("values", []))).to_grid_density(x)

        if self.kind == "uniform":
            low = float(self.params["low"])
            high = float(self.params["high"])
            dens = np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)
            return GridDensity(grid=x, density=dens, name=self.name or "uniform")

        if self.kind == "gamma":
            shape = float(self.params["shape"])
            scale = float(self.params["scale"])
            dens = gamma_dist.pdf(x, a=shape, scale=scale)
            return GridDensity(grid=x, density=dens, name=self.name or "gamma")

        if self.kind == "rescaled_beta":
            alpha = float(self.params["alpha"])
            beta = float(self.params["beta"])
            scale = float(self.params.get("scale", self.params.get("beta_scale", 1.0)))
            shift = float(self.params.get("shift", self.params.get("beta_shift", 0.0)))

            y = (x - shift) / max(scale, 1e-12)
            dens = np.where((y > 0) & (y < 1), beta_dist.pdf(y, a=alpha, b=beta) / max(scale, 1e-12), 0.0)
            return GridDensity(grid=x, density=dens, name=self.name or "rescaled_beta")

        raise ValueError(f"Unsupported parametric law kind: {self.kind}")


def dirac_law(value: float, name: str = "dirac") -> ParametricSpectrumLaw:
    """Construct a Dirac population law."""
    return ParametricSpectrumLaw(kind="dirac", params={"value": float(value)}, name=name)


def dirac_mixture_law(values: np.ndarray, weights: np.ndarray | None = None, name: str = "dirac_mixture") -> ParametricSpectrumLaw:
    """Construct a finite Dirac mixture law."""
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.size == 0:
        raise ValueError("values must be non-empty")
    if weights is None:
        wts = np.ones(vals.size)
    else:
        wts = np.asarray(weights, dtype=float).reshape(-1)
        if wts.size != vals.size:
            raise ValueError("weights must have same length as values")
    return ParametricSpectrumLaw(kind="dirac_mixture", params={"values": vals, "weights": wts}, name=name)


def uniform_law(low: float, high: float, name: str = "uniform") -> ParametricSpectrumLaw:
    """Construct uniform spectral law on [low, high]."""
    return ParametricSpectrumLaw(kind="uniform", params={"low": float(low), "high": float(high)}, name=name)


def gamma_law(shape: float, scale: float, name: str = "gamma") -> ParametricSpectrumLaw:
    """Construct gamma spectral law."""
    return ParametricSpectrumLaw(kind="gamma", params={"shape": float(shape), "scale": float(scale)}, name=name)


def rescaled_beta_law(alpha: float, beta: float, scale: float, shift: float = 0.0, name: str = "rescaled_beta") -> ParametricSpectrumLaw:
    """Construct rescaled-beta spectral law: shift + scale * Beta(alpha, beta)."""
    return ParametricSpectrumLaw(
        kind="rescaled_beta",
        params={"alpha": float(alpha), "beta": float(beta), "scale": float(scale), "shift": float(shift)},
        name=name,
    )


def empirical_discrete_law(eigenvalues: np.ndarray, name: str = "empirical_discrete") -> DiscreteSpectrum:
    """Construct empirical discrete law from sampled eigenvalues."""
    vals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    return DiscreteSpectrum(atoms=vals, weights=None, name=name)


PopulationLawInput = DiscreteSpectrum | GridDensity | ParametricSpectrumLaw


def to_discrete_spectrum(law: PopulationLawInput, n_atoms: int = 400) -> DiscreteSpectrum:
    """Convert a supported law object to :class:`DiscreteSpectrum`."""
    if isinstance(law, DiscreteSpectrum):
        return law
    if isinstance(law, GridDensity):
        return law.to_discrete(n_atoms=n_atoms)
    if isinstance(law, ParametricSpectrumLaw):
        return law.to_discrete(n_atoms=n_atoms)
    if hasattr(law, "to_discrete"):
        return law.to_discrete(n_atoms=n_atoms)  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported population law type: {type(law)}")
