"""Marcenko-Pastur forward transform via Stieltjes fixed-point equations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import root

from .densities import DiscreteSpectrum, GridDensity, PopulationLawInput, to_discrete_spectrum


@dataclass(slots=True)
class MPForwardResult:
    """Output bundle for MP forward transform."""

    population_discrete: DiscreteSpectrum
    transformed_density: GridDensity
    stieltjes_values: np.ndarray
    c: float
    epsilon: float
    diagnostics: dict[str, Any]


def make_upper_half_plane_grid(real_grid: np.ndarray, epsilon: float) -> np.ndarray:
    """Build a complex grid ``z = x + i epsilon`` from real-axis points."""
    x = np.asarray(real_grid, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return x + 1j * epsilon


def _silverstein_residual(
    m_value: complex,
    z: complex,
    atoms: np.ndarray,
    weights: np.ndarray,
    c: float,
) -> complex:
    denom = atoms * (1.0 - c - c * z * m_value) - z
    denom = np.where(np.abs(denom) < 1e-15, denom + 1e-15, denom)
    rhs = np.sum(weights / denom)
    return m_value - rhs


def _solve_with_newton(
    initial_m: complex,
    z: complex,
    population: DiscreteSpectrum,
    c: float,
) -> tuple[complex, bool, float]:
    """Fallback nonlinear solve for the Silverstein equation."""

    def equations(vec: np.ndarray) -> np.ndarray:
        m_val = complex(float(vec[0]), float(vec[1]))
        residual = _silverstein_residual(m_val, z, population.atoms, population.weights, c)
        return np.array([residual.real, residual.imag], dtype=float)

    guess = np.array([initial_m.real, initial_m.imag], dtype=float)
    result = root(equations, guess, method="hybr")
    if not result.success:
        return initial_m, False, float(np.linalg.norm(result.fun))

    solved = complex(float(result.x[0]), float(result.x[1]))
    if solved.imag <= 0:
        solved = complex(solved.real, max(abs(solved.imag), 1e-12))

    residual_norm = float(np.linalg.norm(equations(np.array([solved.real, solved.imag], dtype=float))))
    return solved, True, residual_norm


def mp_stieltjes_fixed_point(
    population: DiscreteSpectrum,
    z: complex,
    c: float,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    initial_m: complex | None = None,
    use_newton_fallback: bool = True,
) -> tuple[complex, int, bool, float, bool]:
    """Solve Silverstein fixed-point equation for ``m_F(z)``.

    Equation:
    ``m(z) = ∫ 1 / (t (1 - c - c z m(z)) - z) dH(t)``

    where ``H`` is the population spectral law and ``c = d/n``.

    Returns
    -------
    tuple
        ``(m_value, n_iter, converged, residual, used_newton_fallback)``.
    """
    if c <= 0:
        raise ValueError("c (aspect ratio) must be positive")

    damp = float(np.clip(damping, 1e-4, 1.0))
    m = initial_m if initial_m is not None else -1.0 / z
    residual = np.inf

    for it in range(1, max_iter + 1):
        denom = population.atoms * (1.0 - c - c * z * m) - z
        denom = np.where(np.abs(denom) < 1e-15, denom + 1e-15, denom)
        rhs = np.sum(population.weights / denom)
        m_next = damp * rhs + (1.0 - damp) * m
        residual = abs(m_next - m)
        if residual < tol:
            if m_next.imag <= 0:
                m_next = complex(m_next.real, max(abs(m_next.imag), 1e-12))
            return m_next, it, True, float(residual), False
        m = m_next

    if use_newton_fallback:
        solved, ok, newton_residual = _solve_with_newton(m, z, population, c)
        if ok:
            return solved, max_iter, True, float(min(residual, newton_residual)), True

    if m.imag <= 0:
        m = complex(m.real, max(abs(m.imag), 1e-12))
    return m, max_iter, False, float(residual), False


def evaluate_stieltjes_transform(
    population: PopulationLawInput,
    z_grid: np.ndarray,
    c: float,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    n_population_atoms: int = 400,
    use_newton_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, Any], DiscreteSpectrum]:
    """Evaluate transformed Stieltjes values on a complex grid."""
    discrete = to_discrete_spectrum(population, n_atoms=n_population_atoms)

    z_values = np.asarray(z_grid, dtype=np.complex128).reshape(-1)
    m_values = np.empty_like(z_values)
    iterations = np.zeros(z_values.size, dtype=int)
    converged = np.zeros(z_values.size, dtype=bool)
    residuals = np.zeros(z_values.size, dtype=float)
    used_fallback = np.zeros(z_values.size, dtype=bool)

    previous_m: complex | None = None
    for idx, z in enumerate(z_values):
        m, it_count, ok, residual, used_newton = mp_stieltjes_fixed_point(
            population=discrete,
            z=complex(z),
            c=c,
            tol=tol,
            max_iter=max_iter,
            damping=damping,
            initial_m=previous_m,
            use_newton_fallback=use_newton_fallback,
        )
        m_values[idx] = m
        iterations[idx] = it_count
        converged[idx] = ok
        residuals[idx] = residual
        used_fallback[idx] = used_newton
        previous_m = m

    diagnostics: dict[str, Any] = {
        "mean_iterations": float(np.mean(iterations)),
        "max_iterations": int(np.max(iterations)),
        "convergence_rate": float(np.mean(converged)),
        "max_residual": float(np.max(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "newton_fallback_rate": float(np.mean(used_fallback)),
        "iterations": iterations,
        "residuals": residuals,
        "converged_mask": converged,
        "used_newton_fallback": used_fallback,
    }
    return m_values, diagnostics, discrete


def density_from_stieltjes(stieltjes_values: np.ndarray) -> np.ndarray:
    """Recover spectral density from Stieltjes values."""
    return np.clip(np.imag(np.asarray(stieltjes_values, dtype=np.complex128)) / np.pi, 0.0, None)


def compute_mp_forward(
    population: PopulationLawInput,
    c: float,
    grid: np.ndarray,
    epsilon: float = 1e-3,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    n_population_atoms: int = 400,
    use_newton_fallback: bool = True,
) -> MPForwardResult:
    """Compute MP forward image density on a real grid.

    Parameters
    ----------
    population:
        Population law ``H`` (parametric, discrete, or grid-based).
    c:
        Aspect ratio ``c=d/n``.
    grid:
        Real-axis grid for evaluating transformed density.
    epsilon:
        Imaginary regularization in ``z = x + i epsilon``.
    """
    x = np.asarray(grid, dtype=float).reshape(-1)
    z_grid = make_upper_half_plane_grid(x, epsilon)

    start = perf_counter()
    m_values, diagnostics, discrete = evaluate_stieltjes_transform(
        population=population,
        z_grid=z_grid,
        c=c,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
        n_population_atoms=n_population_atoms,
        use_newton_fallback=use_newton_fallback,
    )
    elapsed = perf_counter() - start

    density = density_from_stieltjes(m_values)
    transformed = GridDensity(grid=x, density=density, name="mp_forward")

    diagnostics.update(
        {
            "elapsed_seconds": float(elapsed),
            "c": float(c),
            "epsilon": float(epsilon),
            "tol": float(tol),
            "max_iter": int(max_iter),
            "damping": float(damping),
            "n_population_atoms": int(discrete.atoms.size),
        }
    )

    return MPForwardResult(
        population_discrete=discrete,
        transformed_density=transformed,
        stieltjes_values=m_values,
        c=float(c),
        epsilon=float(epsilon),
        diagnostics=diagnostics,
    )


def mp_forward_transform(
    population: PopulationLawInput,
    aspect_ratio: float,
    grid: np.ndarray,
    eta: float = 1e-3,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.7,
    return_diagnostics: bool = False,
    return_result: bool = False,
    n_population_atoms: int = 400,
    use_newton_fallback: bool = True,
) -> GridDensity | tuple[GridDensity, dict[str, Any]] | MPForwardResult:
    """Backward-compatible wrapper for MP forward transform.

    Notes
    -----
    - ``aspect_ratio`` is the notation ``c=d/n``.
    - ``eta`` is the imaginary regularization ``epsilon``.
    """
    result = compute_mp_forward(
        population=population,
        c=aspect_ratio,
        grid=grid,
        epsilon=eta,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
        n_population_atoms=n_population_atoms,
        use_newton_fallback=use_newton_fallback,
    )

    if return_result:
        return result
    if return_diagnostics:
        return result.transformed_density, result.diagnostics
    return result.transformed_density


def mp_dirac_density(
    grid: np.ndarray,
    variance: float,
    c: float | None = None,
    aspect_ratio: float | None = None,
) -> np.ndarray:
    """Closed-form MP density for a single Dirac population eigenvalue."""
    x = np.asarray(grid, dtype=float)
    ratio = c if c is not None else aspect_ratio
    if ratio is None:
        raise ValueError("Either c or aspect_ratio must be provided")

    ratio = max(float(ratio), 1e-12)
    if variance <= 0:
        return np.zeros_like(x)

    sqrt_c = np.sqrt(ratio)
    a = variance * (1.0 - sqrt_c) ** 2
    b = variance * (1.0 + sqrt_c) ** 2

    density = np.zeros_like(x)
    mask = (x > max(a, 0.0)) & (x < b)
    numerator = np.sqrt((b - x[mask]) * (x[mask] - a))
    denominator = 2.0 * np.pi * ratio * variance * np.maximum(x[mask], 1e-15)
    density[mask] = numerator / denominator
    return density
