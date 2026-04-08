"""Plotting functions for spectral densities and laws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity


def plot_grid_density(
    density: GridDensity,
    ax: plt.Axes | None = None,
    label: str | None = None,
    linewidth: float = 2.0,
) -> plt.Axes:
    """Plot one density curve."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(density.grid, density.density, label=label or density.name, linewidth=linewidth)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density")
    return ax


def plot_density_comparison(
    densities: list[GridDensity],
    labels: list[str] | None = None,
    title: str = "Spectral Density Comparison",
    figsize: tuple[float, float] = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot several densities on the same axis."""
    fig, ax = plt.subplots(figsize=figsize)
    for idx, density in enumerate(densities):
        label = labels[idx] if labels is not None else density.name
        plot_grid_density(density, ax=ax, label=label)
    ax.set_title(title)
    if labels is not None or any(d.name for d in densities):
        ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_population_forward_recovered(
    population_density: GridDensity,
    forward_density: GridDensity,
    recovered_density: GridDensity | None = None,
    figsize: tuple[float, float] = (9, 5),
    title: str = "Population, MP Forward, and Recovered Densities",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot population, forward, and optional recovered densities."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_grid_density(population_density, ax=ax, label="population")
    plot_grid_density(forward_density, ax=ax, label="MP forward")
    if recovered_density is not None:
        plot_grid_density(recovered_density, ax=ax, label="inverse recovered")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_discrete_spectrum(
    spectrum: DiscreteSpectrum,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Plot a discrete spectral law as weighted stems."""
    if ax is None:
        _, ax = plt.subplots()
    markerline, stemlines, baseline = ax.stem(spectrum.atoms, spectrum.weights, label=label or spectrum.name)
    plt.setp(stemlines, linewidth=1.2)
    plt.setp(markerline, markersize=4)
    baseline.set_visible(False)
    ax.set_xlabel("eigenvalue atom")
    ax.set_ylabel("weight")
    return ax


def plot_eigen_histogram(
    eigenvalues: np.ndarray,
    bins: int = 45,
    density: bool = True,
    ax: plt.Axes | None = None,
    label: str = "empirical eigenvalues",
    alpha: float = 0.5,
) -> plt.Axes:
    """Plot histogram of eigenvalues."""
    values = np.asarray(eigenvalues, dtype=float)
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(values, bins=bins, density=density, alpha=alpha, label=label)
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("density" if density else "count")
    return ax
