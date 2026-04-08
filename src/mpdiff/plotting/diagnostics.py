"""Plotting helpers for MP inverse diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt

from mpdiff.spectral.densities import DiscreteSpectrum, GridDensity

from .spectra import plot_discrete_spectrum


def plot_inverse_diagnostics(
    observed: GridDensity,
    reconstructed: GridDensity,
    estimated_population: DiscreteSpectrum,
    reference_population: DiscreteSpectrum | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot observed/reconstructed densities and estimated population law."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(observed.grid, observed.density, label="observed", linewidth=2.0)
    axes[0].plot(reconstructed.grid, reconstructed.density, label="reconstructed", linewidth=2.0)
    axes[0].set_xlabel("eigenvalue")
    axes[0].set_ylabel("density")
    axes[0].set_title("Observed vs Reconstructed")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    plot_discrete_spectrum(estimated_population, ax=axes[1], label="estimated")
    if reference_population is not None:
        plot_discrete_spectrum(reference_population, ax=axes[1], label="reference")
    axes[1].set_title("Population Spectrum")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig, (axes[0], axes[1])
