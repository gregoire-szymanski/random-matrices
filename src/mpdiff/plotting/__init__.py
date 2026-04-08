"""Plotting helpers for spectra and trajectories."""

from .diagnostics import plot_inverse_diagnostics
from .paths import plot_diffusion_paths
from .spectra import (
    plot_density_comparison,
    plot_discrete_spectrum,
    plot_eigen_histogram,
    plot_grid_density,
)

__all__ = [
    "plot_grid_density",
    "plot_density_comparison",
    "plot_discrete_spectrum",
    "plot_eigen_histogram",
    "plot_diffusion_paths",
    "plot_inverse_diagnostics",
]
