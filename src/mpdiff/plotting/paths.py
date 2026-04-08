"""Plotting functions for diffusion trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_diffusion_paths(
    times: np.ndarray,
    path: np.ndarray,
    max_dims: int = 5,
    figsize: tuple[float, float] = (9, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the first dimensions of a diffusion path."""
    fig, ax = plt.subplots(figsize=figsize)
    n_dims = min(max_dims, path.shape[1])
    for dim_idx in range(n_dims):
        ax.plot(times, path[:, dim_idx], label=f"X[{dim_idx}]")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_title("Simulated Diffusion Trajectory")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax
