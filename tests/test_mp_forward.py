"""Tests for MP forward transform."""

from __future__ import annotations

import numpy as np

from mpdiff.spectral.densities import DiscreteSpectrum
from mpdiff.spectral.transforms import mp_forward_transform


def test_mp_forward_preserves_first_moment_for_dirac() -> None:
    population = DiscreteSpectrum(atoms=np.array([2.0]), weights=np.array([1.0]))
    grid = np.linspace(0.01, 6.0, 500)

    density = mp_forward_transform(
        population=population,
        aspect_ratio=0.4,
        grid=grid,
        eta=0.003,
        tol=1e-10,
        max_iter=500,
        damping=0.7,
    )

    assert np.all(density.density >= 0.0)
    assert abs(density.moment(1) - 2.0) < 0.15
