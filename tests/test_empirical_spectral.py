"""Tests for empirical spectral utilities."""

from __future__ import annotations

import numpy as np

from mpdiff.spectral.empirical import (
    compute_increments,
    empirical_discrete_spectrum,
    realized_covariance,
    realized_covariance_from_increments,
)


def test_compute_increments_and_realized_covariance_normalizations() -> None:
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    increments = compute_increments(path)
    assert increments.shape == (2, 2)

    rcv_total_time = realized_covariance(path, total_time=1.0, normalization="total_time")
    rcv_n_steps = realized_covariance(path, normalization="n_steps")
    rcv_none = realized_covariance_from_increments(increments, normalization="none")

    expected_sum = np.array([[2.0, 0.0], [0.0, 0.0]])
    assert np.allclose(rcv_total_time, expected_sum)
    assert np.allclose(rcv_n_steps, expected_sum / 2.0)
    assert np.allclose(rcv_none, expected_sum)


def test_empirical_discrete_spectrum_has_unit_weight_sum() -> None:
    eigs = np.array([0.4, 0.8, 1.2, 1.6], dtype=float)
    spectrum = empirical_discrete_spectrum(eigs)

    assert spectrum.atoms.size == eigs.size
    assert np.isclose(float(np.sum(spectrum.weights)), 1.0)
