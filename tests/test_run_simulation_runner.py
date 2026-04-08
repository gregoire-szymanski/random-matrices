"""Integration test for simulation runner outputs."""

from __future__ import annotations

from pathlib import Path

from mpdiff.experiments.run_simulation import run_simulation


def test_run_simulation_writes_expected_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
global:
  seed: 123
  output_dir: {out_dir}
  log_level: INFO
  save_figures: false
  save_arrays: true
  save_metadata: true

simulation:
  d: 12
  T: 1.0
  n_steps: 80
  initial_state: 0.0
  drift_model:
    kind: zero

volatility:
  mode: constant
  constant_model:
    kind: diag_scalar
    scalar: 1.0

mp_forward:
  grid_min: 0.0
  grid_max: 3.0
  num_points: 200

plotting:
  show: false
  plot_paths: false
  plot_eigen_hist: false
  plot_eigen_density: false

benchmark:
  enabled: false
""".format(out_dir=str(out_dir).replace("\\", "/")),
        encoding="utf-8",
    )

    summary = run_simulation(config_path)
    assert summary["dimension"] == 12

    expected = [
        "times.npy",
        "paths.npy",
        "increments.npy",
        "segment_indices.npy",
        "realized_covariance.npy",
        "realized_eigenvalues.npy",
        "target_population_eigenvalues.npy",
        "metadata.json",
    ]
    for filename in expected:
        assert (out_dir / filename).exists(), f"Missing output: {filename}"
