"""Integration test for full end-to-end pipeline runner outputs."""

from __future__ import annotations

from pathlib import Path

from mpdiff.experiments.run_full_pipeline import run_full_pipeline


def test_run_full_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
global:
  seed: 42
  output_dir: {out_dir}
  log_level: INFO
  save_figures: false
  save_arrays: true
  save_metadata: true

simulation:
  d: 10
  T: 1.0
  n_steps: 40
  initial_state: 0.0
  drift_model:
    kind: zero

volatility:
  mode: constant
  constant_model:
    kind: diag_scalar
    scalar: 1.1

mp_forward:
  aspect_ratio: 0.25
  grid_min: 0.0
  grid_max: 4.0
  num_points: 160
  eta: 0.003
  tol: 1.0e-8
  max_iter: 220
  damping: 0.7

mp_inverse:
  method: moment_based
  compare_methods: [moment_based]
  n_support: 16
  support_min: 0.05
  support_max: 3.0
  eta: 0.003
  max_iter: 120
  forward_max_iter: 140
  forward_tol: 1.0e-7

analysis:
  realized_covariance_normalization: n_steps
  empirical_histogram_bins: 30

plotting:
  show: false

benchmark:
  enabled: false
""".format(out_dir=str(out_dir).replace("\\", "/")),
        encoding="utf-8",
    )

    summary = run_full_pipeline(config_path)
    assert summary["method"] == "moment_based"

    expected = [
        "full_pipeline_paths.npy",
        "full_pipeline_increments.npy",
        "full_pipeline_realized_covariance.npy",
        "full_pipeline_realized_eigenvalues.npy",
        "full_pipeline_empirical_density.npz",
        "full_pipeline_method_summary.csv",
        "full_pipeline_method_summary.json",
        "full_pipeline_metadata.json",
        "full_pipeline_report.txt",
    ]
    for filename in expected:
        assert (out_dir / filename).exists(), f"Missing output: {filename}"
