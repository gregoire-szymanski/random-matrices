# Simulation Config Guide

This guide documents the YAML schema used by the diffusion simulation runner.

## 1. Global Section

```yaml
global:
  seed: 20260408
  output_dir: outputs/my_experiment
  log_level: INFO
  save_figures: true
  save_arrays: true
  save_metadata: true
```

- `seed`: reproducible RNG seed.
- `output_dir`: run artifacts target folder.
- save toggles control arrays, figures, metadata JSON.

## 2. Simulation Section

```yaml
simulation:
  d: 100
  T: 1.0
  n_steps: 500
  initial_state: 0.0
  drift_model:
    kind: zero
```

- `d`: diffusion dimension.
- `T`: horizon.
- `n_steps`: number of Euler steps (`n+1` grid points).
- `initial_state`: scalar (broadcasted) or vector length `d`.
- optional `time_grid`: explicit increasing grid from `0` to `T`.

### Drift models

- `zero`
- `constant` with `vector`
- `linear_mean_reversion` with `theta`, `target`
- `time_sine` with `amplitude`, `frequency`, `phase`, `direction`
- `callable` with `callable_path: module:function` and optional `callable_kwargs`

## 3. Volatility Section

### 3.1 Constant

```yaml
volatility:
  mode: constant
  constant_model:
    kind: diag_scalar
    scalar: 1.0
```

### 3.2 Piecewise

```yaml
volatility:
  mode: piecewise
  segments:
    - start: 0.0
      end: 0.4
      scalar: 1.0
      model: ...
    - start: 0.4
      end: 1.0
      scalar: 1.2
      model: ...
```

Segments must be contiguous, ordered, and end exactly at `T`.

### 3.3 Piecewise scalar-times-base

```yaml
volatility:
  mode: piecewise_scaled_base
  scaled_base:
    base_model: ...
    base_matrix_policy: common_fixed
    share_matrix_law_across_segments: true
  segments:
    - start: 0.0
      end: 0.5
      scalar: 0.8
    - start: 0.5
      end: 1.0
      scalar: 1.4
```

Semantics:

- `common_fixed`: one base matrix reused across segments.
- `redraw_per_segment`: independent base draw on each segment.
- `share_matrix_law_across_segments: false` allows segment-level model overrides.

## 4. Covariance Model Definitions

### `diag_scalar`

`Cov = lambda * I_d`.

### `diag_distribution`

Diagonal entries sampled from distribution law.

### `orthogonal_diag`

`Cov = U diag(lambda) U^T` with:

- `orthogonal.method: haar` or `identity`.

### `low_rank_plus_diag`

`Cov = B Sigma B^T + D` with:

- `low_rank.rank = K < d`
- `latent_eigen_distribution` defines Sigma diagonal
- `factor` defines B generation:
  - `gaussian`
  - `identity_block`
  - `from_file` (`matrix_path` required)
- `diagonal_noise` defines D:
  - `scalar_identity`
  - `distribution`

## 5. Distribution Laws

Available `kind` values for eigenvalue/diagonal sampling:

- `dirac`
- `dirac_mixture`
- `uniform`
- `gamma`
- `rescaled_beta`

`rescaled_beta` uses:

```yaml
kind: rescaled_beta
alpha: 2.0
beta: 5.0
beta_scale: 2.0
beta_shift: 0.1
```

## 6. Sampling Policy

Each covariance model has:

```yaml
sampling_policy: draw_once
```

Options:

- `draw_once`: cache and reuse realization for matching model config.
- `redraw_per_segment`: force independent redraw per segment usage.

## 7. Runner Outputs

The simulation runner writes:

- simulated paths and increments,
- realized covariance and eigenvalues,
- integrated target covariance/eigenvalues,
- KDE approximations,
- optional plots,
- metadata with timers and summary statistics.

## 8. Full-Pipeline Analysis Settings

For `full-pipeline` / `end-to-end` runs, the `analysis` section supports:

```yaml
analysis:
  realized_covariance_normalization: total_time
  empirical_density_bandwidth: null
  empirical_histogram_bins: 50
```

- `realized_covariance_normalization`:
  - `total_time` (standard),
  - `n_steps`,
  - `n_steps_minus_one`,
  - `none`.
- `empirical_density_bandwidth`: optional KDE bandwidth override for realized/reference densities.
- `empirical_histogram_bins`: histogram bins in end-to-end comparison plots.

## 9. Example Catalog

Use:

- `configs/simulation_examples/01_constant_isotropic.yaml`
- `configs/simulation_examples/02_constant_diag_gamma.yaml`
- `configs/simulation_examples/03_constant_rotated_haar.yaml`
- `configs/simulation_examples/04_low_rank_plus_diag.yaml`
- `configs/simulation_examples/05_piecewise_two_segments.yaml`
- `configs/simulation_examples/06_piecewise_scaled_base.yaml`
