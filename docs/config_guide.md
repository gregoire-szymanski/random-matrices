# Config Guide

This document describes the YAML configuration format used across all `mpdiff` runners.

## 1. Top-Level Sections

Typical config layout:

```yaml
global: ...
simulation: ...
volatility: ...
mp_forward: ...
mp_inverse: ...
analysis: ...
plotting: ...
benchmark: ...
```

All catalog files under `configs/config_*.yaml` follow this structure.

## 2. `global`

Controls reproducibility and output behavior.

```yaml
global:
  seed: 20260408
  output_dir: outputs/my_experiment
  log_level: INFO
  save_figures: true
  save_arrays: true
  save_metadata: true
```

- `seed`: random seed for deterministic runs.
- `output_dir`: target directory for arrays, plots, tables, metadata.
- `log_level`: logger verbosity.
- save switches: control figure/array/report persistence.

## 3. `simulation`

Defines diffusion dimension and Euler time discretization.

```yaml
simulation:
  d: 100
  T: 1.0
  n_steps: 500
  initial_state: 0.0
  drift_model:
    kind: zero
```

Supported drift kinds:

- `zero`
- `constant`
- `linear_mean_reversion`
- `time_sine`
- `callable`

## 4. `volatility`

Defines covariance model over time.

### 4.1 Constant mode

```yaml
volatility:
  mode: constant
  constant_model:
    kind: diag_scalar
    scalar: 1.2
```

### 4.2 Piecewise mode

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

Segments must be contiguous, ordered, and end exactly at `simulation.T`.

### 4.3 Piecewise scalar-times-base mode

```yaml
volatility:
  mode: piecewise_scaled_base
  scaled_base:
    base_matrix_policy: common_fixed
    base_model: ...
  segments:
    - start: 0.0
      end: 0.5
      scalar: 0.8
    - start: 0.5
      end: 1.0
      scalar: 1.4
```

- `common_fixed`: one base matrix shared by all segments.
- `redraw_per_segment`: independent base matrix draw per segment.

## 5. Covariance Model Kinds

### 5.1 `diag_scalar`

`Cov = lambda I_d`.

### 5.2 `diag_distribution`

Diagonal covariance with eigenvalues drawn from:

- `dirac`
- `dirac_mixture`
- `uniform`
- `gamma`
- `rescaled_beta`

### 5.3 `orthogonal_diag`

`Cov = U diag(lambda) U^T`.

Orthogonal generation:

- `haar`
- `identity` (debugging)

### 5.4 `low_rank_plus_diag`

`Cov = B Sigma B^T + D` with `rank(Sigma)=K<d`.

Key knobs:

- `rank`
- latent eigenvalue distribution for `Sigma`
- `factor.method` for `B`: `gaussian`, `identity_block`, `from_file`
- diagonal noise for `D`: `scalar_identity` or `distribution`

## 6. `mp_forward`

MP forward transform numerics.

```yaml
mp_forward:
  aspect_ratio: 0.24
  grid_min: 0.0
  grid_max: 6.0
  num_points: 540
  eta: 0.0035
  tol: 1.0e-9
  max_iter: 540
  damping: 0.72
```

Notation: `aspect_ratio = c = d/n`.

## 7. `mp_inverse`

Inverse method controls.

```yaml
mp_inverse:
  method: optimization
  compare_methods: [optimization, fixed_point, stieltjes_based]
  # or: compare_all_methods: true
  n_support: 55
  support_min: 0.05
  support_max: 4.0
  eta: 0.0035
  tol: 1.0e-6
  max_iter: 320
  regularization: 8.0e-4
```

Method-specific subsections:

- `fixed_point`
- `optimization`
- `stieltjes_based`
- `moment_based`

See `docs/inversion_methods.md` for details.

## 8. `analysis`

Controls end-to-end spectral analysis behavior.

```yaml
analysis:
  realized_covariance_normalization: total_time
  empirical_density_bandwidth: null
  empirical_histogram_bins: 50
```

`realized_covariance_normalization` options:

- `total_time`
- `n_steps`
- `n_steps_minus_one`
- `none`

## 9. `plotting` and `benchmark`

`plotting` controls figure style and toggles. `benchmark.enabled` toggles timer logging.

## 10. Catalog Configs and Their Purpose

- `config_constant_isotropic.yaml`: baseline isotropic law.
- `config_diagonal_dirac_mixture.yaml`: multimodal diagonal spectral law.
- `config_diagonal_uniform.yaml`: compact continuous law.
- `config_diagonal_gamma.yaml`: skewed continuous law (+ compare-all mode).
- `config_diagonal_rescaled_beta.yaml`: bounded asymmetric law.
- `config_rotated_haar.yaml`: rotated covariance, same eigenvalue law as underlying diagonal.
- `config_low_rank_plus_diag.yaml`: factor + noise spectrum.
- `config_piecewise_two_segments.yaml`: time-varying two-regime model.
- `config_piecewise_scalar_times_base_shared.yaml`: segment scalings on a shared base matrix.
- `config_piecewise_scalar_times_base_redrawn.yaml`: independent segment base redraws.

## 11. Common Pitfalls

- `volatility.segments` not contiguous or not ending at `T`.
- Using both `compare_all_methods: true` and non-empty `compare_methods`.
- Too small `eta` causing unstable fixed-point solves.
- Too narrow inverse support (`support_min`, `support_max`) clipping true law.
- Over-interpreting finite-sample differences as model failure.
