# Spectral Config Guide

This guide documents the YAML structure used by `mp-forward`, `mp-inverse`, and spectral portions of `full-pipeline`.

## 1. MP Forward Section

```yaml
mp_forward:
  aspect_ratio: 0.25   # c = d / n
  grid_min: 0.0
  grid_max: 5.0
  num_points: 500
  eta: 0.003
  tol: 1.0e-9
  max_iter: 500
  damping: 0.72
```

- `aspect_ratio`: explicit `c`. If omitted, runners use `simulation.d / simulation.n_steps`.
- `eta`: imaginary regularization in `z = x + i*eta`.
- `tol`, `max_iter`, `damping`: forward fixed-point solver controls.

## 2. MP Inverse Section

```yaml
mp_inverse:
  method: optimization
  compare_methods: [optimization, fixed_point, stieltjes_based]
  n_support: 55
  support_min: 0.1
  support_max: 3.5
  eta: 0.0035
  tol: 1.0e-6
  max_iter: 300
  regularization: 1.0e-3
  optimizer_max_iter: 120
  forward_max_iter: 250
  forward_tol: 1.0e-7
```

- `method`: primary method used for summary outputs.
- `compare_methods`: optional list of methods to run on the same observed density.
- `n_support`: support-grid size for inverse discretizations.
- `support_min`, `support_max`: inversion support bounds.
- `eta`: inverse-side Stieltjes regularization.
- `regularization`: generic smoothness strength used by nonparametric methods.

## 3. Method-Specific Inverse Subsections

### `fixed_point`

```yaml
mp_inverse:
  fixed_point:
    smoothing_strength: 0.08
    min_kernel_density: 1.0e-16
```

### `optimization`

```yaml
mp_inverse:
  optimization:
    optimizer: L-BFGS-B
    max_iter: 120
```

### `stieltjes_based`

```yaml
mp_inverse:
  stieltjes_based:
    quantile_min: 0.02
    quantile_max: 0.98
```

### `moment_based`

```yaml
mp_inverse:
  moment_based:
    family: gamma
    min_variance: 1.0e-10
```

## 4. Population Spectral Law Input

To use direct population laws in spectral experiments, configure:

```yaml
analysis:
  population_spectrum:
    source: parametric
    kind: gamma
    params:
      shape: 2.5
      scale: 0.6
    n_atoms: 500
```

### Supported `source`

- `from_covariance_model`: derive population law from covariance-model config (default behavior)
- `parametric`
- `atomic`
- `grid`
- `empirical`

### `parametric` kinds

- `dirac`
- `dirac_mixture`
- `uniform`
- `gamma`
- `rescaled_beta`

### `atomic` example

```yaml
analysis:
  population_spectrum:
    source: atomic
    atoms: [0.5, 1.1, 2.0]
    weights: [0.2, 0.5, 0.3]
```

### `grid` example

```yaml
analysis:
  population_spectrum:
    source: grid
    grid: [0.2, 0.5, 1.0, 1.5]
    density: [0.1, 0.4, 0.35, 0.15]
    n_atoms: 300
```

### `empirical` example

```yaml
analysis:
  population_spectrum:
    source: empirical
    eigenvalues: [0.4, 0.6, 0.8, 1.2, 1.6]
```

or

```yaml
analysis:
  population_spectrum:
    source: empirical
    eigenvalues_path: data/eigs.npy
```

## 5. Spectral Example Configs

See `configs/spectral_examples/`:

- `01_dirac_forward_inverse.yaml`
- `02_atomic_mixture_compare_methods.yaml`
- `03_gamma_population.yaml`
- `04_rescaled_beta_population.yaml`
- `05_empirical_discrete_inline.yaml`
- `06_grid_density_input.yaml`

Each file includes inline comments and runs directly with the CLI.
