# Inversion Methods Guide

This document explains the MP inverse methods implemented in `mpdiff`, their assumptions, and practical usage.

## 1. Problem Statement

Given an observed sample spectral density `F` (typically from realized covariance), estimate a population spectral law `H` such that `MP_c(H)` approximately matches `F`, where `c=d/n`.

This inverse map is ill-posed in finite samples. Different regularized approximations are provided.

## 2. Method Interface

All methods follow the same callable interface internally:

`invert(observed_density, aspect_ratio, inverse_config, forward_config)`

Output includes:

- estimated population law (discrete atoms/weights),
- forward reconstruction of observed density,
- method diagnostics.

## 3. Implemented Methods

### 3.1 `fixed_point`

File: `src/mpdiff/spectral/inversion_methods/fixed_point.py`

Core idea:

- Build a dictionary of single-atom MP kernels over a support grid.
- Solve for nonnegative mixture weights using Richardson-Lucy style multiplicative updates.
- Optional smoothing regularizes oscillatory weights.

Key config knobs:

- `n_support`, `support_min`, `support_max`
- `max_iter`, `tol`
- `regularization`
- `fixed_point.smoothing_strength`

Pros:

- flexible nonparametric shape recovery,
- relatively interpretable support/weight representation.

Limitations:

- sensitive to support range,
- can overfit noise without enough smoothing.

### 3.2 `optimization`

File: `src/mpdiff/spectral/inversion_methods/optimization.py`

Core idea:

- Parameterize population law as support grid + softmax weights.
- Minimize mismatch between observed density and forward MP prediction.
- Add smoothness penalty on weight curvature.

Key config knobs:

- `optimization.optimizer`
- `optimization.max_iter` and `optimizer_max_iter`
- `regularization`

Pros:

- generally strongest reconstruction quality in many smooth problems.

Limitations:

- heavier runtime,
- non-convex objective can have local minima.

### 3.3 `stieltjes_based`

File: `src/mpdiff/spectral/inversion_methods/stieltjes_based.py`

Core idea:

- Estimate Stieltjes transform from observed density.
- Apply nonlinear shrinkage mapping pointwise on quantiles.

Key config knobs:

- `stieltjes_based.quantile_min`
- `stieltjes_based.quantile_max`
- `eta`

Pros:

- fast and simple.

Limitations:

- can be unstable/noisy near edges and tails,
- sensitive to regularization parameter `eta`.

### 3.4 `moment_based`

File: `src/mpdiff/spectral/inversion_methods/moment_based.py`

Core idea:

- Use low-order MP moment identities to infer moments of `H`.
- Fit a gamma-family approximation from inferred mean/variance.

Key config knobs:

- `moment_based.family` (currently `gamma`)
- `moment_based.min_variance`

Pros:

- robust coarse baseline,
- fast.

Limitations:

- cannot represent multimodal fine structure.

## 4. Choosing a Method

Suggested default order:

1. `optimization` for best quality when runtime is acceptable.
2. `fixed_point` for nonparametric but faster alternatives.
3. `moment_based` as robust low-complexity sanity baseline.
4. `stieltjes_based` for quick exploratory runs.

## 5. Compare-All and Benchmarking

### Config-driven compare-all

```yaml
mp_inverse:
  method: optimization
  compare_all_methods: true
```

### Selective comparison

```yaml
mp_inverse:
  method: optimization
  compare_methods: [optimization, fixed_point, moment_based]
```

### Dedicated benchmark runner

```bash
mpdiff inversion-benchmark --config configs/config_diagonal_gamma.yaml
```

Outputs:

- `inversion_benchmark_summary.csv`
- `inversion_benchmark_summary.json`
- reconstruction/quality/runtime plots.

## 6. Interpreting Metrics

Main metrics reported per method:

- `population_wasserstein_1`: distance between recovered and reference population densities,
- `population_l2`: shape mismatch,
- `reconstruction_l2`: mismatch between observed density and forward(recovered),
- `runtime_seconds`: computational cost.

No single metric is sufficient. Prefer methods that jointly:

- keep population-distance low,
- keep reconstruction error low,
- remain stable under small config perturbations.

## 7. Common Failure Modes

- **Support misspecification**: true law outside `[support_min, support_max]`.
- **Over-regularization**: oversmoothed recovered law.
- **Under-regularization**: unstable oscillatory weights.
- **Too small `eta`**: unstable forward fixed-point solve.

Practical mitigation:

- widen support range,
- increase `n_support` gradually,
- increase `eta` slightly,
- compare methods instead of relying on one run.
