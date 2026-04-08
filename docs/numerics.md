# Numerical Notes

## Notation

- `H`: population spectral law
- `F`: MP-transformed (sample) spectral law
- `c = d / n`: aspect ratio
- `z = x + i*eta`, with `eta > 0`

## MP Forward Equation

The implementation solves the Silverstein fixed-point equation:

`m_F(z) = ∫ 1 / (t * (1 - c - c*z*m_F(z)) - z) dH(t)`

Then density is recovered from Stieltjes values by:

`f_F(x) = Im(m_F(x + i*eta)) / pi`

## Forward Solver Details

- Population laws are converted to a discrete approximation (`DiscreteSpectrum`) when needed.
- Fixed-point iteration is damped (`damping` in config).
- Warm-start continuation is used along the grid (`m(z_{k-1})` initializes `m(z_k)`).
- If fixed-point iterations do not converge, a Newton/root fallback is attempted.
- Diagnostics include:
  - convergence rate,
  - iteration statistics,
  - residual statistics,
  - fallback usage rate,
  - runtime.

## Inverse Methods

All inverse methods implement the same interface:

`invert(observed_density, aspect_ratio, inverse_config, forward_config)`

### 1) `fixed_point`

- Uses a dictionary of single-atom MP kernels.
- Solves for mixture weights with Richardson-Lucy multiplicative updates.
- Optional smoothing regularizes oscillatory weight profiles.

Best for:
- nonparametric recovery with moderate smoothness assumptions.

Limitations:
- sensitive to support range choice and regularization strength.

### 2) `optimization`

- Parameterizes population law on a fixed support grid.
- Minimizes forward mismatch plus optional smoothness penalty.

Best for:
- higher-quality reconstructions when runtime is acceptable.

Limitations:
- non-convex objective and heavier runtime.

### 3) `stieltjes_based`

- Estimates `m_F` from observed density.
- Applies nonlinear shrinkage mapping on quantiles.

Best for:
- fast approximate recovery.

Limitations:
- can be noisy in tails and sensitive to `eta`.

### 4) `moment_based`

- Uses low-order MP moment identities.
- Fits a gamma approximation for `H`.

Best for:
- robust coarse baseline.

Limitations:
- cannot recover multimodal structure well.

## Failure Modes and Mitigation

- Divergent fixed-point iterations:
  - increase `eta`,
  - increase `damping`,
  - increase `max_iter`.
- Noisy inverse estimate:
  - increase `regularization`,
  - narrow support,
  - increase `n_support` gradually.
- Poor method agreement:
  - compare methods via `mp_inverse.compare_methods`,
  - inspect reconstruction metrics and convergence plots.

## Comparison Metrics

Implemented comparison metrics include:

- density distances: L1, L2,
- Wasserstein-1,
- support endpoint differences,
- absolute moment errors (default moments 1,2,3).

These are reported by inverse runners and saved in metadata outputs.
