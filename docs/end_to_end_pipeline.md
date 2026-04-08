# End-to-End Pipeline Guide

This guide documents the full workflow used by:

- `mpdiff full-pipeline --config <yaml>`
- `mpdiff end-to-end --config <yaml>`

The pipeline executes:

1. simulate path,
2. compute increments,
3. compute realized covariance,
4. eigendecompose,
5. estimate empirical spectral density,
6. invert MP map,
7. compare recovered population law to simulation reference law.

## 1. Realized Covariance Definition

Given simulated path samples `X_{t_0}, ..., X_{t_n}` with increments
`ΔX_i = X_{t_{i+1}} - X_{t_i}`,

the runner computes:

`RCV = (Σ_i ΔX_i ΔX_i^T) / scale`

where `scale` is controlled by `analysis.realized_covariance_normalization`:

- `total_time` (default): `scale = T` (standard integrated-covariance normalization),
- `n_steps`: `scale = n`,
- `n_steps_minus_one`: `scale = max(n-1, 1)`,
- `none`: `scale = 1`.

## 2. What Is the “Original Population Law”?

For time-varying piecewise volatility, there is no single instantaneous covariance at all times.
The pipeline uses the **time-averaged covariance** as reference:

`Sigma_bar = (1/T) ∫_0^T Sigma_t dt`

and defines the original population spectral law as the eigenvalue law of `Sigma_bar`.

### Constant covariance models

- If volatility is constant, `Sigma_bar = Sigma`, so reference law is exactly the model’s population law.

### Rotated models (`U Sigma U^T`)

- Rotation changes eigenvectors but not eigenvalues.
- Reference spectrum is the eigenvalue law of the rotated covariance (same eigenvalues as diagonal core).

### Low-rank-plus-diagonal (`B Sigma B^T + D`)

- Reference spectrum is the eigenvalue law of the full matrix `B Sigma B^T + D`.
- It is not just the latent `Sigma` eigenvalues and not just `D`; it is the combined model spectrum.

### Piecewise models

- Reference law comes from eigenvalues of integrated/time-averaged covariance over segments.

## 3. Empirical Spectrum Objects

From realized covariance eigenvalues, the runner builds:

- raw eigenvalue histogram (plot),
- KDE density estimate (`GridDensity`),
- empirical discrete law (`DiscreteSpectrum`) with uniform weights.

## 4. MP Inversion Comparison

For each inverse method (primary plus optional `compare_methods`), the runner saves:

- recovered population atoms/weights,
- forward reconstruction of observed density,
- discrepancy metrics.

Metrics include:

- population recovery: `L1`, `L2`, `Wasserstein-1`, support endpoint diffs, moment errors,
- observed reconstruction: `L1`, `L2`, `Wasserstein-1`.

A method summary table is written to:

- `full_pipeline_method_summary.csv`
- `full_pipeline_method_summary.json`

## 5. Outputs

Typical files:

- arrays: path, increments, realized covariance, eigenvalues,
- densities: empirical / reference / reconstructed,
- recovered laws per method (atoms/weights),
- plots: overlay comparisons and histograms,
- reports: `full_pipeline_metadata.json`, `full_pipeline_report.txt`, summary table CSV/JSON.

## 6. Finite-Sample Limitations and Interpretation

In finite `d, n`:

- realized eigenvalues are noisy and broadened,
- KDE smooths empirical spikes and can bias tails,
- MP inverse is ill-posed and regularization-sensitive,
- method disagreement is expected in difficult regimes.

Interpret discrepancies by checking together:

- population recovery Wasserstein/L2,
- observed reconstruction error,
- support and moment errors,
- stability across inverse methods.

If methods disagree strongly, treat recovered law as an approximate diagnostic rather than exact truth.
