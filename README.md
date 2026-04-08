# mpdiff

`mpdiff` is a Python 3.11+ project for:

- simulating high-dimensional diffusions with piecewise-constant volatility,
- building covariance models across several structural regimes,
- analyzing empirical and population spectral laws,
- computing Marcenko-Pastur (MP) forward transforms,
- solving MP inverse problems with swappable methods,
- running reproducible experiments via config-driven CLI workflows.

## Main Features

### Diffusion simulation

- Euler simulation on `[0, T]` with `n+1` time points.
- Piecewise-constant covariance schedules.
- Reproducible random generation via global seeds.

### Covariance / volatility models

Supported covariance models include:

1. `diag_scalar`: diagonal with one repeated eigenvalue.
2. `diag_distribution`: diagonal eigenvalues from
   - `dirac`,
   - finite `dirac_mixture`,
   - `uniform`,
   - `gamma`,
   - `rescaled_beta` (`a * Beta(alpha, beta) + b`).
3. `orthogonal_diag`: `U diag(λ) U^T` with configurable orthogonal generation.
4. `low_rank_plus_diag`: `B Σ B^T + D`, with low-rank latent + diagonal noise.
5. Piecewise mode where each segment is any model above.
6. Piecewise scaled-base mode where each segment is a scalar times a base random matrix law, with explicit controls for shared/fixed/redrawn base matrices.

### Spectral analysis

- Empirical spectral density estimation from realized covariance.
- MP forward transform using Silverstein fixed-point equations.
- MP inverse with multiple methods:
  - `optimization`
  - `fixed_point`
  - `stieltjes_based`
  - `moment_based`

## Project Layout

```text
project_root/
  pyproject.toml
  README.md
  src/
    mpdiff/
      config/
      simulation/
      spectral/
      plotting/
      utils/
      experiments/
      cli.py
  configs/
  notebooks/
  tests/
  docs/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## CLI Usage

Run one of the provided experiment modes:

```bash
mpdiff simulation --config configs/constant_diag_dirac.yaml
mpdiff mp-forward --config configs/constant_diag_dirac.yaml
mpdiff mp-inverse --config configs/constant_diag_dirac.yaml
mpdiff full-pipeline --config configs/piecewise_models.yaml
```

Outputs (arrays, plots, logs) are written to `global.output_dir`.

## Config System

Configs can be YAML or TOML. The schema includes:

- global runtime controls (`seed`, logging, output paths),
- simulation (`d`, `T`, `n_steps`, drift, optional custom time grid),
- volatility mode and covariance model definitions,
- MP forward and inverse numerical tolerances/iterations,
- plotting and benchmarking switches.

Example segment-based volatility:

```yaml
volatility:
  mode: piecewise
  segments:
    - start: 0.0
      end: 0.4
      scalar: 1.0
      model:
        kind: diag_distribution
        eigen_distribution:
          kind: uniform
          low: 0.4
          high: 1.1
    - start: 0.4
      end: 1.0
      scalar: 0.8
      model:
        kind: low_rank_plus_diag
        low_rank:
          rank: 8
```

## Inverse Method Switching

Set `mp_inverse.method` to compare approaches in identical pipelines:

- `fixed_point`: fast kernel deconvolution-style update.
- `optimization`: minimizes forward mismatch under smoothness penalty.
- `stieltjes_based`: pointwise inversion using estimated Stieltjes transform.
- `moment_based`: low-order moment matching + parametric fit.

## Tests

```bash
pytest
```

The test suite covers config loading, covariance construction PSD checks, simulation reproducibility, MP forward numerical sanity, and inverse recovery quality.

## Notebooks

Starter notebooks are provided in `notebooks/` for:

- forward/inverse MP demonstrations,
- end-to-end simulation pipeline walkthroughs.

## Documentation

Additional notes are available in:

- `docs/architecture.md`
- `docs/numerics.md`
