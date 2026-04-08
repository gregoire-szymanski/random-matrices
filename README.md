# mpdiff

`mpdiff` is a Python 3.11+ numerical research toolkit for:

- high-dimensional diffusion simulation with piecewise-constant volatility,
- realized covariance spectral analysis,
- Marcenko-Pastur (MP) forward transforms,
- approximate MP inverse recovery with multiple competing methods,
- method benchmarking and side-by-side comparison.

The codebase is organized for reproducible experiments: config-driven runs, deterministic seeds, explicit timers, and saved reports.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

Run one end-to-end experiment from config:

```bash
mpdiff end-to-end --config configs/config_constant_isotropic.yaml
```

Run MP inversion benchmarking (single benchmark law, multiple methods):

```bash
mpdiff inversion-benchmark --config configs/config_diagonal_gamma.yaml
```

Minimal start (copy/paste):

```bash
mpdiff simulation --config configs/config_constant_isotropic.yaml
mpdiff end-to-end --config configs/config_constant_isotropic.yaml
```

## CLI Commands

- `mpdiff simulation --config <yaml>`: simulation + realized spectrum outputs.
- `mpdiff mp-forward --config <yaml>`: population law -> MP-forward density.
- `mpdiff mp-inverse --config <yaml>`: synthetic inverse (population -> forward -> inverse).
- `mpdiff full-pipeline --config <yaml>`: simulation -> inverse -> comparison.
- `mpdiff end-to-end --config <yaml>`: alias of `full-pipeline`.
- `mpdiff inversion-benchmark --config <yaml>`: side-by-side inversion benchmark table + plots.

## Example CLI Commands

```bash
mpdiff simulation --config configs/config_rotated_haar.yaml
mpdiff mp-forward --config configs/config_diagonal_gamma.yaml
mpdiff mp-inverse --config configs/config_diagonal_dirac_mixture.yaml
mpdiff full-pipeline --config configs/config_piecewise_two_segments.yaml
mpdiff inversion-benchmark --config configs/config_diagonal_gamma.yaml
```

## Config Catalog

Ready-to-use configs (all runnable, commented):

1. `configs/config_constant_isotropic.yaml`
2. `configs/config_diagonal_dirac_mixture.yaml`
3. `configs/config_diagonal_uniform.yaml`
4. `configs/config_diagonal_gamma.yaml`
5. `configs/config_diagonal_rescaled_beta.yaml`
6. `configs/config_rotated_haar.yaml`
7. `configs/config_low_rank_plus_diag.yaml`
8. `configs/config_piecewise_two_segments.yaml`
9. `configs/config_piecewise_scalar_times_base_shared.yaml`
10. `configs/config_piecewise_scalar_times_base_redrawn.yaml`

## Inversion Modes

Use one method:

```yaml
mp_inverse:
  method: optimization
```

Compare selected methods:

```yaml
mp_inverse:
  method: optimization
  compare_methods: [optimization, fixed_point, stieltjes_based]
```

Compare all registered methods:

```yaml
mp_inverse:
  method: optimization
  compare_all_methods: true
```

## Design Choices

- Config system: one explicit YAML schema for simulation, spectral numerics, inversion, plotting, and outputs.
- Reproducibility: all experiment runners build RNGs from `global.seed`.
- Inversion architecture: interchangeable method interface with config-driven dispatch and optional compare-all mode.
- Benchmarking: dedicated helper/runner provides method-wise runtime and quality tables for the same observed law.
- Output organization: each runner writes strongly-prefixed artifacts (`full_pipeline_*`, `inversion_benchmark_*`, etc.) so mixed runs remain traceable in one directory.

## Numerical Stability Warnings

- MP inverse recovery is ill-posed in finite samples; recovered laws are approximate.
- Small `eta`, narrow support bounds, or low regularization can make inversions unstable.
- Do not rely on one method/result only: compare methods and inspect both population and reconstruction metrics.
- Large method disagreement is a warning signal, not just noise.

## Outputs

Depending on runner and save switches, outputs include:

- arrays: paths, increments, realized covariance, eigenvalues,
- densities: empirical/reference/reconstructed,
- recovered population atoms/weights per method,
- plots: overlays, histograms, reconstruction diagnostics,
- tables: method comparison CSV/JSON,
- reports: metadata JSON + human-readable text report.

For end-to-end runs, key files are:

- `full_pipeline_method_summary.csv`
- `full_pipeline_method_summary.json`
- `full_pipeline_metadata.json`
- `full_pipeline_report.txt`

## Notebooks

- `notebooks/config_examples/`: one notebook per config in the catalog above.
- `notebooks/benchmark_inversion_methods.ipynb`: compares all inversion methods on shared benchmark problems.
- `notebooks/end_to_end_examples/`: model-family focused end-to-end notebooks.

## Documentation

- `docs/project_guide.md` (central entry point)
- `docs/quickstart.md`
- `docs/cli.md`
- `docs/testing.md`
- `docs/config_guide.md`
- `docs/inversion_methods.md`
- `docs/notebooks_guide.md`
- `docs/end_to_end_pipeline.md`
- `docs/numerics.md`
- `docs/architecture.md`

## Testing

```bash
pytest
```

The suite covers config loading, covariance builders, simulation reproducibility, MP forward/inverse behavior, end-to-end outputs, and empirical spectral utilities.
