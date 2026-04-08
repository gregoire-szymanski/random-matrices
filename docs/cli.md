# CLI Guide

The `mpdiff` command is the primary interface for running experiments from YAML/TOML configs.

## 1. General Usage

```bash
mpdiff <command> --config <path/to/config.yaml>
```

List available commands:

```bash
mpdiff --help
```

Get command help:

```bash
mpdiff mp-forward --help
```

All commands require:

- `--config`: path to `.yaml`, `.yml`, or `.toml` config.

## 2. Commands

### `simulation`

Run diffusion simulation and realized spectral analysis artifacts.

```bash
mpdiff simulation --config configs/config_constant_isotropic.yaml
```

Main outputs (if enabled):

- path/increment arrays,
- realized covariance and eigenvalues,
- empirical and target spectral density files,
- optional path and spectrum plots,
- metadata with timers.

### `mp-forward`

Run MP forward transform from configured population law.

```bash
mpdiff mp-forward --config configs/config_diagonal_gamma.yaml
```

Main outputs:

- population and MP-forward densities,
- Stieltjes values,
- forward diagnostics metadata,
- optional comparison plot.

### `mp-inverse`

Run synthetic inverse workflow: `H -> MP(H) -> H_hat`.

```bash
mpdiff mp-inverse --config configs/config_diagonal_dirac_mixture.yaml
```

Main outputs:

- per-method recovered population atoms/weights,
- reconstructed observed densities,
- method diagnostics metadata,
- optional reconstruction and method-comparison plots.

### `full-pipeline`

Run full path-driven workflow:

`simulate -> realized covariance -> empirical spectrum -> MP inverse -> compare with reference population law`.

```bash
mpdiff full-pipeline --config configs/config_piecewise_two_segments.yaml
```

Main outputs:

- simulation arrays,
- realized spectral artifacts,
- per-method recovery artifacts,
- method summary CSV/JSON,
- metadata JSON and text report,
- overlay/histogram comparison plots.

### `end-to-end`

Alias for `full-pipeline`.

```bash
mpdiff end-to-end --config configs/config_piecewise_two_segments.yaml
```

Use either form; behavior is the same.

### `inversion-benchmark`

Benchmark inversion methods side-by-side on one config-defined benchmark law.

```bash
mpdiff inversion-benchmark --config configs/config_diagonal_gamma.yaml
```

Main outputs:

- benchmark summary CSV/JSON,
- runtime and quality plots,
- observed/reconstructed density files,
- benchmark metadata.

## 3. Typical Workflows

### Workflow A: First-time sanity check

```bash
mpdiff simulation --config configs/config_constant_isotropic.yaml
mpdiff mp-forward --config configs/config_constant_isotropic.yaml
mpdiff mp-inverse --config configs/config_constant_isotropic.yaml
```

### Workflow B: Full research loop

```bash
mpdiff full-pipeline --config configs/config_low_rank_plus_diag.yaml
mpdiff inversion-benchmark --config configs/config_low_rank_plus_diag.yaml
```

### Workflow C: Compare methods broadly

Set in config:

```yaml
mp_inverse:
  method: optimization
  compare_all_methods: true
```

Then run:

```bash
mpdiff end-to-end --config configs/config_diagonal_gamma.yaml
```

## 4. Config Tips For CLI Runs

- Keep `global.seed` fixed for reproducibility.
- Use unique `global.output_dir` per run to avoid mixed artifacts.
- Turn off heavy outputs while debugging:

```yaml
global:
  save_figures: false
  save_arrays: true
  save_metadata: true
```

- Enable timer logging:

```yaml
benchmark:
  enabled: true
```

## 5. Exit/Failure Behavior

Typical failure sources:

- invalid config keys/values,
- inconsistent piecewise segment definitions,
- unsupported inverse method names,
- numerical settings too aggressive for stable inversion.

When a run fails, start by validating config semantics and then relax numerical settings (`eta`, support bounds, iteration limits).

## 6. Programmatic Equivalents

Every CLI command maps to one runner function:

- `simulation` -> `mpdiff.experiments.run_simulation.run_simulation`
- `mp-forward` -> `mpdiff.experiments.run_mp_forward.run_mp_forward`
- `mp-inverse` -> `mpdiff.experiments.run_mp_inverse.run_mp_inverse`
- `full-pipeline` / `end-to-end` -> `mpdiff.experiments.run_full_pipeline.run_full_pipeline`
- `inversion-benchmark` -> `mpdiff.experiments.run_inversion_benchmark.run_inversion_benchmark`

Example:

```python
from mpdiff.experiments.run_full_pipeline import run_full_pipeline
summary = run_full_pipeline("configs/config_constant_isotropic.yaml")
print(summary)
```
