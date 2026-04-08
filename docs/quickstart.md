# Quickstart

This is the fastest way to run `mpdiff` end-to-end.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 2. Run one complete experiment

```bash
mpdiff end-to-end --config configs/config_constant_isotropic.yaml
```

This produces arrays, summary tables, metadata, and plots under:

- `outputs/config_constant_isotropic/`

## 3. Inspect results quickly

```bash
ls outputs/config_constant_isotropic
```

Key files to open first:

- `full_pipeline_method_summary.csv`
- `full_pipeline_metadata.json`
- `full_pipeline_overlay_population_empirical_recovered.png`
- `full_pipeline_overlay_observed_reconstructed_forward.png`

## 4. Run one inversion benchmark

```bash
mpdiff inversion-benchmark --config configs/config_diagonal_gamma.yaml
```

Check:

- `outputs/config_diagonal_gamma/inversion_benchmark_summary.csv`
- `outputs/config_diagonal_gamma/inversion_benchmark_runtime.png`

## 5. Open a notebook

```bash
jupyter lab
```

Recommended first notebook:

- `notebooks/config_examples/01_config_constant_isotropic.ipynb`

## 6. Switch inversion methods

Edit your config:

```yaml
mp_inverse:
  method: optimization
  compare_methods: [optimization, fixed_point, moment_based]
```

Then rerun `mpdiff end-to-end --config <your_config.yaml>`.
