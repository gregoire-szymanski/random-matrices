# Notebooks Guide

This guide explains the notebook assets included with `mpdiff` and how to use them productively.

## 1. Notebook Sets

### 1.1 Config Example Notebooks

Directory: `notebooks/config_examples/`

There is one notebook for each config file in `configs/config_*.yaml`:

- `01_config_constant_isotropic.ipynb`
- `02_config_diagonal_dirac_mixture.ipynb`
- `03_config_diagonal_uniform.ipynb`
- `04_config_diagonal_gamma.ipynb`
- `05_config_diagonal_rescaled_beta.ipynb`
- `06_config_rotated_haar.ipynb`
- `07_config_low_rank_plus_diag.ipynb`
- `08_config_piecewise_two_segments.ipynb`
- `09_config_piecewise_scalar_times_base_shared.ipynb`
- `10_config_piecewise_scalar_times_base_redrawn.ipynb`

Each notebook:

- summarizes the model intent,
- loads the matching YAML config,
- runs end-to-end pipeline,
- displays summary tables and saved figures,
- provides brief interpretation guidance.

### 1.2 Inversion Comparison Notebook

File: `notebooks/benchmark_inversion_methods.ipynb`

This notebook:

- defines several benchmark spectral laws,
- runs all inversion methods,
- compares runtime and reconstruction quality,
- shows combined tables and plots.

### 1.3 Legacy / Additional Notebooks

Other folders (`notebooks/simulation_examples/`, `notebooks/spectral_examples/`, `notebooks/end_to_end_examples/`) remain available for targeted workflows.

## 2. Running Notebooks

From project root:

```bash
jupyter lab
```

Recommended environment:

- use the same virtual environment as CLI runs,
- install dev dependencies with `pip install -e .[dev]`.
- run the first code cell in each notebook before other cells: it resolves `PROJECT_ROOT`
  and injects `src/` into `sys.path` so `mpdiff` imports work even if the kernel starts in a subfolder.

## 3. Reproducibility Tips

- Keep `global.seed` fixed in YAML.
- Keep output directories distinct per notebook/config to avoid accidental file reuse.
- Re-run all cells after changing config values.

## 4. Typical Workflow in a Config Notebook

1. Load config.
2. Run `run_end_to_end(config_path)`.
3. Inspect method summary CSV/JSON.
4. Inspect overlay/histogram plots.
5. Interpret discrepancy metrics (Wasserstein/L2/runtime).

## 5. Typical Workflow in the Comparison Notebook

1. Build benchmark population laws.
2. Generate observed laws using MP forward transform.
3. Run all inversion methods with shared settings.
4. Aggregate quality and runtime metrics.
5. Visualize trade-offs (accuracy vs runtime).

## 6. Common Pitfalls in Notebook Usage

- Forgetting to rerun after config edits.
- Comparing metrics from different grids without harmonization.
- Overfitting conclusions from one random seed.
- Ignoring runtime-quality tradeoffs when choosing default method.

## 7. Suggested Extensions

- Add additional benchmark laws (heavy tails, multimodal mixtures).
- Add repeated-seed Monte Carlo loops for confidence intervals.
- Export notebook summaries to CSV for external analysis.
