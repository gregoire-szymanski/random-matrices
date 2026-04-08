# mpdiff

`mpdiff` provides production-quality simulation tooling for high-dimensional diffusions with piecewise-constant volatility and configurable covariance models.

This README focuses on the simulation and configuration system.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run a Simulation

Use the CLI simulation runner:

```bash
mpdiff simulation --config configs/simulation_examples/01_constant_isotropic.yaml
```

or

```bash
python -m mpdiff.cli simulation --config configs/simulation_examples/01_constant_isotropic.yaml
```

## Simulation Model

The Euler scheme is:

`X_{t_{i+1}} = X_{t_i} + b_i * dt + sigma_i * sqrt(dt) * Z_i`, with `Z_i ~ N(0, I_d)`.

- `sigma_i` is constant on each piecewise segment from config.
- Drift supports `zero`, `constant`, `linear_mean_reversion`, `time_sine`, and optional imported `callable`.

## YAML Config Structure

Top-level sections:

- `global`: seed, output dir, logging, save switches.
- `simulation`: `d`, `T`, `n_steps`, `initial_state`, `drift_model`, optional `time_grid`.
- `volatility`: mode and covariance model definitions.
- `plotting`: simulation plotting controls.
- `benchmark`: timer logging switch.

Volatility modes:

1. `constant`
2. `piecewise`
3. `piecewise_scaled_base` (segment law `Cov_j = c_j * M_j`)

Covariance model kinds:

1. `diag_scalar`: `Cov = lambda * I_d`
2. `diag_distribution`: diagonal eigenvalues from one law
3. `orthogonal_diag`: `Cov = U diag(lambda) U^T`
4. `low_rank_plus_diag`: `Cov = B Sigma B^T + D`

Supported diagonal/eigenvalue distributions:

- `dirac`
- `dirac_mixture`
- `uniform`
- `gamma`
- `rescaled_beta` (implemented as `beta_scale * Beta(alpha, beta) + beta_shift`)

## Example Configs

All examples are in `configs/simulation_examples/` and include inline YAML comments:

1. `01_constant_isotropic.yaml`
2. `02_constant_diag_gamma.yaml`
3. `03_constant_rotated_haar.yaml`
4. `04_low_rank_plus_diag.yaml`
5. `05_piecewise_two_segments.yaml`
6. `06_piecewise_scaled_base.yaml`

## Produced Output Files

For one simulation run, outputs are written to `global.output_dir`.

Arrays:

- `times.npy`
- `paths.npy`
- `increments.npy`
- `segment_indices.npy`
- `realized_covariance.npy`
- `realized_eigenvalues.npy`
- `target_integrated_covariance.npy`
- `target_population_eigenvalues.npy`
- `realized_density.npz`
- `target_density.npz`

Metadata:

- `metadata.json` (config path, timers, drift info, key summary statistics)

Figures (if enabled):

- `paths_sample.png`
- `eigen_histogram.png`
- `eigen_density_comparison.png`

## Notebooks

One notebook per simulation example is provided under `notebooks/simulation_examples/`:

- `01_constant_isotropic.ipynb`
- `02_constant_diag_gamma.ipynb`
- `03_constant_rotated_haar.ipynb`
- `04_low_rank_plus_diag.ipynb`
- `05_piecewise_two_segments.ipynb`
- `06_piecewise_scaled_base.ipynb`

Each notebook:

- runs the corresponding config,
- plots sample coordinates,
- compares realized and target population eigenvalue histograms.

## Additional Docs

- `docs/simulation_config_guide.md`
- `docs/architecture.md`
- `docs/numerics.md`

## Tests

```bash
pytest
```
