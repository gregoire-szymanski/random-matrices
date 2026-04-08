# mpdiff

`mpdiff` is a Python 3.11+ project for:

- simulating high-dimensional diffusions with piecewise-constant volatility,
- computing realized covariance spectra,
- applying the Marcenko-Pastur (MP) forward transform,
- approximately inverting the MP map with multiple numerical methods.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## CLI

```bash
mpdiff simulation --config configs/simulation_examples/01_constant_isotropic.yaml
mpdiff mp-forward --config configs/spectral_examples/01_dirac_forward_inverse.yaml
mpdiff mp-inverse --config configs/spectral_examples/02_atomic_mixture_compare_methods.yaml
mpdiff full-pipeline --config configs/simulation_examples/05_piecewise_two_segments.yaml
```

## Project Layout

- `src/mpdiff/config/`: dataclass schemas, YAML/TOML loading, validation
- `src/mpdiff/simulation/`: covariance builders, segment schedules, Euler simulation
- `src/mpdiff/spectral/`: spectral laws, MP forward transform, inverse methods, metrics
- `src/mpdiff/plotting/`: density/path/diagnostics plotting utilities
- `src/mpdiff/experiments/`: runners for simulation, forward MP, inverse MP, and full pipeline
- `configs/`: ready-to-run YAML configs
- `notebooks/`: exploratory examples
- `tests/`: unit and integration-style numerical tests

## Spectral Law Inputs

Population spectral laws can be represented as:

- parametric laws (`dirac`, `dirac_mixture`, `uniform`, `gamma`, `rescaled_beta`),
- discrete atomic laws,
- sampled grid densities,
- empirical discrete laws from sampled eigenvalues.

The core objects are:

- `DiscreteSpectrum`
- `GridDensity`
- `ParametricSpectrumLaw`

All can be converted to a discrete approximation used by MP numerics.

## MP Forward Map

Notation in code and docs: `c = d / n` (aspect ratio).

The forward solver uses the Silverstein fixed-point equation for the Stieltjes transform
`m_F(z)` with `z = x + i*eta`, then recovers density as:

`f_F(x) = Im(m_F(x+i*eta)) / pi`.

Implementation details:

- damped fixed-point iterations,
- continuation warm start along the real grid,
- optional Newton fallback for difficult points,
- diagnostics: convergence rates, residuals, iteration counts, fallback usage.

## MP Inverse Methods

Available methods (switch with `mp_inverse.method`):

- `fixed_point`: Richardson-Lucy style deconvolution on MP Dirac kernels,
- `optimization`: minimize forward mismatch over discretized population measure,
- `stieltjes_based`: nonlinear shrinkage using estimated observed Stieltjes transform,
- `moment_based`: low-order moment inversion with gamma-family approximation.

Run several methods on the same input with `mp_inverse.compare_methods`.

## Key Config Sections

- `mp_forward`: `aspect_ratio`, grid bounds, `eta`, `tol`, `max_iter`, `damping`
- `mp_inverse`: method choice, support grid settings, regularization, tolerances
- `mp_inverse.fixed_point|optimization|stieltjes_based|moment_based`: method-specific options
- `analysis.population_spectrum`: optional direct spectral-law input for MP experiments

If `analysis.population_spectrum` is omitted, runners derive a reference population law from covariance-model configs.

## Example Spectral Configs

- `configs/spectral_examples/01_dirac_forward_inverse.yaml`
- `configs/spectral_examples/02_atomic_mixture_compare_methods.yaml`
- `configs/spectral_examples/03_gamma_population.yaml`
- `configs/spectral_examples/04_rescaled_beta_population.yaml`
- `configs/spectral_examples/05_empirical_discrete_inline.yaml`
- `configs/spectral_examples/06_grid_density_input.yaml`

Each YAML file includes inline comments.

## Spectral Notebooks

- `notebooks/01_mp_forward_inverse_demo.ipynb`
- `notebooks/spectral_examples/*.ipynb` (one notebook per spectral example config)

## Outputs

Runners save arrays, plots, and metadata JSON under `global.output_dir`, including:

- population/observed/reconstructed densities,
- estimated population atoms/weights (per inverse method),
- diagnostics and discrepancy metrics (L1/L2/Wasserstein/support/moments),
- comparison plots and optional convergence plots.

## Docs

- `docs/architecture.md`
- `docs/simulation_config_guide.md`
- `docs/spectral_config_guide.md`
- `docs/numerics.md`

## Tests

```bash
pytest
```
