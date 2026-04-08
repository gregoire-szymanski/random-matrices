# Architecture

## Core Design

The package is organized around a clear separation of concerns:

1. `config/`: dataclass schema, YAML/TOML loading, validation.
2. `simulation/`: covariance builders, segment schedule construction, diffusion simulator.
3. `spectral/`: spectral law objects, empirical estimators, MP forward/inverse algorithms.
4. `plotting/`: reusable plotting routines for paths and spectra.
5. `experiments/`: runnable workflows that compose all layers.
6. `cli.py`: command dispatch.

## Data Flow

Typical full-pipeline flow:

1. Load and validate config.
2. Build volatility schedule (constant / piecewise / scaled base).
3. Simulate diffusion path.
4. Build realized covariance and empirical spectral density.
5. Run MP inverse on observed density.
6. Compare estimated population law with reference/integrated population law.
7. Save artifacts and plots.

## Extensibility

- Add new covariance models by extending `build_covariance_matrix`.
- Add inverse algorithms under `spectral/inversion_methods/` and register in `spectral/inverse.py`.
- Add new experiment templates in `experiments/` and wire into CLI.

## Simulation-Centric References

- Config schema and parsing: `src/mpdiff/config/schemas.py`
- Config validation: `src/mpdiff/config/validation.py`
- Covariance and volatility builders: `src/mpdiff/simulation/covariance_builders.py`
- Piecewise schedule policies: `src/mpdiff/simulation/volatility_segments.py`
- Euler simulation core: `src/mpdiff/simulation/diffusion.py`
- Example-driven config guide: `docs/simulation_config_guide.md`
