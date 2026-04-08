# Testing Guide

This guide explains how `mpdiff` tests are structured and how to debug numerical failures.

## 1. Testing Philosophy

`mpdiff` combines numerical methods and stochastic simulation, so tests are designed around:

- deterministic reproducibility (fixed seeds),
- mathematically meaningful invariants (non-negativity, normalization, moment sanity),
- approximation-aware tolerances (not exact symbolic equality),
- output contract checks for experiment runners.

The suite intentionally validates both:

- local units (e.g., covariance construction, MP fixed-point solver behavior),
- lightweight integration paths (e.g., runner writes expected output files).

## 2. Test Layout

- `tests/test_config_loader.py`: config parser/validation behaviors and invalid config detection.
- `tests/test_config_catalog_files.py`: all top-level catalog configs load.
- `tests/test_simulation_example_configs.py`: simulation examples load.
- `tests/test_spectral_config_examples.py`: spectral examples load.
- `tests/test_covariance_builders.py`: covariance model outputs and PSD/sqrt handling.
- `tests/test_volatility_schedule.py`: segment mode logic and integrated covariance consistency.
- `tests/test_simulation.py`: simulation shape/reproducibility checks.
- `tests/test_empirical_spectral.py`: realized covariance and empirical density helpers.
- `tests/test_mp_forward.py`: MP forward sanity against closed-form/known constraints.
- `tests/test_mp_inverse.py`: inverse method recovery and method-comparison behavior.
- `tests/test_inverse_method_resolution.py`: inversion method switch resolution.
- `tests/test_inversion_benchmark_helpers.py`: benchmark helper correctness.
- `tests/test_run_simulation_runner.py`: simulation runner output files.
- `tests/test_run_full_pipeline_runner.py`: full pipeline output files.
- `tests/test_cli_parser.py`: CLI command registration.
- `tests/test_notebook_references.py`: notebook references to config files remain valid.

## 3. Running Tests

Run all tests:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_mp_inverse.py
```

Run a single test function:

```bash
pytest tests/test_mp_forward.py::test_mp_forward_dirac_matches_closed_form_reasonably
```

Run tests matching a keyword:

```bash
pytest -k "inverse"
```

Verbose mode:

```bash
pytest -v
```

Stop on first failure:

```bash
pytest -x
```

Re-run only failed tests from last run:

```bash
pytest --lf
```

## 4. Coverage (Optional)

If you want coverage reports:

```bash
pip install pytest-cov
pytest --cov=mpdiff --cov-report=term-missing
```

You can generate HTML coverage:

```bash
pytest --cov=mpdiff --cov-report=html
```

Then open `htmlcov/index.html`.

## 5. Expected Runtime

Typical local runtime is short (often under ~30 seconds), but depends on:

- BLAS/LAPACK backend,
- CPU and vectorization,
- first-run matplotlib/font cache initialization.

## 6. Numerical Tolerances: What "Pass" Means

A passing numerical test usually means:

- approximation error is within a tolerance band,
- inferred/recovered moments are reasonably close,
- densities are finite and nonnegative,
- runner artifacts are complete and structurally valid.

It does **not** imply exact reconstruction of population laws in finite samples.

Important reasons:

- MP transforms are solved numerically on finite grids,
- inverse recovery is regularized and ill-posed,
- simulation noise and finite dimensions introduce variability.

## 7. Common Failure Modes and Debugging

### A. MP forward/inverse test failure

Check first:

- `eta`, `tol`, `max_iter`, `damping` in configs used by tests,
- support bounds for inverse (`support_min`, `support_max`),
- whether a method was switched to a weaker/faster setting.

Debug approach:

1. Re-run the failing test in verbose mode.
2. Inspect diagnostics written by methods (iterations, residual behavior).
3. Temporarily increase `max_iter` and slightly increase `eta`.
4. Re-check with fixed seed.

### B. Config test failure

Likely causes:

- invalid new YAML key names,
- `compare_all_methods` used together with non-empty `compare_methods`,
- broken piecewise segment continuity.

### C. Runner output-contract failure

Likely causes:

- output filename changes without test updates,
- `save_arrays`/`save_metadata` toggles changed in test fixture configs.

## 8. Adding New Tests

Recommended pattern:

1. Keep tests deterministic (`seed` fixed).
2. Use small dimensions and small grids for speed.
3. Assert robust quantities, not fragile exact values.
4. For new runners, add output contract tests similar to existing runner tests.
5. For new inverse methods, add:
   - dispatch/registration test,
   - basic numerical sanity test,
   - comparison test through `compare_inverse_methods`.

Minimal template:

```python
from mpdiff.spectral.transforms import compute_mp_forward


def test_new_behavior() -> None:
    # Arrange
    ...
    # Act
    result = compute_mp_forward(...)
    # Assert
    assert ...
```

## 9. CI/Local Practical Advice

- Run `pytest -q` before committing.
- Run targeted tests (`-k`, single file) while iterating.
- When touching configs/notebooks, also run:

```bash
pytest tests/test_config_catalog_files.py tests/test_notebook_references.py
```

This catches most documentation/config drift quickly.
