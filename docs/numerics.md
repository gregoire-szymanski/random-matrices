# Numerical Notes

## MP Forward Solver

The MP forward transform uses the fixed-point equation for the Stieltjes transform:

`m(z) = ∫ 1 / (t (1 - c - c z m(z)) - z) dH(t)`

Implementation details:

- Damped fixed-point iterations for stability.
- Warm-start continuation along the real grid for faster convergence.
- Density recovery via `f(x) = Im(m(x+iη)) / π`.

Configurable numerical controls include:

- `eta`
- `tol`
- `max_iter`
- `damping`

## MP Inverse Methods

### Optimization

- Parameterizes population law on fixed support with softmax weights.
- Minimizes squared mismatch between observed and forward-predicted densities.
- Optional smoothness regularization on second differences of weights.

### Fixed Point

- Uses a dictionary of single-atom MP kernels.
- Richardson-Lucy style multiplicative updates with optional smoothing.

### Stieltjes-Based

- Estimates Stieltjes transform from observed density.
- Applies pointwise nonlinear shrinkage inversion.

### Moment-Based

- Uses low-order MP moment relations to infer target mean/variance.
- Fits a gamma family approximation for the population spectrum.

## Stability Guidance

- Increase `eta` if fixed-point equations become unstable.
- Increase `max_iter` and/or decrease `tol` for higher accuracy.
- Use moderate regularization in inverse routines to reduce oscillatory weights.
- Validate inverse quality with forward-reconstruction error and moment checks.
