# HPC for Scientific AI

Scientific AI workloads often combine simulation, numerical methods, and learned models.

## Common Workloads

- PINNs
- surrogate modeling
- climate forecasting
- simulation parameter sweeps
- inverse problems

## Why HPC Helps

These tasks often need:

- large grids
- repeated solver runs
- heavy tensor computation
- many experiments

## Example

A PINN for a PDE may require thousands of collocation points and repeated higher-order derivative computation, which makes GPU acceleration and memory planning important.

## Summary

Scientific AI sits naturally inside HPC because both computation and numerical structure are heavy.
