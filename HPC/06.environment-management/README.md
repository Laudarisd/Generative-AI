# Environment Management

Reproducibility in HPC depends heavily on environment control.

If package versions drift, the same code may behave differently across nodes or reruns.

## Common Tools

- `conda`
- `venv`
- environment modules
- containers such as Docker or Apptainer/Singularity

## Why This Matters

AI projects often depend on:

- exact CUDA versions
- PyTorch builds
- compiled extensions
- matching driver support

## Conda Example

```bash
conda create -n ai-hpc python=3.11
conda activate ai-hpc
pip install torch torchvision
```

## Requirements File

```text
torch
transformers
datasets
numpy
```

## Best Practice

Store:

- environment file
- launch script
- important version notes

## Summary

Environment management is not housekeeping. It is part of experiment correctness.
