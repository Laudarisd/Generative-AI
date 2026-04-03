# GPU Fundamentals

GPUs are the main compute engines behind modern deep learning.

They are good at large parallel operations such as:

- matrix multiplication
- convolution
- attention
- batched tensor operations

## Why GPUs Beat CPUs for Deep Learning

Deep learning uses many repeated arithmetic operations across large tensors.

GPUs are designed for that style of work:

- massive parallelism
- high memory bandwidth
- specialized tensor hardware

## Key Terms

- **CUDA**: NVIDIA's GPU computing platform
- **VRAM**: memory on the GPU
- **tensor cores**: hardware units for fast mixed-precision math
- **mixed precision**: using lower-precision formats like FP16/BF16 to improve speed and memory use

## Common Bottlenecks

- out-of-memory errors
- small batch sizes causing poor utilization
- slow data loading starving the GPU

## Python Example

```python
import torch

if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print(z.shape, z.device)
```

## Mixed Precision Example

```python
import torch
from torch.cuda.amp import autocast

x = torch.randn(512, 512, device="cuda")
y = torch.randn(512, 512, device="cuda")

with autocast():
    z = x @ y

print(z.dtype)
```

## Summary

Understanding GPU memory, precision, and utilization is essential before trying to scale to clusters or multi-GPU jobs.
