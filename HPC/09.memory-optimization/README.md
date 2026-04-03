# Memory Optimization

GPU memory is one of the main constraints in deep learning.

## Where Memory Goes

- model weights
- optimizer state
- gradients
- activations
- temporary buffers

## Common Techniques

- mixed precision
- gradient checkpointing
- gradient accumulation
- sharded training
- parameter-efficient fine-tuning

## Why Gradient Checkpointing Helps

Instead of storing every activation, some activations are recomputed during backward pass.

This trades extra compute for lower memory use.

## PyTorch Example

```python
import torch

x = torch.randn(2048, 2048, device="cuda")
print("allocated MB:", torch.cuda.memory_allocated() / 1024**2)
```

## Summary

Memory optimization is often the difference between "model cannot run" and "model fits comfortably."
