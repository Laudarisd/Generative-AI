# PyTorch on HPC

PyTorch is one of the most common frameworks for AI work on clusters.

## Common PyTorch Tasks on HPC

- single-GPU training
- multi-GPU training
- distributed training with DDP
- checkpointing and resume

## Single-GPU Example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
).cuda()

x = torch.randn(16, 10, device="cuda")
y = model(x)
print(y.shape)
```

## Checkpoint Example

```python
import torch

checkpoint = {"epoch": 3, "model_state": {"dummy": 1}}
torch.save(checkpoint, "checkpoint.pt")
loaded = torch.load("checkpoint.pt")
print(loaded["epoch"])
```

## Summary

PyTorch on HPC is mostly about doing ordinary training correctly under distributed, scheduled, remote conditions.
