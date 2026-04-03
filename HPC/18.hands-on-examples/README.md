# Hands-On Examples

This chapter collects practical HPC-style snippets you can reuse.

## Example 1: Simple Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

python train.py
```

## Example 2: Tiny GPU Training Loop

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5):
    x = torch.randn(32, 10, device="cuda")
    y = torch.randn(32, 1, device="cuda")
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    print(step, loss.item())
```

## Example 3: Resume Checkpoint Logic

```python
import os
import torch

path = "checkpoint.pt"

if os.path.exists(path):
    ckpt = torch.load(path)
    print("resuming from epoch", ckpt["epoch"])
else:
    print("starting fresh")
```

## Summary

Hands-on examples make the rest of the HPC section operational rather than only descriptive.
