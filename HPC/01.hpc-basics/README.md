# HPC Basics

High-performance computing, or **HPC**, means using powerful compute systems to solve large or expensive problems efficiently.

In AI, HPC usually means:

- many CPU cores
- one or more GPUs
- large RAM and storage
- fast networking
- shared clusters rather than only local machines

## Why HPC Matters for AI

Modern AI workloads are expensive because they involve:

- large datasets
- big matrix operations
- long training runs
- repeated experiments

Without HPC, many practical workflows would be too slow.

## Core Hardware Terms

- **CPU**: handles general-purpose computation, data loading, orchestration
- **GPU**: accelerates large parallel tensor operations
- **RAM**: system memory used by the CPU side
- **VRAM**: GPU memory used for tensors, activations, and model weights
- **Storage**: disks or network filesystems for datasets and checkpoints
- **Interconnect**: network or internal links that connect nodes and devices

## Typical AI Compute Setups

### Workstation

- one machine
- maybe one or several GPUs
- good for local development

### HPC Cluster

- multiple nodes
- shared scheduler
- shared storage
- batch-job workflow

### Cloud AI Compute

- rented instances
- flexible scaling
- often higher operational cost if unmanaged

## Practical Example

A student training a small CNN on CIFAR-10 may only need a single GPU workstation.

A team pretraining a large language model needs:

- many GPUs
- distributed training
- checkpoint storage
- orchestration and scheduling

## Python Example: Simple Device Check

```python
import torch

print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
```

## Summary

HPC is not just "big computers." It is the full discipline of organizing hardware, software, storage, and scheduling so large workloads become feasible.
