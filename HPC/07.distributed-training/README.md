# Distributed Training

Distributed training is how large models are trained across multiple GPUs or nodes.

## Why Distribution Is Needed

Single-device training becomes limiting when:

- the model is too large for one GPU
- the batch size needs to be larger
- training time is too long

## Main Parallelism Types

### Data Parallelism

Each device gets a different mini-batch and a copy of the model.

Gradients are synchronized across devices.

### Model Parallelism

Different parts of the model are placed on different devices.

### Tensor Parallelism

Individual large tensor operations are split across devices.

### Pipeline Parallelism

Different layers or blocks run on different devices in stages.

## PyTorch DDP Example

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = torch.nn.Linear(10, 2).cuda()
model = DDP(model, device_ids=[torch.cuda.current_device()])
```

## Practical Concepts

- synchronization cost
- communication overhead
- global batch size
- sharded optimizer states

## Summary

Distributed training is not just "more GPUs." It is a tradeoff between memory, communication, and throughput.
