# Performance Optimization

Performance optimization is about finishing more work with the same hardware.

## Typical Bottlenecks

- low GPU utilization
- dataloader stalls
- poor batch size choices
- slow storage reads
- excessive synchronization

## What To Measure

- GPU utilization
- throughput
- examples per second
- tokens per second
- step time

## Common Levers

- increase batch size if memory allows
- improve dataloader workers
- use mixed precision
- reduce CPU-side bottlenecks
- avoid tiny kernels and excessive overhead

## Python Example

```python
import time

start = time.time()
for _ in range(1000000):
    pass
print("elapsed:", time.time() - start)
```

This is trivial, but timing discipline matters in HPC optimization.

## Summary

Performance tuning is not a final polish step. It often determines whether an experiment is feasible at all.
