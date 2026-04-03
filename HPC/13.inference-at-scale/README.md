# Inference at Scale

Training is not the only HPC problem. Large-scale inference also needs careful system design.

## Main Concerns

- throughput
- latency
- batching
- queueing
- memory footprint

## Why Inference Is Different from Training

During inference:

- gradients are not needed
- latency may matter much more
- memory behavior can still be limiting

## Common Approaches

- batch inference jobs
- online serving
- optimized inference engines
- quantized deployment

## Summary

Inference at scale is a systems problem as much as a modeling problem.
