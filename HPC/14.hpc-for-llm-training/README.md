# HPC for LLM Training

LLM training pushes HPC systems hard because token counts, model sizes, and optimizer states become enormous.

## Why LLMs Are Special

- long sequence processing
- large embedding tables
- huge parameter counts
- expensive checkpoints
- distributed optimization requirements

## Typical Workflow

1. prepare tokenizer and dataset
2. shard data
3. launch distributed training
4. save periodic checkpoints
5. run evaluation jobs

## Practical Challenges

- memory pressure
- slow checkpoint writes
- synchronization overhead
- token throughput bottlenecks

## Summary

LLM training is one of the clearest reasons HPC knowledge matters in modern AI.
