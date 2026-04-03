# Monitoring and Debugging

Long jobs fail in many ways. Monitoring is how you catch problems early instead of hours later.

## What To Watch

- GPU utilization
- VRAM usage
- CPU usage
- step time
- training loss
- validation loss
- disk usage

## Common Tools

- `nvidia-smi`
- scheduler job status commands
- log files
- experiment trackers

## `nvidia-smi` Example

```bash
nvidia-smi
```

## Training Curves

The existing file [training_validation_loss_plot.png](../training_validation_loss_plot.png) is a simple example of the kind of monitoring artifact you want to preserve.

## Debugging Questions

- did the job actually start on the requested GPU?
- is the dataloader stuck?
- is loss exploding?
- did the filesystem fill up?

## Summary

Monitoring is how HPC work becomes operational instead of guesswork.
