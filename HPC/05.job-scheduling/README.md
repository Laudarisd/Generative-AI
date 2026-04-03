# Job Scheduling

On shared HPC systems, you usually do not run training directly. You submit jobs to a scheduler such as Slurm.

## Why Scheduling Exists

Schedulers manage:

- fairness
- resource allocation
- queueing
- accounting
- reproducibility of launches

## Common Slurm Concepts

- `sbatch`: submit a batch job
- `srun`: launch tasks
- partition or queue
- wall time
- GPU request
- CPU and RAM request

## Example Slurm Script

```bash
#!/bin/bash
#SBATCH --job-name=train-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

python train.py
```

## Interactive vs Batch

### Interactive

Useful for:

- debugging
- quick experiments
- environment checks

### Batch

Useful for:

- long runs
- production experiments
- reproducible jobs

## Job Arrays

Job arrays help run many similar experiments such as hyperparameter sweeps.

## Summary

Schedulers are the traffic system of HPC clusters. If you understand scheduling well, your experiments become cleaner and more reliable.
