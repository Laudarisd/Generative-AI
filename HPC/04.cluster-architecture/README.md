# Cluster Architecture

An HPC cluster is not one giant computer. It is a coordinated system of machines with different roles.

## Typical Components

- **login node**: where users connect, edit scripts, and submit jobs
- **compute node**: where heavy jobs actually run
- **head/controller node**: coordinates scheduling and cluster services
- **shared storage**: common filesystem accessible from multiple nodes

## Why This Separation Exists

Clusters keep interactive administration separate from heavy computation so:

- the system stays organized
- users do not overload shared entry points
- compute resources can be scheduled fairly

## Node Terminology

Each compute node may have:

- CPUs
- RAM
- one or more GPUs
- local scratch disk

## Practical Rule

Do not run heavy training on the login node.

Use the scheduler to request a compute node.

## Example Architecture

- login node: lightweight shell access
- 8 GPU nodes: each with 4 GPUs
- shared `/home` and `/scratch`
- Slurm scheduler

## Summary

Cluster architecture matters because your workflow depends on where code runs, where data lives, and how compute is assigned.
