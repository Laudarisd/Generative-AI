# Linux and Shell for AI

Most serious HPC systems are Linux-based, and most interaction happens through the shell.

If you cannot navigate the shell comfortably, HPC work becomes slow and error-prone.

## Core Skills

- connecting with `ssh`
- listing files with `ls`
- changing directories with `cd`
- viewing logs with `cat`, `less`, `tail`
- copying data with `scp` or `rsync`
- monitoring processes

## Why This Matters in AI

Real training workflows often involve:

- remote login
- launching jobs from terminal
- reading logs while jobs run
- copying checkpoints and datasets

## Useful Commands

```bash
pwd
ls -lh
cd project/
tail -f train.log
du -sh checkpoints/
```

## `tmux` and Long Sessions

`tmux` lets you keep a shell session alive even if your terminal disconnects.

This is useful for:

- long preprocessing
- interactive debugging
- monitoring training

## Example Workflow

```bash
ssh user@cluster
tmux new -s train
python train.py
```

If your connection drops, reconnect and run:

```bash
tmux attach -t train
```

## Python Example: Logging to a File

```python
for epoch in range(3):
    print(f"epoch={epoch} loss={1.0 / (epoch + 1):.4f}")
```

Run with shell redirection:

```bash
python train.py > train.log 2>&1
```

## Summary

Linux shell skill is basic infrastructure for HPC work. It is not optional once experiments become remote, long-running, or distributed.
