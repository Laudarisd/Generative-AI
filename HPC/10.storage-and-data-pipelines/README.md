# Storage and Data Pipelines

HPC performance depends heavily on how data is stored and delivered.

## Common Storage Types

- local disk
- network shared filesystem
- scratch storage
- object storage in cloud environments

## Why Data Pipelines Matter

Even powerful GPUs sit idle if:

- files are slow to read
- preprocessing is too expensive
- the dataloader cannot keep up

## Best Practices

- stage large datasets to fast storage
- avoid millions of tiny file reads when possible
- shard datasets for scalable access
- checkpoint carefully and consistently

## Python Example

```python
from pathlib import Path

data_dir = Path("data")
files = list(data_dir.glob("*.json"))
print("file_count:", len(files))
```

## Summary

Data movement is part of compute. Fast models still lose badly if data pipelines are poor.
