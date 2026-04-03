# Training in the Cloud

Cloud training platforms provide managed infrastructure for storing data, launching training jobs, tracking artifacts, and deploying models.

## 1. Why Teams Use the Cloud

Cloud platforms help with:

- GPU access
- distributed training
- artifact storage
- experiment tracking
- scalable deployment
- team collaboration

## 2. Main Cloud Platforms

### AWS

Common services include:

- Amazon S3 for storage
- Amazon SageMaker for training, pipelines, and deployment
- EC2 for custom GPU machines
- ECR for containers

### Google Cloud

Common services include:

- Google Cloud Storage
- Vertex AI for training and managed ML workflows
- Compute Engine for custom infrastructure
- BigQuery for data workflows

### Microsoft Azure

Common services include:

- Azure Blob Storage
- Azure Machine Learning
- Azure compute clusters
- model registry and endpoint services

## 3. Common Workflow

1. upload dataset to cloud storage
2. prepare training container or notebook environment
3. launch training job on GPUs
4. monitor logs and metrics
5. save checkpoints and artifacts
6. register model
7. deploy endpoint or batch inference job

## 4. Example: Why Storage Matters

Training usually reads:

- tokenized data
- checkpoints
- configs
- logs

If storage throughput is poor, expensive GPUs can sit idle waiting for data.

## 5. Managed Platform vs Raw GPU VM

### Managed Platform

Advantages:

- easier orchestration
- cleaner job lifecycle
- integrated model registry
- built-in pipelines

### Raw GPU VM

Advantages:

- maximum flexibility
- custom environment control
- sometimes lower platform overhead

## 6. SageMaker-Style Workflow

A typical SageMaker flow uses:

- S3 dataset storage
- a training script
- an estimator or job spec
- output artifact storage
- endpoint deployment if needed

### Example Script Pattern

```python
# train.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

print("training for", args.epochs, "epochs")
```

A managed service can pass hyperparameters into the script and capture artifacts.

## 7. Vertex AI Style Thinking

You usually define:

- training container
- machine type
- accelerator type
- input data path
- output model path

This is useful when you want a managed Google Cloud workflow rather than manually controlling every VM step yourself.

## 8. Azure ML Style Thinking

You usually define:

- workspace
- compute cluster
- environment
- data asset
- job spec
- registered model or endpoint

Azure ML is especially strong when the rest of the organization already uses Azure infrastructure and governance.

## 9. Cloud Cost Drivers

Major cost drivers include:

- GPU type
- training duration
- storage volume
- data transfer
- endpoint uptime
- failed experiments

## 10. Minimal Training Config Example

```python
training_job = {
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "instance_type": "A10-or-better",
}

print(training_job)
```

## 11. Best Practices

- preprocess data before expensive runs
- checkpoint often
- monitor GPU utilization
- log configs and dataset versions
- separate development from production workloads
- shut down idle endpoints
- pin package versions

## 12. Practical Advice

Start with the simplest platform that matches your team's needs.

Use managed platforms when:

- multiple people share the workflow
- reproducibility matters
- deployment pipelines matter

Use raw infrastructure when:

- you need custom low-level control
- managed abstractions get in the way

## Summary

Cloud training is not only about renting GPUs. It is about managing the full lifecycle of data, jobs, checkpoints, models, and endpoints. AWS, Vertex AI, and Azure ML all support that lifecycle, but they differ in how much is managed for you.
