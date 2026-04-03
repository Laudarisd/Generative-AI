# Parameter Understanding

Modern language models are large because they contain a very large number of trainable parameters. A parameter is usually a learned scalar value inside a weight matrix or bias vector.

## 1. What a Parameter Is

In a linear layer:

```math
y = Wx + b
```

the entries of $W$ and $b$ are parameters.

If:

- $W \in \mathbb{R}^{m \times n}$
- $b \in \mathbb{R}^{m}$

then the parameter count is:

```math
mn + m
```

## 2. Why Parameter Count Matters

Parameter count affects:

- memory usage
- training compute
- inference cost
- storage size
- deployment hardware
- latency and throughput tradeoffs

A larger parameter count usually increases capacity, but it also increases cost.

## 3. Simple Calculation Example

Suppose a feed-forward layer maps 4096 features to 11008 features.

```math
W_1 \in \mathbb{R}^{11008 \times 4096}
```

The parameter count is:

```math
11008 \times 4096 = 45{,}088{,}768
```

That is already about 45 million parameters in just one matrix.

If the block also has a second projection back to 4096, then another matrix of roughly the same scale is added.

## 4. Where Parameters Live in a Transformer

A transformer stores parameters in:

- token embeddings
- attention projections $W^Q, W^K, W^V, W^O$
- feed-forward layers
- layer normalization parameters
- output head

In modern LLMs, the feed-forward layers usually hold a large fraction of the total parameters.

## 5. Embedding Layer Example

If a model has:

- vocabulary size = 50,000
- hidden dimension = 4096

then the embedding table has:

```math
50{,}000 \times 4096 = 204{,}800{,}000
```

about 205 million parameters.

## 6. Attention Layer Parameter Count

If hidden size is $d = 4096$, each dense projection from $d$ to $d$ has:

```math
4096 \times 4096 = 16{,}777{,}216
```

parameters.

A standard attention block has four major dense matrices:

- query projection
- key projection
- value projection
- output projection

So attention alone contributes roughly:

```math
4 \times 16{,}777{,}216 \approx 67 \text{ million}
```

before biases and implementation details.

## 7. Feed-Forward Layer Count

For a transformer MLP with:

- model dimension = 4096
- expansion dimension = 11008

we often have two large matrices:

```math
4096 \times 11008
```

and:

```math
11008 \times 4096
```

Together this is about 90 million parameters, which is why MLP blocks are often heavier than attention blocks.

## 8. Why Models Become Heavy

A model becomes heavy because parameter count interacts with several costs at once.

### Memory Cost

If weights are stored in FP16, each parameter uses 2 bytes.

Approximate memory for weights only:

```math
\text{memory} \approx \text{parameters} \times 2 \text{ bytes}
```

Examples:

- 7B parameters -> about 14 GB for weights in FP16
- 13B parameters -> about 26 GB
- 70B parameters -> about 140 GB

This is only the weight memory. Training needs much more because gradients, optimizer states, and activations also consume memory.

## 9. Why Training Is More Expensive Than Inference

During training, the system stores:

- model weights
- activations for backpropagation
- gradients
- optimizer states such as Adam moments

Adam usually stores two extra tensors per parameter, so the effective memory footprint is much larger than just the raw weights.

## 10. Public Real-World Examples

Some well-known public parameter counts include:

| Model | Publicly Known Parameter Count |
| --- | --- |
| GPT-2 XL | 1.5B |
| GPT-3 | 175B |
| Llama 2 7B | 7B |
| Llama 2 13B | 13B |
| Llama 2 70B | 70B |
| Llama 3 8B | 8B |
| Llama 3 70B | 70B |
| Mistral 7B | 7B |
| Mixtral 8x7B | sparse MoE with 8 experts |

For some closed models, such as newer GPT or Claude families, exact parameter counts are often not publicly disclosed.

## 11. Why Closed Models Can Still Feel Bigger

Even if exact parameter counts are hidden, model capability depends on more than just one number:

- architecture improvements
- data quality
- context length engineering
- fine-tuning and alignment
- mixture-of-experts routing
- inference stack optimization

So a model with fewer active parameters can sometimes outperform an older denser model.

## 12. Dense Models vs Mixture-of-Experts

### Dense Model

Every token passes through the same parameters.

### Mixture-of-Experts

Only a subset of expert layers is active for each token.

This allows very large total parameter counts while keeping active compute lower than a fully dense model of the same total size.

## 13. Why Parameter Count Is Not the Whole Story

Two models with the same parameter count can behave very differently because of:

- tokenizer quality
- training tokens
- data filtering
- architecture
- optimization schedule
- context window
- fine-tuning quality

## 14. Python Example: Counting Parameters

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("total parameters:", total)
print("trainable parameters:", trainable)
```

## 15. Python Example: Estimate Weight Memory

```python
def estimate_memory_gb(num_params, bytes_per_param=2):
    return num_params * bytes_per_param / (1024 ** 3)

for params in [7_000_000_000, 13_000_000_000, 70_000_000_000]:
    print(params, "->", round(estimate_memory_gb(params), 2), "GB")
```

## 16. Practical Rule of Thumb

When someone says a model is "big," ask three separate questions:

1. how many total parameters does it have?
2. how many active parameters are used per token?
3. what precision and serving strategy are being used?

That is much more informative than the raw headline number alone.

## Summary

Parameter count is the first bridge between model theory and systems engineering. It explains why modern LLMs require large GPUs, distributed training, quantization, and efficient serving engines. Understanding parameter count also helps you reason about why some models are cheap to experiment with and others require cluster-scale infrastructure.
