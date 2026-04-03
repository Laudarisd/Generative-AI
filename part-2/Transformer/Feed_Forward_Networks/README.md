# Feed Forward Networks

Feed-forward networks are applied independently at each token position after attention.

## 1. Core Formula

A transformer FFN often looks like:

```math
\mathrm{FFN}(x) = W_2 \phi(W_1 x + b_1) + b_2
```

where $\phi$ is an activation such as GELU or ReLU.

## 2. Why FFNs Matter

Attention mixes information across tokens.

The FFN then transforms each token representation nonlinearly.

You can think of the transformer block as having two jobs:

- attention decides which tokens matter
- the FFN decides how to transform each token's feature vector after that context has been collected

## 3. Common Hidden Expansion

The intermediate dimension is often larger than the model dimension.

Example:

- model dimension = 768
- FFN hidden dimension = 3072

This expansion increases representational power without changing the sequence length.

## 4. Position-Wise Means

The FFN is usually applied independently at every token position.

If:

```math
X \in \mathbb{R}^{N \times D}
```

then the same FFN parameters are used for every token vector $x_i \in \mathbb{R}^{D}$.

That is why FFNs are called position-wise feed-forward networks.

## 5. Why Nonlinearity Matters

Without a nonlinear activation, the two linear layers would collapse into one effective linear map. The activation makes the block more expressive.

Modern transformers often use:

- GELU
- SwiGLU
- GEGLU
- ReLU in older architectures

## 6. Gated FFN Variants

Many modern LLMs use gated FFN variants such as SwiGLU:

```math
\mathrm{SwiGLU}(x) = (xW_1) \odot \mathrm{SiLU}(xW_2)
```

These variants often improve quality compared with plain ReLU-style FFNs.

## 7. PyTorch Example

```python
import torch
import torch.nn as nn

ffn = nn.Sequential(
    nn.Linear(8, 32),
    nn.GELU(),
    nn.Linear(32, 8),
)

x = torch.randn(4, 10, 8)
y = ffn(x)
print(y.shape)
```

## 8. PyTorch Example with Explicit Shapes

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 5
d_model = 16
d_ff = 64

x = torch.randn(batch_size, seq_len, d_model)

linear1 = nn.Linear(d_model, d_ff)
act = nn.GELU()
linear2 = nn.Linear(d_ff, d_model)

h = linear1(x)
a = act(h)
y = linear2(a)

print("input:", x.shape)
print("hidden:", h.shape)
print("output:", y.shape)
```

## 9. Practical Interpretation

If attention lets a token gather information like:

```text
"this token should pay attention to the subject, verb, and previous context"
```

then the FFN helps reshape that mixed information into a more useful internal representation for later layers.

## Summary

FFNs are the per-token nonlinear processing blocks that complement attention inside transformers.
