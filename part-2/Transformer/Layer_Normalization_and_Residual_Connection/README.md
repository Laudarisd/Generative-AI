# Layer Normalization and Residual Connection

Residual connections and layer normalization are core stability mechanisms in transformers.

## 1. Residual Connections

Residual update:

```math
y = x + \mathrm{Sublayer}(x)
```

Why it matters:

- better gradient flow
- easier deep optimization
- preserves input signal

## 2. Layer Normalization

LayerNorm normalizes features within a token representation.

It helps stabilize training.

For a token vector $x \in \mathbb{R}^{D}$, LayerNorm computes normalized features using the token's own mean and variance across its hidden dimensions.

## 3. Typical Transformer Block Pattern

A simplified pattern is:

```math
x' = x + \mathrm{Attention}(x)
```

```math
y = x' + \mathrm{FFN}(x')
```

with normalization applied either before or after sublayers depending on architecture style.

## 4. Post-Norm vs Pre-Norm

### Post-Norm

The original transformer paper commonly presented normalization after the residual addition.

### Pre-Norm

Many modern LLMs use pre-norm because it often improves optimization stability for deep networks.

That gives a pattern like:

```math
H' = H + \mathrm{Attention}(\mathrm{LN}(H))
```

```math
H'' = H' + \mathrm{FFN}(\mathrm{LN}(H'))
```

## 5. Why Residuals Matter So Much

Without residual shortcuts, very deep models become much harder to train because useful information and gradients must pass through every transformation without an easy path forward.

Residuals let the model learn:

- "keep this representation mostly unchanged"
- "adjust only what needs changing"

That makes deep stacked blocks far more trainable.

## 6. PyTorch Example

```python
import torch
import torch.nn as nn

x = torch.randn(2, 5, 8)
ln = nn.LayerNorm(8)
out = ln(x)
print(out.shape)
```

## 7. Residual Example

```python
import torch

x = torch.randn(2, 5, 8)
sublayer_out = torch.randn(2, 5, 8)
y = x + sublayer_out

print(y.shape)
```

## 8. Intuition

Attention and FFN layers are the "work" layers.

Residuals and normalization are the "stability and trainability" layers.

Without them, transformer training at scale would be much more brittle.

## Summary

Residuals help information and gradients move through the network, while LayerNorm improves optimization stability.
