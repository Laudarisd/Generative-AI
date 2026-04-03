# Positional Encoding

Transformers process tokens in parallel, so they need explicit order information.

## 1. Sinusoidal Encoding

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
```

The idea is that each position gets a deterministic vector. Nearby positions receive related patterns, while different frequencies let the model reason about both short and long distances.

## 2. Learnable Positional Embeddings

Instead of fixed trigonometric patterns, a model can learn position vectors directly.

## 3. Relative Positioning

Some transformers encode relative distance rather than absolute location.

This is useful because in many language tasks, the distance between two tokens matters more than the exact absolute position.

## 4. Why Position Information Is Needed

Without positional information, the token sequence:

```text
dog bites man
```

and:

```text
man bites dog
```

would look like the same bag of tokens to plain self-attention.

## 5. Other Position Methods

Modern transformer systems use several approaches:

- sinusoidal encodings
- learned absolute embeddings
- relative position bias
- rotary positional embeddings (RoPE)
- ALiBi

RoPE is especially common in modern decoder-only LLMs because it integrates position into the attention computation itself.

## 6. Python Example

```python
import numpy as np

d_model = 8
position = 2
encoding = []
for i in range(d_model // 2):
    angle = position / (10000 ** (2 * i / d_model))
    encoding.append(np.sin(angle))
    encoding.append(np.cos(angle))

print(np.array(encoding))
```

## 7. PyTorch Example: Learnable Positional Embeddings

```python
import torch
import torch.nn as nn

max_len = 16
d_model = 32

position_embedding = nn.Embedding(max_len, d_model)
positions = torch.arange(0, 10)
pos_vectors = position_embedding(positions)

print(pos_vectors.shape)
```

## 8. Practical Comparison

| Method | Strength | Limitation |
| --- | --- | --- |
| Sinusoidal | simple, deterministic, no extra learned params | less flexible |
| Learned absolute | easy and expressive | may generalize less beyond trained length |
| Relative bias | captures distance relationships well | more implementation complexity |
| RoPE | strong for modern LLM attention | conceptually less intuitive at first |

## Summary

Positional encoding gives transformers sequence order, which plain self-attention does not know by itself.
