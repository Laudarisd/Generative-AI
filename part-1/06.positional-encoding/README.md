# Positional Encoding

## Why Positional Encoding Exists

A transformer sees tokens in parallel. That is powerful, but it also means the architecture does not naturally know sequence order.

Without positional information, these two sequences would look too similar:

```text
dog bites man
man bites dog
```

The words are the same, but the meaning is different.

---

## 1. Sinusoidal Positional Encoding

One classic approach is:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)
```

where:

- $pos$ is the token position
- $i$ is the embedding dimension index
- $d_{model}$ is the hidden size

---

## 2. Learnable Positional Embeddings

Another approach is to learn a position vector for each position directly.

Why people use learnable versions:

- simple
- flexible
- works well in practice

---

## 3. Relative Positional Methods

Some transformer variants encode relative distance instead of absolute position.

This can help the model reason about:

- nearby dependencies
- longer sequence generalization

---

## 4. Python Example

```python
import numpy as np

d_model = 6
position = 3

encoding = []
for i in range(d_model // 2):
    angle = position / (10000 ** (2 * i / d_model))
    encoding.append(np.sin(angle))
    encoding.append(np.cos(angle))

print(np.array(encoding))
```

This is a tiny version of sinusoidal positional encoding.
