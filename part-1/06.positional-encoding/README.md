# Positional Encoding

## Why Positional Encoding Exists

A transformer sees tokens in parallel. That is powerful, but it also means the architecture does not naturally know sequence order.

Without positional information, these two sequences would look too similar:

```text
dog bites man
man bites dog
```

The words are the same, but the meaning is different.

So the model needs a way to inject order.

---

## 1. The Core Problem

Self-attention by itself is permutation-friendly. It compares token representations, but it does not automatically know which token came first.

That means position information must be added or encoded somehow.

This is one of the subtle but essential design ideas in transformers.

---

## 2. Sinusoidal Positional Encoding

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

### Why Sine and Cosine?

These functions provide:

- smooth variation across positions
- multiple frequencies
- a structured way to encode relative offset information

This lets nearby and far positions have related but distinct patterns.

---

## 3. How Positional Information Enters the Model

In the classic transformer, positional encoding is added to token embeddings:

```math
z_i = e_i + p_i
```

where:

- $e_i$ is the token embedding
- $p_i$ is the positional encoding

That means the transformer receives both:

- what token this is
- where it is in the sequence

---

## 4. Learnable Positional Embeddings

Another approach is to learn a position vector for each position directly.

Why people use learnable versions:

- simple
- flexible
- works well in practice

Possible downside:

- extrapolation to much longer unseen sequence lengths may be weaker than with carefully designed relative methods

---

## 5. Relative Positional Methods

Some transformer variants encode relative distance instead of absolute position.

This can help the model reason about:

- nearby dependencies
- longer sequence generalization
- shift-invariant patterns

Relative methods often matter in modern long-context transformer design.

---

## 6. Rotary and Other Modern Variants

Modern systems often use alternatives such as:

- relative bias methods
- rotary position embeddings, or RoPE
- ALiBi-style biasing

These methods try to improve:

- long-context behavior
- extrapolation
- attention quality over distance

You do not need every implementation detail yet, but you should know positional encoding did not stop with the original sine-cosine design.

---

## 7. Practical Intuition

Suppose two sentences use the same words in different orders:

- `the model solved the problem`
- `the problem solved the model`

Token identity alone is not enough. Position changes meaning.

Positional encoding is one of the mechanisms that helps preserve that difference.

---

## 8. Python Example: Sinusoidal Encoding

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

---

## 9. Python Example: Add Position to Token Embeddings

```python
import numpy as np

token_embedding = np.array([0.2, 0.5, 0.1, 0.7])
position_embedding = np.array([0.0, 0.1, 0.0, -0.1])

combined = token_embedding + position_embedding
print(combined)
```

This is the core idea at the input stage of a transformer.

---

## 10. Byte Pair Encoding, or BPE

This topic technically belongs to tokenization rather than positional encoding, but it is worth mentioning here because both are part of the input pipeline before the transformer starts deeper processing.

Byte Pair Encoding, or **BPE**, is a subword tokenization method.

### Core Idea

Start from smaller units, then repeatedly merge frequent symbol pairs.

This gives a vocabulary that:

- handles rare words better than pure word-level tokenization
- keeps sequences shorter than character-level tokenization
- works well across natural language and code

### Tiny Example

Suppose frequent pairs include:

- `l` + `ow` -> `low`
- `low` + `er` -> `lower`

Over time, common subword patterns become vocabulary items.

### Why BPE Matters for LLMs

- reduces vocabulary explosion
- handles unseen words via subword pieces
- gives practical token lengths
- works well for multilingual and code-heavy data

### Simple Python Illustration

```python
from collections import Counter

words = ["low", "lower", "lowest", "newer"]

pairs = Counter()
for word in words:
    chars = list(word)
    for i in range(len(chars) - 1):
        pairs[(chars[i], chars[i + 1])] += 1

print(pairs.most_common(5))
```

This is not a full BPE tokenizer, but it shows the merge intuition.

### Important Clarification

- **BPE** answers: how should text be split into tokens?
- **Positional encoding** answers: how should order be represented once tokens already exist?

They solve different problems.

---

## 11. Why This Matters in LLMs

Without positional information:

- next-token prediction would be much weaker
- syntax would become ambiguous
- long-sequence structure would suffer

Positional encoding is not a cosmetic add-on. It is necessary for sequence meaning.

---

## 12. Chapter Summary

- transformers process tokens in parallel, so order must be injected explicitly
- sinusoidal encodings are the classic solution
- learnable and relative positional methods are common alternatives
- modern transformer variants often use more advanced positional schemes such as RoPE
- BPE is part of tokenization, not positional encoding, but both matter before and at transformer input time
- positional information is essential for syntax, semantics, and sequence structure

## Practice Questions

1. Why does plain self-attention need positional information?
2. Why are sine and cosine useful for positional encoding?
3. What is the practical difference between absolute and relative positional methods?
4. Why might learnable positional embeddings generalize differently from sinusoidal ones?
5. Why is position essential even when token embeddings are already strong?
6. Why is BPE solving a different problem from positional encoding?
