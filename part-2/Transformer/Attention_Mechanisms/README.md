# Attention Mechanisms

Attention is the core mechanism that made transformers powerful.

## 1. Core Idea

Instead of compressing an entire sequence into one fixed vector, attention lets each token look at other tokens directly and decide how much each one matters.

If the word "it" appears in a sentence, attention can learn to focus on the earlier noun it refers to. That is the key intuition: dynamic relevance instead of fixed compression.

## 2. Scaled Dot-Product Attention

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

where:

- $Q$ = queries
- $K$ = keys
- $V$ = values
- $d_k$ = key dimension

### What the formula means

1. compare each query to every key using a dot product
2. divide by $\sqrt{d_k}$ for numerical stability
3. apply softmax so the scores become normalized weights
4. use those weights to combine the value vectors

This means every output token becomes a weighted mixture of information from other tokens.

## 3. Why Scaling by $\sqrt{d_k}$ Matters

Without scaling, dot products can become too large, causing softmax to become overly sharp and gradients to become less stable.

## 4. Multi-Head Attention

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W^O
```

with:

```math
head_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

Each head learns a different projection of the input. One head may focus on local syntax, another on long-range coreference, and another on positional patterns.

## 5. Self-Attention vs Cross-Attention

### Self-Attention

The model attends within the same sequence.

### Cross-Attention

The decoder attends to encoder outputs.

### Causal Self-Attention

In decoder-only models, attention is masked so token $t$ can only see tokens up to position $t$.

This enforces autoregressive generation.

## 6. Worked Intuition Example

Suppose the sentence is:

```text
The cat sat quietly
```

If the token "sat" attends to all tokens, it may assign high weight to:

- "cat" because it is the subject
- "quietly" because it modifies the action

That produces a contextual representation of "sat" that already contains information from related words.

## 7. Attention Matrix Interpretation

If the sequence length is $N$, then the attention weights form an $N \times N$ matrix:

```math
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
```

where each row sums to 1.

Row $i$ tells you how token $i$ distributes its attention across all tokens.

## 8. Tiny PyTorch Example

```python
import torch
import torch.nn.functional as F

Q = torch.randn(3, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 4)

scores = Q @ K.T / (Q.shape[-1] ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V

print("scores shape:", scores.shape)
print("weights row sums:", weights.sum(dim=-1))
print(out.shape)
```

## 9. PyTorch Example with Causal Mask

```python
import torch
import torch.nn.functional as F

seq_len = 4
d = 8
x = torch.randn(seq_len, d)

scores = x @ x.T / (d ** 0.5)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(scores, dim=-1)

print(weights)
```

This mask prevents each token from looking into the future.

## 10. Why Attention Changed NLP

- better long-range interaction than many RNN-based systems
- parallel training over sequence positions
- dynamic context mixing instead of fixed windows
- natural extension to cross-modal and cross-sequence settings

## 11. Costs and Limitations

Standard full attention has quadratic cost in sequence length:

```math
\mathcal{O}(N^2)
```

This becomes expensive for very long contexts. That is why efficient attention methods matter.

## Summary

Attention is the mechanism that lets transformers compare, weight, and combine token information dynamically.
