# Transformer Overview

Transformers are the dominant architecture family behind modern LLMs, many vision models, multimodal systems, and a large share of state-of-the-art sequence models.

They became popular because they removed the strict sequential bottleneck of RNNs and let models learn long-range dependencies more effectively.

## 1. Big Picture

A transformer is not just "attention." It is a stack of interacting components:

- token embeddings
- positional information
- self-attention
- feed-forward networks
- residual connections
- layer normalization
- task-specific output heads

These blocks repeat many times, allowing deep contextual processing.

## 2. Core Attention Equation

```math
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

where:

- $Q$ are queries
- $K$ are keys
- $V$ are values
- $d_k$ is the key dimension

The model computes similarity between queries and keys, converts those scores into weights, and uses them to mix value vectors.

## 3. A Transformer Block

A simplified pre-norm transformer block often looks like:

```math
H' = H + \mathrm{Attention}(\mathrm{LN}(H))
```

```math
H'' = H' + \mathrm{FFN}(\mathrm{LN}(H'))
```

This means:

1. normalize token representations
2. apply attention
3. add a residual shortcut
4. normalize again
5. apply a per-token feed-forward network
6. add another residual shortcut

## 4. Why Transformers Beat Older Sequence Models

### Parallelism

RNNs process tokens one step at a time. Transformers can process a whole sequence in parallel during training.

### Long-Range Context

Attention lets one token directly compare itself with distant tokens in the same sequence.

### Scaling

Transformers benefited strongly from more data, more parameters, and more compute.

## 5. Encoder, Decoder, and Encoder-Decoder Variants

### Encoder-Only

Best for understanding tasks.

Examples:

- BERT
- RoBERTa

### Decoder-Only

Best for autoregressive generation.

Examples:

- GPT
- Llama
- Mistral

### Encoder-Decoder

Best for source-to-target tasks.

Examples:

- T5
- BART

## 6. Shapes Matter

If a batch of token embeddings has shape:

```math
X \in \mathbb{R}^{B \times N \times D}
```

then:

- $B$ = batch size
- $N$ = sequence length
- $D$ = hidden dimension

Attention mixes information across the sequence dimension $N$, while FFNs transform each token along the feature dimension $D$.

## 7. Practical Example: Tiny Transformer Block

```python
import torch
import torch.nn as nn

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=32, nhead=4, d_ff=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        ffn_in = self.ln2(x)
        x = x + self.ffn(ffn_in)
        return x

block = TinyTransformerBlock()
x = torch.randn(2, 10, 32)
y = block(x)
print(y.shape)
```

## 8. Common Weaknesses

Transformers are powerful, but not free:

- attention cost grows with sequence length
- memory use can be large
- training requires substantial compute
- position handling is not automatic
- inference latency matters at scale

These weaknesses motivated sparse attention, efficient attention, KV caching, FlashAttention, and many long-context methods.

## 9. Why This Chapter Has Subchapters

Each major component deserves its own explanation:

- [Attention Mechanisms](Attention_Mechanisms/README.md)
- [Encoder-Decoder Structure](Encoder_Decoder_Structure/README.md)
- [Feed Forward Networks](Feed_Forward_Networks/README.md)
- [Layer Normalization and Residual Connection](Layer_Normalization_and_Residual_Connection/README.md)
- [Positional Encoding](Positional_Encoding/README.md)

## Summary

Transformers are a system of cooperating ideas, not a single formula. Attention handles interactions across tokens, FFNs refine token-wise representations, normalization and residuals stabilize optimization, and positional information restores order. Together, these pieces made large-scale language modeling practical.
