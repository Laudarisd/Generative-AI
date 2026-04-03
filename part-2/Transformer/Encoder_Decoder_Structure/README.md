# Encoder-Decoder Structure

The original transformer architecture contains both an encoder stack and a decoder stack.

## 1. Encoder Role

The encoder reads the source sequence and builds contextual representations for every token.

If the source is:

```text
English sentence -> "the weather is nice"
```

the encoder turns this sequence into learned vectors that contain context, meaning, and relationships between words.

## 2. Decoder Role

The decoder generates the target sequence one step at a time.

## 3. Encoder-Decoder Objective

```math
P(y_t \mid y_{<t}, x)
```

where:

- $x$ is the source sequence
- $y_t$ is the next target token

The decoder learns to model:

- the source sequence through cross-attention
- the already-generated target prefix through masked self-attention

## 4. Why This Matters

This structure is ideal for:

- translation
- summarization
- rewriting
- question generation
- style transfer

## 5. Decoder Masking

The decoder uses causal masking so it cannot see future target tokens during training.

## 6. Cross-Attention

The decoder also reads encoder outputs through cross-attention.

This is what makes encoder-decoder transformers different from decoder-only LLMs:

- decoder-only models use only prefix context
- encoder-decoder models can condition explicitly on a separate source input

## 7. Full Data Flow

The data flow is:

1. source tokens -> encoder embeddings
2. encoder stack -> contextual source memory
3. target prefix tokens -> decoder embeddings
4. masked self-attention over the target prefix
5. cross-attention over encoder outputs
6. linear layer + softmax to predict next target token

## 8. Training vs Inference

### Training

Teacher forcing is commonly used. The model sees the correct previous target tokens.

### Inference

The model generates tokens autoregressively:

```text
<bos> -> token1 -> token2 -> token3 -> ...
```

## 9. Practical Example: Translation Intuition

Source:

```text
I love machine learning
```

Target prefix during decoding:

```text
J'
```

The decoder predicts the next target token conditioned on:

- the encoder's representation of the English sentence
- the partial French output already produced

## 10. Python Example: Structural Intuition

```python
source = ["translate", "this"]
target_prefix = ["traduire"]

print("encoder reads:", source)
print("decoder has generated:", target_prefix)
```

## 11. Tiny PyTorch Skeleton

```python
import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
decoder_layer = nn.TransformerDecoderLayer(d_model=32, nhead=4, batch_first=True)

encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

src = torch.randn(2, 5, 32)
tgt = torch.randn(2, 4, 32)

memory = encoder(src)
out = decoder(tgt, memory)

print("memory:", memory.shape)
print("decoder output:", out.shape)
```

## 12. When to Prefer Encoder-Decoder Models

They are a strong choice when:

- input and output are clearly different sequences
- faithful conditioning matters
- summarization or translation is the main task

They are often less convenient than decoder-only models for general chat-style open-ended interaction, which is why decoder-only models dominate assistant products.

## Summary

Encoder-decoder transformers are built for source-to-target sequence generation.
