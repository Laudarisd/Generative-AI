# Encoding vs Decoding

## Why This Difference Matters

A lot of confusion in NLP comes from mixing up encoder models, decoder models, and encoder-decoder models.

---

## 1. Encoding

An encoder reads the full input and builds contextual representations.

Attention formula:

```math
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

where:

- $Q$ = queries
- $K$ = keys
- $V$ = values
- $d_k$ = key dimension

### Multi-Head Attention

Instead of doing attention once, transformers do it across multiple heads:

```math
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
```

Each head learns a different projection of the input and can focus on different relationships.

### Practical Example

Sentence:

```text
The animal didn't cross the road because it was tired.
```

An encoder can use both left and right context to resolve what `it` refers to.

### Good Use Cases

- classification
- retrieval
- tagging
- embeddings

---

## 2. Decoding

A decoder generates one token at a time.

Probability of the next token:

```math
P(x_t \mid x_{<t})
```

### Why Causal Masking Is Necessary

During training, position $t$ must not see future positions. Otherwise the model would cheat.

This is why decoder attention is masked while encoder attention is not.

### Practical Example

Prompt:

```text
Write a Python function to reverse a string:
```

The model emits the next token repeatedly until the answer is complete.

### Good Use Cases

- chat
- code generation
- autocomplete
- long-form writing

---

## 3. Encoder-Decoder

An encoder-decoder model uses:

- encoder for source understanding
- decoder for target generation

Typical objective:

```math
P(y_t \mid y_{<t}, x)
```

This is useful for tasks like translation and summarization.

## 3.1 Comparison Table

| Component | Encoder | Decoder | Encoder-Decoder |
| --- | --- | --- | --- |
| Context access | full bidirectional context | left-to-right only | full source + causal target |
| Main job | understand input | generate next token | transform source into target |
| Example models | BERT, RoBERTa | GPT, Llama, Qwen | T5, BART |

---

## 4. Python Example: Causal Mask Intuition

```python
tokens = ["I", "love", "AI"]

for i in range(len(tokens)):
    visible_context = tokens[: i + 1]
    print("step", i, "can see", visible_context)
```

This shows the core decoder rule: each position only sees the tokens up to that point.
