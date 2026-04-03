# Encoding vs Decoding

## Why This Difference Matters

A lot of confusion in NLP comes from mixing up encoder models, decoder models, and encoder-decoder models.

The words sound similar, but the architectural roles are different.

---

## 1. What an Encoder Does

An encoder reads the input and builds contextual representations.

Self-attention formula:

```math
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

where:

- $Q$ = queries
- $K$ = keys
- $V$ = values
- $d_k$ = key dimension

### Key Encoder Property

In a standard encoder, each token can attend to the full input sequence.

That means the representation for one token can use:

- left context
- right context
- global sentence structure

This is why encoder models are strong for understanding tasks.

---

## 2. Multi-Head Attention

Instead of doing attention once, transformers do it across multiple heads:

```math
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
```

Each head learns a different projection of the input and can focus on different relationships.

### Why Multiple Heads Matter

Different heads may focus on:

- syntax
- long-distance references
- punctuation cues
- local phrase boundaries
- structural code patterns

---

## 3. Practical Encoder Example

Sentence:

```text
The animal didn't cross the road because it was tired.
```

An encoder can use both left and right context to help resolve what `it` refers to.

That is the core strength of bidirectional contextualization.

### Good Encoder Use Cases

- classification
- retrieval
- tagging
- embeddings
- sentence understanding

---

## 4. What a Decoder Does

A decoder generates one token at a time.

Probability of the next token:

```math
P(x_t \mid x_{<t})
```

This makes the decoder naturally suited for generation.

### Key Decoder Property

At generation time, token $t$ should only depend on earlier tokens.

That is why decoder models are causal.

---

## 5. Why Causal Masking Is Necessary

During training, position $t$ must not see future positions. Otherwise the model would cheat by reading the answer token before predicting it.

That is why decoder self-attention is masked while encoder self-attention is not.

### Intuition

- encoder: full context is allowed
- decoder: future context is blocked

---

## 6. Practical Decoder Example

Prompt:

```text
Write a Python function to reverse a string:
```

The model emits the next token repeatedly until the answer is complete.

### Good Decoder Use Cases

- chat
- code generation
- autocomplete
- long-form writing
- story generation

---

## 7. Encoder-Decoder Models

An encoder-decoder model uses:

- encoder for source understanding
- decoder for target generation

Typical objective:

```math
P(y_t \mid y_{<t}, x)
```

where:

- $x$ is the source sequence
- $y_t$ is the next target token
- $y_{<t}$ is the already generated target prefix

This is especially useful for:

- translation
- summarization
- structured transformation tasks

---

## 8. Why Encoder-Decoder Models Are Different

Encoder-decoder models are not just "middle ground."

They are designed for tasks where:

- input and output both matter strongly
- output should be conditioned on a source sequence
- full source understanding is useful
- target generation must still stay causal

That makes them a natural fit for sequence transduction.

---

## 9. Cross-Attention

In encoder-decoder models, the decoder often uses cross-attention over encoder outputs.

That allows the decoder to:

- read source-side representations
- generate target-side tokens conditionally

This is one of the defining differences between decoder-only and encoder-decoder architectures.

---

## 10. Comparison Table

| Component | Encoder | Decoder | Encoder-Decoder |
| --- | --- | --- | --- |
| Context access | full bidirectional context | left-to-right only | full source + causal target |
| Main job | understand input | generate next token | transform source into target |
| Example models | BERT, RoBERTa | GPT, Llama, Qwen | T5, BART |
| Typical output style | representations or labels | continuation | source-conditioned generation |

---

## 11. When To Prefer Which

### Encoder

Prefer when the main task is understanding.

Examples:

- classification
- search relevance
- embedding generation

### Decoder

Prefer when the main task is free-form generation.

Examples:

- chat
- code completion
- long-form generation

### Encoder-Decoder

Prefer when there is a clear source-to-target mapping.

Examples:

- translation
- summarization
- rewriting

---

## 12. Python Example: Causal Mask Intuition

```python
tokens = ["I", "love", "AI"]

for i in range(len(tokens)):
    visible_context = tokens[: i + 1]
    print("step", i, "can see", visible_context)
```

This shows the core decoder rule: each position only sees the tokens up to that point.

---

## 13. Python Example: Encoder Intuition

```python
tokens = ["The", "animal", "was", "tired"]

for i, token in enumerate(tokens):
    print("token", token, "can use full context:", tokens)
```

This is crude, but conceptually accurate for full-sequence encoder attention.

---

## 14. Chapter Summary

- encoders build rich bidirectional contextual representations
- decoders generate tokens causally from left to right
- encoder-decoder systems combine source understanding with target generation
- masking is central to decoder training correctness
- the architecture should match the task, not only popularity

## Practice Questions

1. Why can an encoder use both left and right context?
2. Why would a decoder "cheat" without a causal mask?
3. Why is encoder-decoder natural for translation?
4. Why are decoder-only models dominant in chat systems?
5. What role does cross-attention play in encoder-decoder models?
