# LLM History

## Why History Matters

LLMs did not appear suddenly. They are the result of several waves of progress in natural language processing, optimization, hardware, and data scaling.

Understanding the history helps explain why current models look the way they do.

---

## 1. Before Transformers

Earlier NLP systems relied on:

- rules and symbolic systems
- n-gram language models
- hidden Markov models
- recurrent neural networks
- LSTMs and GRUs

### The Limitation

These methods could work, but they struggled with:

- long-range dependencies
- parallel training
- large context handling

Example:

In a long paragraph, the subject introduced in the first sentence may matter in the last sentence. RNN-style models often had difficulty preserving that context well.

---

## 2. The Transformer Shift

The 2017 paper *Attention Is All You Need* changed the field by replacing recurrence with attention.

Core attention formula:

```math
\mathrm{Attention}(Q, K, V) =
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Why it mattered:

- parallel training became easier
- long-distance token interactions became easier to model
- scaling behavior improved

---

## 3. Early Transformer Language Models

Important milestones:

- **BERT**: strong encoder model for language understanding
- **GPT / GPT-2**: decoder-only direction for generation
- **T5**: text-to-text framing for many NLP tasks

### Practical Difference

- BERT was great at understanding tasks
- GPT-style models became dominant for generation

---

## 4. The Scaling Era

The field learned that performance often improves when we scale:

- data
- parameter count
- compute

This is one reason modern LLMs became much stronger than earlier language models.

## 4.1 How LLMs Are Developed

Modern LLM development usually includes:

1. collecting large text and code corpora
2. choosing an architecture
3. pretraining on large-scale next-token or related objectives
4. fine-tuning or instruction tuning
5. alignment and evaluation

### Practical Example

A coding assistant model may start with general web and code data, then be further tuned on instruction-response coding pairs so it becomes more useful in chat form.

---

## 5. Instruction Tuning and Chat Models

Base models learn language patterns, but users want helpful assistants.

That led to:

- supervised instruction tuning
- preference optimization
- RLHF-style alignment

The result is the modern chat assistant behavior most people associate with LLMs.

## 5.1 Specialized and Multimodal Models

As the field matured, model families diversified:

- general-purpose chat models
- coding models
- multilingual models
- multimodal models that can process text and images

This is why the modern model landscape includes:

- text-only LLMs
- VLMs
- code-specialized models
- real-time voice systems built around language models

---

## 6. A Tiny Python Example: N-Gram Intuition

This is not an LLM, but it helps show how language modeling started.

```python
from collections import defaultdict

text = "the cat sat on the mat the cat slept"
words = text.split()

bigram_counts = defaultdict(int)

for i in range(len(words) - 1):
    bigram = (words[i], words[i + 1])
    bigram_counts[bigram] += 1

print(dict(bigram_counts))
```

Old language models counted local patterns like this. LLMs still model next-token prediction, but with much richer neural representations.
