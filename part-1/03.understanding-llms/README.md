# Understanding LLMs

## What This Chapter Tries To Do

This chapter gives a practical mental model of what an LLM is, how it is trained, and why it behaves the way it does.

---

## 1. What an LLM Learns

An LLM learns statistical patterns in token sequences.

Core objective:

```math
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
```

Meaning:

- $x_t$ is the token at position $t$
- $x_{<t}$ means all earlier tokens

### Practical Example

Prompt:

```text
The capital of France is
```

A good LLM should assign high probability to the token `Paris`.

---

## 2. Why LLMs Feel Intelligent

They do not "understand" in a human way, but they can model huge amounts of structure:

- grammar
- style
- facts seen during training
- common reasoning patterns
- code structure

That is enough to make them very useful.

---

## 3. Training at a High Level

During training, the model predicts the next token and is penalized when it is wrong:

```math
\mathcal{L} = -\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})
```

### Python Example

```python
import math

prob_correct_token = 0.8
loss = -math.log(prob_correct_token)
print(loss)
```

If the model gives the correct token high probability, the loss is small. If the probability is low, the loss is large.

---

## 4. What Makes Modern LLMs Strong

- transformer architecture
- large-scale pretraining
- better optimization
- instruction tuning
- high-quality serving and inference systems

## 4.1 Major Model Architecture Families

| Family | Strength | Typical Use |
| --- | --- | --- |
| Encoder-only | understanding | classification, embeddings, retrieval |
| Decoder-only | generation | chat, code generation, completion |
| Encoder-decoder | source-to-target generation | translation, summarization |

### Practical Model Examples

- BERT: encoder-only
- GPT / Llama / Qwen: decoder-only
- T5 / BART: encoder-decoder

---

## 5. Common Uses

- chat assistants
- code generation
- summarization
- retrieval-augmented answering
- translation

## 5.1 Why LLMs Became So Popular

They combine:

- flexible prompting
- broad task coverage
- reusable pretrained knowledge
- strong interface fit for conversational products

## 6. Representative Model Landscape

| Model Family | Architecture Style | Typical Strength | Availability |
| --- | --- | --- | --- |
| BERT / RoBERTa | encoder-only | understanding, classification, embeddings | open models available |
| GPT family | decoder-only | generation, chat, coding | mostly API / hosted depending on version |
| Llama | decoder-only | open-weight general-purpose LLMs | open weights available |
| Qwen | decoder-only and multimodal variants | multilingual and open ecosystem | many open releases |
| Mistral | decoder-only and multimodal variants | efficient serving and strong general generation | mixed open and commercial |
| T5 / FLAN-T5 | encoder-decoder | translation, summarization, structured tasks | open models available |
| Claude / Gemini | large commercial assistant families | chat, reasoning, multimodal tasks | closed hosted APIs |

### Practical Reading of the Table

- if you want embeddings or classification, encoder families are still important
- if you want chat or code generation, decoder families dominate
- if you want source-to-target generation, encoder-decoder models remain useful

## 7. LLMs in the Larger Model Ecosystem

LLMs matter most for text and code, but they are not the only important generative models.

Related model families include:

- diffusion models for image generation
- VLMs for image + text tasks
- speech models for transcription and voice generation

This broader context matters because modern AI products are often systems made from multiple model types rather than one single model.

## 7. Transformer as the Foundation

Most modern LLMs are built on the transformer architecture.

Key ingredients:

- multi-head self-attention
- feed-forward layers
- residual connections
- layer normalization

Why transformers won:

- parallel sequence processing
- strong long-range dependency modeling
- good scaling behavior with data and compute

### A Useful Summary

- transformers are the architecture
- encoders and decoders are architectural roles inside transformer systems
- LLMs are large language models usually built with transformer blocks

---

## 8. A Tiny Sampling Example

```python
import random

next_token_probs = {
    "Paris": 0.7,
    "Lyon": 0.2,
    "London": 0.1,
}

token = random.choices(
    population=list(next_token_probs.keys()),
    weights=list(next_token_probs.values()),
    k=1,
)[0]

print(token)
```

This is the basic intuition of generation: choose the next token from a learned probability distribution.
