# Understanding LLMs

## What This Chapter Tries To Do

This chapter gives a practical mental model of what an LLM is, how it is trained, how it generates answers, and why it behaves the way it does.

The goal is not only to say "an LLM predicts the next token." The goal is to make that sentence meaningful.

---

## 1. What an LLM Learns

An LLM learns statistical structure in token sequences.

Core objective:

```math
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
```

Meaning:

- $x_t$ is the token at position $t$
- $x_{<t}$ means all earlier tokens
- the model predicts each token from previous context

### Practical Example

Prompt:

```text
The capital of France is
```

A strong LLM should assign high probability to the token `Paris`.

---

## 2. Why This Is More Powerful Than It Looks

"Predict the next token" sounds simple, but it creates pressure to model many kinds of structure:

- grammar
- style
- topic flow
- factual regularities
- reasoning patterns
- formatting conventions
- code syntax

To predict the next token well, the model has to internalize a huge amount of regularity from text and code.

---

## 3. Why LLMs Feel Intelligent

They do not "understand" in the human sense, but they model huge amounts of structure well enough to produce useful behavior.

That often looks like intelligence because the outputs can reflect:

- coherent syntax
- relevant facts
- multi-step formatting
- code completion ability
- instruction-following behavior

They are not human thinkers, but they are also far more than autocomplete in the ordinary sense.

---

## 4. Training at a High Level

During training, the model predicts the next token and is penalized when it is wrong:

```math
\mathcal{L} = -\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})
```

### Intuition

- if the model gives the correct token high probability, loss is small
- if the model gives the correct token low probability, loss is large

### Python Example

```python
import math

prob_correct_token = 0.8
loss = -math.log(prob_correct_token)
print(loss)
```

---

## 5. The LLM Pipeline at a High Level

A simplified LLM pipeline looks like this:

1. raw text is tokenized
2. tokens become embeddings
3. transformer layers process those representations
4. output logits are produced for the next token
5. softmax converts logits to probabilities
6. one token is selected
7. the process repeats

This is the loop behind generation.

---

## 6. Pretraining

Pretraining is the stage where the model learns broad language patterns from massive data.

Typical data sources:

- web text
- books
- code
- technical documents
- multilingual corpora

What pretraining gives the model:

- language fluency
- broad pattern recognition
- domain exposure
- general text continuation ability

What it does not automatically give:

- perfect instruction following
- safety alignment
- product-ready behavior

---

## 7. Instruction Tuning and Alignment

After pretraining, many modern LLMs are instruction-tuned.

That means they are trained further on examples such as:

- question -> answer
- instruction -> helpful response
- prompt -> structured output

Later stages may add:

- preference optimization
- RLHF-style tuning
- safety tuning

This is why chat assistants behave differently from raw base models.

---

## 8. Inference and Sampling

At inference time, the model does not only compute probabilities. It must also choose an output token.

Common strategies:

- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling
- beam search in some settings

### Tiny Sampling Example

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

---

## 9. What Makes Modern LLMs Strong

Several ingredients work together:

- transformer architecture
- large-scale pretraining
- better optimization
- instruction tuning
- alignment methods
- serving and inference engineering

This is important: model quality is not only architecture. It is the whole pipeline.

---

## 10. Major Architecture Families

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

## 11. Transformer as the Foundation

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

## 12. Context Windows and Limitations

An LLM only sees a finite context window at once.

That means:

- it does not have unlimited memory
- very long histories may be truncated or compressed
- performance may degrade across long contexts

This is one reason retrieval and external memory systems matter in practice.

---

## 13. Hallucination and Reliability

LLMs generate likely continuations, not guaranteed truth.

That means they can:

- answer correctly
- answer plausibly but incorrectly
- invent citations or details
- sound confident when uncertain

This is why strong products often combine LLMs with:

- retrieval
- verification
- tool use
- structured constraints

---

## 14. Why Prompting Works

Prompting works because the pretrained model has seen many patterns where:

- instructions are followed by answers
- questions are followed by explanations
- code comments are followed by code
- examples suggest a format

Prompting is not magic. It is interface design over a pretrained statistical model.

---

## 15. STIE Concepts for LLM Systems

I am using `STIE` here as a practical LLM workflow lens:

- **S**: Sampling
- **T**: Tuning
- **I**: Inference
- **E**: Evaluation

These are not the only important stages in LLM work, but they are useful for understanding how models are actually used and improved.

### Sampling

Sampling is how we choose output tokens from the model's predicted distribution.

Common options:

- greedy decoding
- temperature sampling
- top-k
- top-p

Sampling affects:

- creativity
- determinism
- factual stability
- verbosity

### Tuning

Tuning means adapting a model beyond raw pretraining.

Examples:

- supervised fine-tuning
- instruction tuning
- preference optimization
- domain adaptation

Tuning affects:

- style
- helpfulness
- safety behavior
- task specialization

### Inference

Inference is the real-time process of using the trained model to generate outputs for user prompts.

This includes:

- tokenization
- context construction
- forward passes
- decoding
- stopping rules

Inference quality depends on:

- model quality
- prompt quality
- decoding strategy
- context management

### Evaluation

Evaluation tells us whether the model is actually good for the target use case.

Possible evaluation styles:

- benchmark tasks
- human preference studies
- factuality checks
- code correctness tests
- latency and cost measurement

### Why STIE Matters

A model can be strong in one stage and weak in another.

Examples:

- great pretraining but poor tuning
- strong model but poor inference settings
- good generations but weak evaluation discipline

This is why LLM engineering is not only about parameter count.

---

## 16. Representative Model Landscape

| Model Family | Architecture Style | Typical Strength | Availability |
| --- | --- | --- | --- |
| BERT / RoBERTa | encoder-only | understanding, classification, embeddings | open models available |
| GPT family | decoder-only | generation, chat, coding | mostly API / hosted depending on version |
| Llama | decoder-only | open-weight general-purpose LLMs | open weights available |
| Qwen | decoder-only and multimodal variants | multilingual and open ecosystem | many open releases |
| Mistral | decoder-only and multimodal variants | efficient serving and strong general generation | mixed open and commercial |
| T5 / FLAN-T5 | encoder-decoder | translation, summarization, structured tasks | open models available |
| Claude / Gemini | large commercial assistant families | chat, reasoning, multimodal tasks | closed hosted APIs |

---

## 17. LLMs in the Larger Model Ecosystem

LLMs matter most for text and code, but they are not the only important generative models.

Related model families include:

- diffusion models for image generation
- VLMs for image + text tasks
- speech models for transcription and voice generation
- voice-interactive assistant systems

Modern AI products are often systems made from multiple model types rather than one single model.

---

## 18. Tiny Python Example: Next-Token Loss Over a Sequence

```python
import math

probs_for_correct_tokens = [0.9, 0.8, 0.6]
loss = -sum(math.log(p) for p in probs_for_correct_tokens)

print("sequence_loss:", loss)
```

---

## 19. Chapter Summary

- an LLM learns token-sequence statistics
- next-token prediction is the core training objective
- pretraining gives broad pattern knowledge
- instruction tuning and alignment shape assistant behavior
- inference requires token selection strategies, not only probability estimation
- transformers are the architecture most modern LLMs rely on
- strong products wrap LLMs with retrieval, tools, and system design

## Practice Questions

1. Why does next-token prediction force a model to learn syntax and style?
2. What is the difference between pretraining and instruction tuning?
3. Why can an LLM sound correct even when it is wrong?
4. Why is sampling strategy part of model behavior?
5. Why are LLM products usually more than only the base model?
