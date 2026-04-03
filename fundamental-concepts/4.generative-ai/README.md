# Generative AI

## What Makes Generative AI Different

Traditional predictive ML often answers:

- what class is this?
- what number should we predict?

Generative AI answers:

- what new text should we write?
- what image should we create?
- what code should we generate?
- what audio should we produce?

---

## 1. Discriminative vs Generative Models

Discriminative models often learn:

```math
P(y \mid x)
```

Generative models often learn:

```math
P(x)
\quad \text{or} \quad
P(x, y)
```

### Practical Comparison

- sentiment classifier: predicts positive or negative
- recommendation score model: predicts click probability
- LLM: generates the next sentence
- diffusion model: generates an image from a prompt

---

## 2. Main Families of Generative Models

### Autoregressive Models

```math
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
```

Examples:

- GPT-style LLMs
- code generation models

### Variational Autoencoders

VAEs learn a latent representation and decode from it.

Core idea:

- encoder maps data into a latent space
- decoder reconstructs samples from latent variables

### GANs

GANs use two networks:

- generator
- discriminator

The generator tries to create realistic samples. The discriminator tries to distinguish real samples from fake ones.

### Diffusion Models

Diffusion models learn to reverse a noising process.

This is the core idea behind many modern image generation systems.

---

## 3. Why LLMs Are Generative Models

An LLM predicts the next token:

```math
P(x_t \mid x_{<t})
```

Repeating that step builds:

- paragraphs
- summaries
- code
- structured outputs
- dialogue responses

### Practical Example

Prompt:

```text
Write a short apology email for a delayed response.
```

The model repeatedly predicts the next token until a full answer is produced.

---

## 4. Conditioning and Prompting

Generative models usually do not create outputs from nothing in practical applications. They are often conditioned on:

- a text prompt
- a class label
- a latent code
- another image or audio signal

Examples:

- text prompt to image generator
- prompt plus history to chatbot
- source sentence to translation model

---

## 5. Strengths and Limitations

### Strengths

- flexible output generation
- broad task coverage
- strong interface fit for chat and creative tools
- useful for automation and content synthesis

### Limitations

- hallucination
- bias in generated output
- high compute cost
- evaluation can be difficult

---

## 6. Tiny Python Example: Sampling Next Token Probabilities

This is not a real LLM. It is only a toy example to show the idea of a probability distribution over next tokens.

```python
import random

next_token_probs = {
    "Paris": 0.7,
    "Lyon": 0.2,
    "London": 0.1,
}

tokens = list(next_token_probs.keys())
weights = list(next_token_probs.values())

sampled = random.choices(tokens, weights=weights, k=5)
print(sampled)
```

This demonstrates the idea that the model defines a distribution over possible continuations.

Continue to [Loss Functions, Optimization, and Regularization](../5.loss-functions-optimization-regularization/README.md).

---

## 7. Major Modalities in Generative AI

Generative AI is not only text generation.

### Text Generation

Examples:

- chatbots
- summarization
- code generation
- question answering

Typical model family:

- autoregressive transformers

### Image Generation

Examples:

- text-to-image systems
- image editing
- inpainting

Typical model families:

- diffusion models
- autoregressive image models

### Audio and Voice Generation

Examples:

- text-to-speech
- speech continuation
- voice cloning

### Video Generation

Examples:

- text-to-video
- image-to-video
- video editing

Each modality changes the data representation, but the high-level idea is the same: learn a distribution and sample from it.

---

## 8. Training Objectives in Generative Models

Different generative models learn in different ways.

### Autoregressive Objective

Predict the next token given previous tokens:

```math
\mathcal{L} = - \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
```

This is the core objective behind many LLMs.

### Reconstruction Objective

Autoencoders and VAEs try to reconstruct the input from a compressed latent representation.

### Adversarial Objective

GANs train a generator and discriminator in competition.

### Denoising Objective

Diffusion models learn how to remove noise step by step.

The objective changes, but the practical goal stays the same: produce realistic and useful outputs.

---

## 9. Decoding and Sampling

A trained generative model still needs a strategy for choosing outputs at inference time.

Common decoding methods:

- **greedy decoding**: always pick the highest-probability next token
- **beam search**: keep several strong candidates
- **temperature sampling**: control randomness
- **top-k sampling**: sample from the top $k$ candidates
- **top-p sampling**: sample from the smallest set whose cumulative probability exceeds $p$

### Practical Example

If temperature is low, outputs become more deterministic.

If temperature is high, outputs become more diverse but also more error-prone.

### Python Example

```python
import numpy as np

logits = np.array([2.2, 1.1, 0.4])
temperature = 0.7

scaled = logits / temperature
exp = np.exp(scaled - scaled.max())
probs = exp / exp.sum()

print("probabilities:", probs)
print("greedy_choice:", np.argmax(probs))
```

---

## 10. Fine-Tuning, Prompting, and Retrieval

Real generative AI systems are rarely only "pretrain once and deploy".

Common adaptation methods:

- **prompting**: guide behavior through instructions and examples
- **fine-tuning**: continue training on task-specific data
- **retrieval-augmented generation (RAG)**: fetch external knowledge and give it to the model

### Practical Example

If you want a model to answer company-policy questions:

- prompting may help format
- fine-tuning may improve tone or task style
- RAG may provide the freshest policy content

---

## 11. Evaluation Is Harder Than in Classical ML

Generative AI is harder to evaluate because many outputs can be acceptable at once.

Possible evaluation signals:

- exact match
- BLEU, ROUGE, or similar overlap metrics
- perplexity
- human preference
- task success
- grounded factuality

### Practical Example

Two summaries may use different wording but both be correct. That makes pure string matching weak.

---

## 12. Risks and Responsible Use

Professional understanding of generative AI includes its risks.

Important concerns:

- hallucination
- prompt injection
- copyright and licensing
- privacy leakage
- harmful or biased output
- unreliable reasoning under distribution shift

This matters because strong generation quality does not automatically mean strong truthfulness or safety.
