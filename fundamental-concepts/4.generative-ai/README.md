# Generative AI

## What Makes Generative AI Different

Traditional predictive ML often answers:

- what class is this?
- what number should we predict?

Generative AI answers:

- what new text should we write?
- what image should we create?
- what code should we generate?

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

### VAEs

VAEs learn a latent representation and decode from it.

### GANs

GANs use a generator and a discriminator in competition.

### Diffusion Models

Diffusion models learn to reverse a noising process.

---

## 3. Why LLMs Are Generative Models

An LLM predicts the next token:

```math
P(x_t \mid x_{<t})
```

Repeating that step builds paragraphs, summaries, code, and more.

### Python Example

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

Continue to [Loss Functions, Optimization, and Regularization](../5.loss-functions-optimization-regularization/README.md).
