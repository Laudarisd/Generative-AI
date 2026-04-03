# Loss Functions, Optimization, and Regularization

## Why This Chapter Deserves Its Own Place

Many beginners can name a model but still do not understand how it learns.

This chapter answers three practical questions:

1. how does the model know it is wrong?
2. how does it update itself?
3. how do we stop it from memorizing the training data too much?

---

## 1. Loss Functions

A loss function measures prediction error.

If the model makes good predictions, the loss is low.

If the model makes bad predictions, the loss is high.

### 1.1 Mean Squared Error

```math
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

Used in:

- regression
- reconstruction tasks

### Python Example

```python
import numpy as np

y_true = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.5, 6.0])

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)
```

### 1.2 Mean Absolute Error

```math
\mathcal{L}_{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
```

MAE is often more robust to outliers than MSE.

### 1.3 Binary Cross-Entropy

```math
\mathcal{L}_{BCE} =
- \frac{1}{N}\sum_{i=1}^{N}
\left[
 y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)
\right]
```

Used in binary classification.

### Python Example

```python
import numpy as np

y_true = np.array([1, 0, 1])
y_pred = np.array([0.95, 0.20, 0.60])

eps = 1e-9
bce = -np.mean(
    y_true * np.log(y_pred + eps) +
    (1 - y_true) * np.log(1 - y_pred + eps)
)

print("BCE:", bce)
```

### 1.4 Cross-Entropy

```math
\mathcal{L}_{CE} = - \sum_{k=1}^{K} y_k \log \hat{y}_k
```

Used in:

- multi-class classification
- token prediction in language models

### 1.5 Negative Log-Likelihood

```math
\mathcal{L}_{NLL} = - \log P_\theta(y \mid x)
```

This appears directly in probabilistic modeling and LLM training.

### 1.6 Other Useful Losses

- Huber loss for robust regression
- KL divergence for comparing distributions
- hinge loss for SVM-style margins
- focal loss for imbalanced classification

---

## 2. Optimization

Once we compute loss, we need to reduce it. Optimization updates the model parameters.

### 2.1 Gradient Descent

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
```

where:

- $\theta$ are model parameters
- $\eta$ is the learning rate
- $\nabla_\theta \mathcal{L}$ is the gradient

### Python Example

```python
w = 5.0
learning_rate = 0.1

for step in range(5):
    grad = 2 * (w - 1)  # derivative of (w - 1)^2
    w = w - learning_rate * grad
    print(step, w)
```

### 2.2 Stochastic Gradient Descent

Instead of using the full dataset, SGD uses one sample or a small batch.

Why it matters:

- faster updates
- scales to large data
- common in deep learning

### 2.3 Momentum

Momentum smooths updates over time.

```math
v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}
```

```math
\theta_{t+1} = \theta_t - \eta v_t
```

### 2.4 Adam and AdamW

Adam tracks moving averages of gradients:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```

```math
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

AdamW is a very common optimizer for transformer models.

---

## 3. Regularization

Regularization means techniques that improve generalization and reduce overfitting.

### 3.1 L2 Regularization / Weight Decay

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_2^2
```

### 3.2 L1 Regularization

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_1
```

### 3.3 Dropout

Dropout randomly disables some neurons during training.

```python
import torch
import torch.nn as nn

drop = nn.Dropout(p=0.5)
x = torch.ones(5)

print(drop(x))
```

### 3.4 Early Stopping

Stop training when validation performance stops improving.

### 3.5 Data Augmentation

Examples:

- rotate and crop images
- synonym replacement in text
- noise injection in audio

---

## 4. Overfitting vs Underfitting

- **Overfitting**: model performs very well on training data but poorly on unseen data
- **Underfitting**: model is too weak or too poorly trained to capture the pattern

Typical signs:

- overfitting: training loss down, validation loss up
- underfitting: both losses remain high

---

## 5. Activation Functions

Activation functions introduce nonlinearity into neural networks.

### ReLU

```math
\mathrm{ReLU}(x) = \max(0, x)
```

### Sigmoid

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### Tanh

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

### Softmax

```math
\mathrm{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

### GELU

GELU is widely used in transformer models such as BERT-style architectures.

---

## 6. Putting It Together

Training a model usually means:

1. choose a loss function
2. compute gradients
3. update parameters with an optimizer
4. apply regularization to improve generalization
5. choose activations that help learning and expressiveness

That pattern holds for:

- linear models
- neural networks
- transformers
- LLMs

Continue to [Bayesian Thinking](../6.bayesian-thinking/README.md).

---

## 7. Learning Rate Schedules and Training Stability

The learning rate is one of the most important hyperparameters in optimization.

If it is too small:

- training becomes very slow

If it is too large:

- training can oscillate or diverge

Common schedules:

- step decay
- cosine decay
- linear warmup
- exponential decay

### Why Warmup Matters

Large modern models, especially transformers, often train more stably if the learning rate starts small and increases gradually in the first phase.

### Python Example

```python
base_lr = 1e-3
warmup_steps = 5

for step in range(10):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        lr = base_lr
    print(step, lr)
```

---

## 8. Batch Size and Gradient Noise

Batch size is the number of examples used to estimate the gradient in one update.

Small batch sizes:

- noisier gradients
- often better regularization
- less memory use

Large batch sizes:

- smoother gradients
- better hardware utilization
- often need learning-rate tuning

### Practical Example

If a model trains well with batch size 32 but becomes unstable at 2048, the issue may not be the architecture. It may be the optimization settings.

---

## 9. Perplexity and Language Modeling Loss

In language modeling, cross-entropy and negative log-likelihood are often reported as **perplexity**.

```math
\mathrm{Perplexity} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log p_i\right)
```

Lower perplexity usually means the model assigns higher probability to the observed sequence.

### Why it matters

Perplexity is a natural metric for next-token prediction, though it does not fully capture usefulness or factual accuracy.

### Python Example

```python
import numpy as np

token_probs = np.array([0.8, 0.6, 0.5, 0.9])
nll = -np.mean(np.log(token_probs))
perplexity = np.exp(nll)

print("nll:", nll)
print("perplexity:", perplexity)
```

---

## 10. Label Smoothing and Generalization

Label smoothing is a simple technique that softens one-hot targets.

Instead of assigning probability 1.0 to the correct class and 0.0 to all others, we assign something like:

- correct class: 0.9
- remaining probability spread over other classes

Why this helps:

- reduces overconfidence
- may improve calibration
- can improve generalization

### Practical Example

This is common in classification and sequence modeling where very sharp target distributions can make the model too certain.

---

## 11. Regularization in Practice

Regularization is not only one formula. It is a family of practical controls.

Common tools:

- weight decay
- dropout
- early stopping
- data augmentation
- smaller model size
- label smoothing
- more data

Professional practice means choosing the cheapest regularization that solves the real issue instead of randomly stacking techniques.
