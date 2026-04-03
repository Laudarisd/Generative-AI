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

Common in regression:

```math
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

where:

- $y_i$ is the true value
- $\hat{y}_i$ is the predicted value

### Practical Example

If the real house price is `200000` and the model predicts `180000`, the error is squared and contributes to the loss.

### Python Example

```python
import numpy as np

y_true = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.5, 6.0])

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)
```

### 1.2 Binary Cross-Entropy

Used in binary classification:

```math
\mathcal{L}_{BCE} =
- \frac{1}{N}\sum_{i=1}^{N}
\left[
y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)
\right]
```

### Practical Example

For fraud detection:

- true label `1`
- model output `0.95`

This gives lower loss than predicting `0.10`.

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

### 1.3 Cross-Entropy

Used in multi-class classification and token prediction:

```math
\mathcal{L}_{CE} = - \sum_{k=1}^{K} y_k \log \hat{y}_k
```

This is one of the most important losses in deep learning and LLM training.

### 1.4 Negative Log-Likelihood

```math
\mathcal{L}_{NLL} = - \log P_\theta(y \mid x)
```

This appears directly in language modeling.

### Practical Example for LLMs

If the correct next token is `"Paris"`:

- probability `0.90` means small loss
- probability `0.01` means large loss

That simple idea is at the heart of LLM training.

---

## 2. Optimization

Once we compute loss, we need to reduce it. Optimization is the process that updates the model parameters.

## 2.1 Gradient Descent

The standard update rule:

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
```

where:

- $\theta$ are model parameters
- $\eta$ is the learning rate
- $\nabla_\theta \mathcal{L}$ is the gradient

### Practical Meaning

The gradient tells us which direction increases the loss.

Moving in the negative direction tends to reduce the loss.

### Python Example

```python
w = 5.0
learning_rate = 0.1

for step in range(5):
    grad = 2 * (w - 1)  # derivative of (w - 1)^2
    w = w - learning_rate * grad
    print(step, w)
```

## 2.2 Stochastic Gradient Descent

Instead of computing the gradient on the full dataset, we compute it on a batch:

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{batch}
```

Why it matters:

- faster
- scales to large datasets
- standard in deep learning

## 2.3 Adam and AdamW

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

Why people use Adam:

- stable in practice
- works well for deep models
- common in transformers

AdamW adds decoupled weight decay and is especially common in LLM training.

---

## 3. Regularization

Regularization means techniques that improve generalization.

Without regularization, a model may overfit:

- very good on training data
- weak on new unseen data

## 3.1 L2 Regularization / Weight Decay

Penalizes large weights:

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_2^2
```

Why it helps:

- discourages very large parameter values
- often improves generalization

## 3.2 L1 Regularization

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_1
```

This can encourage sparsity.

## 3.3 Dropout

Dropout randomly disables some neurons during training.

Why it helps:

- prevents co-adaptation
- reduces overfitting in many networks

### Python Example

```python
import torch
import torch.nn as nn

drop = nn.Dropout(p=0.5)
x = torch.ones(5)

print(drop(x))
```

## 3.4 Early Stopping

Stop training when validation performance stops improving.

This is one of the simplest and most effective practical regularization methods.

## 3.5 Data Augmentation

Generate additional training variations.

Examples:

- rotate and crop images
- replace words with synonyms in text augmentation
- inject noise in audio

---

## 4. Overfitting vs Underfitting

- **Overfitting**: model memorizes training data
- **Underfitting**: model is too simple or not trained enough

### Practical Signs

Overfitting often looks like:

- training loss keeps going down
- validation loss starts going up

Underfitting often looks like:

- both training and validation loss stay high

---

## 5. Putting It Together

Training a model usually means:

1. choose a loss function
2. compute gradients
3. update parameters with an optimizer
4. apply regularization to improve generalization

That pattern holds for:

- linear models
- neural networks
- transformers
- LLMs

Continue to [Part 1: LLM Foundations](../../part-1/README.md).
