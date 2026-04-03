# Optimization and Losses

This chapter rebuilds the old optimization, loss, and activation notes into one compact reference.

## 1. Loss Functions

A loss function measures prediction error.

### Mean Squared Error

```math
\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
```

### Mean Absolute Error

```math
\mathcal{L}_{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
```

### Binary Cross-Entropy

```math
\mathcal{L}_{BCE} = - \frac{1}{N}\sum_{i=1}^{N}\left[y_i\log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]
```

### Cross-Entropy

```math
\mathcal{L}_{CE} = - \sum_{k=1}^{K} y_k \log \hat{y}_k
```

### KL Divergence

```math
D_{KL}(P\|Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}
```

## 2. Gradient Descent

Standard update:

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
```

where:

- $\theta$ is the parameter vector
- $\eta$ is the learning rate

### Python Example

```python
w = 5.0
lr = 0.1

for _ in range(5):
    grad = 2 * (w - 1)
    w -= lr * grad
    print(w)
```

## 3. Common Optimizers

### SGD

Uses one example or a mini-batch at each step.

### Momentum

```math
v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}
```

```math
\theta_{t+1} = \theta_t - \eta v_t
```

### RMSProp

Tracks a moving average of squared gradients.

### Adam

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```

```math
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

## 4. Regularization

### L2 / Weight Decay

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_2^2
```

### L1

```math
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \|\theta\|_1
```

### Other Tools

- dropout
- early stopping
- data augmentation
- label smoothing

## 5. Activation Functions

### Sigmoid

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

### Tanh

```math
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
```

### ReLU

```math
\mathrm{ReLU}(x)=\max(0,x)
```

### Softmax

```math
\mathrm{Softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
```

### GELU

Common in transformers. Smooth and effective in large deep models.

## 6. Training Dynamics

Watch for:

- learning rate too high
- learning rate too low
- overfitting
- unstable gradients
- bad initialization

## 7. Perplexity

For language modeling:

```math
\mathrm{Perplexity} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log p_i\right)
```

## 8. Practical Model Matching

- regression: MSE, MAE, Huber
- classification: BCE, CE
- segmentation: Dice, IoU, BCE hybrids
- metric learning: triplet loss, contrastive loss
- LLMs: cross-entropy / NLL with next-token prediction

## Practice Problems

1. Derive the gradient of MSE for a scalar prediction.
2. Explain why Adam is often easier to use than plain SGD.
3. Compare L1 and L2 regularization.
4. Explain why softmax is used at the output of multi-class classifiers.
5. Compute one BCE term for $y=1$ and $\hat{y}=0.9$.
