# Machine Learning Mathematics

This chapter connects the math directly to ML models so the formulas feel operational rather than abstract.

## 1. Supervised Learning Objective

Dataset:

```math
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
```

Objective:

```math
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_\theta(x_i), y_i)
```

This is the central optimization problem in supervised ML.

## 2. Linear Regression

Prediction:

```math
\hat{y} = w^\top x + b
```

Loss:

```math
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

## 3. Logistic Regression

Probability:

```math
P(y=1\mid x) = \sigma(w^\top x + b)
```

Used for binary classification.

## 4. Softmax Classification

For $K$ classes:

```math
P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
```

This is the backbone of many classifiers and token prediction heads.

## 5. Backpropagation Intuition

Backpropagation applies the chain rule repeatedly through layers.

If:

```math
z = Wx + b, \quad a = \phi(z)
```

then gradients flow from the loss back to:

- output
- activation
- linear transformation
- parameters

## 6. Attention Mathematics

Scaled dot-product attention:

```math
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

Multi-head attention:

```math
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O
```

Why the scale term matters:

- without it, dot products grow too large
- softmax becomes too sharp
- gradients become harder to train

## 7. Positional Encoding

Classic sinusoidal positional encoding:

```math
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

## 8. Embeddings

Embedding matrix:

```math
E \in \mathbb{R}^{V \times d}
```

Each token maps to a row vector in this matrix.

## 9. Probabilistic View of LLMs

Autoregressive factorization:

```math
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
```

Training objective:

```math
\mathcal{L} = - \sum_{t=1}^{T}\log P(x_t \mid x_{<t})
```

## 10. Tiny Python Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 3.0], [4.0, 5.0]])

scores = Q @ K.T / np.sqrt(2)
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
output = weights @ V

print(output)
```

## Practice Problems

1. Derive the gradient of a linear regression loss.
2. Explain why softmax outputs sum to 1.
3. Compute a toy attention matrix by hand.
4. Explain the shape of an embedding matrix.
5. Show how next-token prediction defines a full sequence probability.
