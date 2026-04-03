# Matrix Theory

This chapter goes one level deeper than basic linear algebra. Matrix theory matters whenever you want to reason carefully about rank, conditioning, decompositions, stability, and the behavior of large linear systems.

## 1. Rank and Nullity

The rank of a matrix is the dimension of its column space.

The nullity is the dimension of its null space.

Rank-nullity theorem:

```math
\operatorname{rank}(A) + \operatorname{nullity}(A) = n
```

where $n$ is the number of columns of $A$.

Why this matters in ML:

- rank controls expressiveness of linear maps
- low-rank approximations compress models
- null spaces explain non-identifiability

## 2. Symmetric, Positive Definite, and Orthogonal Matrices

### Symmetric

```math
A = A^\top
```

### Positive Definite

```math
\mathbf{x}^\top A \mathbf{x} > 0 \quad \text{for all } \mathbf{x}\neq 0
```

Positive-definite matrices appear in:

- covariance matrices
- quadratic optimization
- Gaussian processes

### Orthogonal

```math
Q^\top Q = I
```

Orthogonal matrices preserve lengths and angles.

## 3. Condition Number

The condition number measures sensitivity:

```math
\kappa(A) = \|A\| \|A^{-1}\|
```

Large condition number means the problem is ill-conditioned.

Practical impact:

- unstable numerical solutions
- unreliable inverse computation
- training instability in some settings

## 4. Spectral Radius

```math
\rho(A) = \max_i |\lambda_i|
```

This determines the long-term behavior of repeated multiplication by $A$.

Applications:

- recurrent systems
- iterative algorithms
- dynamical stability

## 5. Trace and Frobenius Norm

Trace:

```math
\operatorname{tr}(A) = \sum_i A_{ii}
```

Frobenius norm:

```math
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}
```

These appear in:

- matrix regularization
- covariance identities
- reconstruction losses

## 6. SVD and Low-Rank Approximation

If:

```math
A = U\Sigma V^\top
```

then the best rank-$k$ approximation is formed by keeping the top $k$ singular values.

Why this matters:

- dimensionality reduction
- compression
- denoising
- latent semantic structure

### Python Example

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
U, S, VT = np.linalg.svd(A, full_matrices=False)

print("singular values:", S)
```

## 7. Projection Matrices

A projection matrix maps vectors onto a subspace.

For an orthogonal projection onto the column space of $A$:

```math
P = A(A^\top A)^{-1}A^\top
```

This matters for:

- least squares
- residual analysis
- geometry of linear regression

## 8. Least Squares

When $A\mathbf{x}=\mathbf{b}$ has no exact solution, solve:

```math
\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2
```

Normal equations:

```math
A^\top A \mathbf{x} = A^\top \mathbf{b}
```

### Python Example

```python
import numpy as np

A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
b = np.array([1.0, 2.0, 2.5])

x, *_ = np.linalg.lstsq(A, b, rcond=None)
print(x)
```

## 9. Matrix Theory in Transformers

Transformer math is full of matrix structure:

- embeddings as lookup matrices
- attention scores as similarity matrices
- output projections as learned linear maps
- normalization acting on vectors but implemented with efficient tensor operations

Attention score matrix:

```math
S = \frac{QK^\top}{\sqrt{d_k}}
```

This is matrix theory in action, not just notation.

## Practice Problems

1. Explain what rank means geometrically.
2. Compute the trace and Frobenius norm of a small matrix.
3. Show why orthogonal matrices preserve norm.
4. Solve a least-squares line fit with a tiny dataset.
5. Explain why low-rank approximation can compress embeddings.
