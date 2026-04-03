# Matrix Theory

This chapter goes one level deeper than basic linear algebra. Matrix theory matters whenever you want to reason carefully about rank, conditioning, decompositions, stability, and the behavior of large linear systems.

It is especially important in modern AI because large models are built from matrix operations, and many practical problems are really questions about matrix structure, numerical stability, and approximation quality.

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

### Positive Semidefinite

Sometimes we only need:

```math
\mathbf{x}^\top A \mathbf{x} \ge 0
```

This is the positive semidefinite, or **PSD**, case.

PSD matrices are extremely important in:

- covariance matrices
- kernels
- Gaussian processes
- quadratic forms

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

### Worked Example

If a matrix is very close to singular, tiny perturbations in the input can create large changes in the output of a linear solve.

That is why numerically stable methods matter more than symbolic formulas in practical ML systems.

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

### Spectral Norm

Another important norm is the spectral norm:

```math
\|A\|_2 = \sigma_{max}(A)
```

where $\sigma_{max}(A)$ is the largest singular value.

This matters in:

- Lipschitz bounds
- stable training
- operator control

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

### Eckart-Young Intuition

Keeping the largest singular values gives the best low-rank approximation in Frobenius norm.

That is one reason SVD is such a central tool in compression and representation learning.

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

### Key Properties

For an orthogonal projection matrix $P$:

```math
P^2 = P
```

and:

```math
P^\top = P
```

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

## 9. Determinants Revisited

Determinant properties:

```math
\det(AB)=\det(A)\det(B)
```

```math
\det(A^\top)=\det(A)
```

```math
\det(A)=0 \iff A \text{ is singular}
```

Geometric interpretation:

- determinant measures signed volume scaling

## 10. Diagonalization

A matrix is diagonalizable if:

```math
A = PDP^{-1}
```

where $D$ is diagonal.

Why diagonalization matters:

- repeated powers become easier
- linear dynamical systems become interpretable
- spectral methods become tractable

## 11. Pseudoinverse

For non-square or rank-deficient matrices, the Moore-Penrose pseudoinverse is a generalized inverse.

Notation:

```math
A^+
```

This is useful in:

- least squares
- minimum norm solutions
- underdetermined systems

## 12. Kronecker Products and Block Matrices

Kronecker product:

```math
A \otimes B
```

This builds larger structured matrices from smaller ones.

Why this matters:

- tensor operations
- structured covariance
- systems modeling
- advanced numerical methods

Block matrices matter because large ML systems often have structured parameter matrices and covariance matrices.

## 13. Matrix Calculus Basics

Matrix theory and calculus meet in optimization.

Useful identities:

```math
\frac{\partial}{\partial x}(a^\top x)=a
```

```math
\frac{\partial}{\partial x}(x^\top A x)=(A+A^\top)x
```

If $A$ is symmetric:

```math
\frac{\partial}{\partial x}(x^\top A x)=2Ax
```

This identity appears constantly in quadratic optimization.

## 14. Worked Example: Quadratic Form

Let:

```math
A=
\begin{bmatrix}
2 & 1\\
1 & 3
\end{bmatrix},
\quad
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
```

Then:

```math
x^\top A x = 2x_1^2 + 2x_1x_2 + 3x_2^2
```

This is a quadratic surface, and the matrix controls its curvature.

## 15. Matrix Theory in Transformers

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

Other examples:

- weight matrices in MLP layers
- covariance-like statistics in normalization analysis
- low-rank adaptation methods
- structured compression of transformer weights

## 16. Python Example: Pseudoinverse and Rank

```python
import numpy as np

A = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

print("rank:", np.linalg.matrix_rank(A))
print("pseudoinverse:\n", np.linalg.pinv(A))
```

## Practice Problems

1. Explain what rank means geometrically.
2. Compute the trace and Frobenius norm of a small matrix.
3. Show why orthogonal matrices preserve norm.
4. Solve a least-squares line fit with a tiny dataset.
5. Explain why low-rank approximation can compress embeddings.
6. Explain the difference between positive definite and positive semidefinite.
7. Why does a large condition number make a problem unstable?
8. Compute the derivative of $x^\top A x$ when $A$ is symmetric.
