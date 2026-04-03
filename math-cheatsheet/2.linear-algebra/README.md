# Linear Algebra

Linear algebra is the mathematical backbone of modern AI. Embeddings, attention, neural network layers, PCA, and optimization all rely on vectors and matrices.

## 1. Vectors

A vector in $\mathbb{R}^n$ is an ordered list of numbers:

```math
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
```

Basic operations:

- addition
- scalar multiplication
- dot product
- norm

Dot product:

```math
\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i
```

Norm:

```math
\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^\top \mathbf{v}}
```

### Practical Meaning

- embeddings are vectors
- similarity search often uses cosine similarity
- gradients are vectors in parameter space

## 2. Vector Spaces, Span, Basis, Dimension

Key ideas:

- **span**: all linear combinations of given vectors
- **basis**: a minimal set that spans a space
- **dimension**: number of basis vectors

Why this matters:

- low-rank approximation depends on dimension reduction
- PCA finds new basis vectors aligned with variance

## 3. Matrices

A matrix is a rectangular array:

```math
A \in \mathbb{R}^{m \times n}
```

Matrix multiplication:

```math
(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}
```

Why matrix multiplication matters in AI:

- linear layers use $y = Wx + b$
- attention uses $QK^\top$
- batched processing uses matrix operations for speed

### Python Example

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

print(A @ B)
```

## 4. Systems of Linear Equations

Many problems reduce to:

```math
A\mathbf{x} = \mathbf{b}
```

Solutions may be:

- unique
- infinitely many
- none

This is central to least squares, regression, and numerical methods.

## 5. Determinants and Inverses

For square matrices:

```math
\det(A) \neq 0 \implies A^{-1} \text{ exists}
```

Important interpretation:

- determinant describes volume scaling
- zero determinant means the transformation collapses dimension

## 6. Eigenvalues and Eigenvectors

If:

```math
A\mathbf{v} = \lambda \mathbf{v}
```

then $\mathbf{v}$ is an eigenvector and $\lambda$ is the eigenvalue.

Applications:

- PCA
- stability analysis
- Markov chains
- spectral methods

## 7. Orthogonality

Two vectors are orthogonal if:

```math
\mathbf{u}^\top \mathbf{v} = 0
```

Orthogonality matters because:

- it reduces redundancy
- orthonormal bases simplify geometry
- many decompositions depend on it

## 8. Matrix Decompositions

Important decompositions:

- LU
- QR
- eigendecomposition
- SVD

Singular Value Decomposition:

```math
A = U \Sigma V^\top
```

This is one of the most useful tools in applied AI and engineering.

## 9. Embeddings and Linear Layers

An embedding table is a matrix:

```math
E \in \mathbb{R}^{V \times d}
```

where:

- $V$ is vocabulary size
- $d$ is embedding dimension

A token id indexes a row of $E$.

A linear layer applies:

```math
y = Wx + b
```

This is one of the simplest but most reused formulas in deep learning.

### Python Example

```python
import numpy as np

W = np.array([[1.0, -1.0], [0.5, 2.0]])
x = np.array([2.0, 3.0])
b = np.array([0.1, -0.2])

print(W @ x + b)
```

## 10. Cosine Similarity

Cosine similarity between two vectors is:

```math
\cos(\theta) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2\|\mathbf{v}\|_2}
```

Used in:

- embedding search
- semantic retrieval
- recommendation systems

### Python Example

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([2.0, 1.0, 0.0])

sim = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
print(sim)
```

## Practice Problems

1. Compute the dot product of $(1,2,3)$ and $(4,5,6)$.
2. Solve a $2 \times 2$ linear system by hand.
3. Find eigenvalues of a diagonal matrix.
4. Explain why orthogonal vectors are useful in embeddings.
5. Multiply two small matrices and verify the shape.
