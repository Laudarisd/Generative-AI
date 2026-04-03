# Real Analysis

Real analysis gives the rigorous foundation behind continuity, convergence, approximation, optimization, and probability limits. You do not need full theorem-proof mastery to study AI, but the core ideas make advanced mathematics much clearer.

## 1. Sequences and Convergence

A sequence $\{x_n\}$ converges to $L$ if:

```math
\forall \varepsilon > 0,\ \exists N \text{ such that } n \ge N \Rightarrow |x_n - L| < \varepsilon
```

This idea appears in:

- optimizer convergence
- iterative numerical methods
- stochastic approximation

## 2. Cauchy Sequences

A sequence is Cauchy if its terms get arbitrarily close to each other.

Why this matters:

- it gives a way to describe convergence without already knowing the limit
- complete spaces guarantee Cauchy sequences converge

## 3. Continuity and Uniform Continuity

Continuity at a point means small input change leads to small output change.

Uniform continuity is stronger: one $\delta$ works across the whole domain.

This matters in approximation theory and stability analysis.

## 4. Sequences of Functions

Pointwise convergence:

```math
f_n(x) \to f(x) \quad \text{for each } x
```

Uniform convergence:

```math
\sup_x |f_n(x)-f(x)| \to 0
```

Why this matters:

- uniform convergence preserves more useful properties
- approximation theory for neural networks and kernels often uses this language

## 5. Compactness and Boundedness

These ideas explain why optimization and approximation arguments work on restricted domains.

In many practical settings:

- bounded parameter regions are easier to analyze
- compact domains allow existence theorems

## 6. Differentiability

Differentiability is stronger than continuity.

If a function is differentiable, it is locally well approximated by a linear map.

In higher dimensions, the derivative is the Jacobian.

## 7. Convexity

A function $f$ is convex if:

```math
f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)
```

for $\lambda \in [0,1]$.

Why convexity matters:

- global optimization is easier
- linear regression and logistic regression have well-behaved objectives
- deep learning usually uses non-convex objectives, so this becomes a contrast point

## 8. Lipschitz Continuity

A function is Lipschitz if:

```math
|f(x)-f(y)| \le L|x-y|
```

for some constant $L$.

This matters in:

- optimization theory
- numerical stability
- robustness arguments

## 9. Measure and Integration Intuition

Real analysis supports modern probability via measure theory.

You do not need the full machinery early, but it helps to know:

- probability is a measure
- expectation is an integral
- densities are integrated over domains

## 10. AI-Relevant Interpretation

Real analysis helps you reason about:

- why gradient descent can converge or fail
- why approximation theorems are stated carefully
- why some functions are stable and others are not
- how limits of models behave as width, depth, or data size grows

### Tiny Python Example

```python
values = [1.0]
for _ in range(6):
    values.append(values[-1] / 2)

print(values)
```

This sequence converges to 0 and gives a simple intuition for convergence.

## 11. Worked Example: Uniform Convergence

Consider:

```math
f_n(x)=\frac{x}{n}
```

on $[0,1]$.

For each fixed $x$, $f_n(x)\to 0$.

Also:

```math
\sup_{x \in [0,1]} \left|\frac{x}{n}\right| = \frac{1}{n} \to 0
```

So the convergence is uniform.

## 12. Why Real Analysis Helps With AI Maturity

Real analysis changes the way you read ML papers:

- approximation claims become clearer
- convergence statements become less mysterious
- stability arguments become easier to judge
- regularity assumptions stop looking decorative

### Tiny Python Example: Convex Function

```python
def f(x):
    return x**2

xs = [-2, -1, 0, 1, 2]
print([f(x) for x in xs])
```

## Practice Problems

1. Give an example of a convergent and a divergent sequence.
2. Explain pointwise vs uniform convergence in plain language.
3. Show why every differentiable function is continuous.
4. Give an example of a convex function used in ML.
5. Explain why bounded domains are easier to analyze.
6. Show that $f_n(x)=x/n$ converges uniformly to 0 on $[0,1]$.
