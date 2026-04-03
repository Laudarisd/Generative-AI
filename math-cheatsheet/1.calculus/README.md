# Calculus

Calculus is the language of change. In machine learning, it appears whenever we compute gradients, optimize a loss, or describe a physical process with differential equations.

## 1. Limits and Continuity

The limit of a function describes what value it approaches near a point:

```math
\lim_{x \to a} f(x)
```

A function is continuous at $x=a$ if:

```math
\lim_{x \to a} f(x) = f(a)
```

Why this matters in AI:

- continuous losses are easier to optimize
- smooth activations and kernels often behave more predictably
- differential equations assume differentiable or continuous structure

## 2. Derivatives

The derivative measures local rate of change:

```math
f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}
```

Common rules:

- power rule: $\frac{d}{dx}x^n = nx^{n-1}$
- exponential: $\frac{d}{dx}e^x = e^x$
- logarithm: $\frac{d}{dx}\log x = \frac{1}{x}$
- chain rule: $\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$

### Practical Example

If:

```math
f(w) = (w-3)^2
```

then:

```math
f'(w) = 2(w-3)
```

This is the gradient signal used by gradient descent.

### Python Example

```python
def f(w):
    return (w - 3) ** 2

def grad_f(w):
    return 2 * (w - 3)

w = 8.0
for _ in range(5):
    w = w - 0.1 * grad_f(w)
    print(w, f(w))
```

## 3. Partial Derivatives and Gradients

For multivariable functions, we differentiate with respect to one variable at a time:

```math
\frac{\partial f}{\partial x}, \quad \frac{\partial f}{\partial y}
```

The gradient stacks all partial derivatives:

```math
\nabla f(x_1, \dots, x_n) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```

In ML, if $L(\theta)$ is the loss, then $\nabla_\theta L$ tells us how to change parameters.

## 4. Jacobian and Hessian

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

```math
J_{ij} = \frac{\partial f_i}{\partial x_j}
```

The Hessian of a scalar function is the matrix of second derivatives:

```math
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
```

Why this matters:

- Jacobians appear in backpropagation and change-of-variables formulas
- Hessians describe curvature and help explain optimization behavior

## 5. Taylor Approximation

A first-order approximation near $a$ is:

```math
f(x) \approx f(a) + f'(a)(x-a)
```

A second-order approximation is:

```math
f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2
```

This is useful for:

- understanding Newton-style optimization
- approximating nonlinear behavior locally

## 6. Integrals

An integral accumulates quantity over an interval:

```math
\int_a^b f(x)\,dx
```

The Fundamental Theorem of Calculus links derivatives and integrals:

```math
\frac{d}{dx}\int_a^x f(t)\,dt = f(x)
```

Applications in AI and engineering:

- probability densities integrate to 1
- expected values are integrals or sums
- physics models use integrals for energy, mass, and flux

## 7. Multivariable Calculus

Useful objects:

- directional derivative
- gradient
- divergence
- curl
- multiple integrals

For constrained optimization, Lagrange multipliers are central:

```math
\nabla f(x,y,\dots) = \lambda \nabla g(x,y,\dots)
```

This appears in optimization theory and maximum-entropy style problems.

## 8. Differential Equations

An ordinary differential equation, or **ODE**, relates a function to its derivatives:

```math
\frac{dy}{dt} = f(t, y)
```

A partial differential equation, or **PDE**, uses partial derivatives:

```math
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
```

PDEs matter for:

- heat diffusion
- fluid flow
- electromagnetics
- physics-informed neural networks

## 9. Worked Gradient Example

Suppose:

```math
L(w_1, w_2) = (w_1 + 2w_2 - 5)^2
```

Then:

```math
\frac{\partial L}{\partial w_1} = 2(w_1 + 2w_2 - 5)
```

```math
\frac{\partial L}{\partial w_2} = 4(w_1 + 2w_2 - 5)
```

Gradient descent updates both parameters simultaneously.

### Python Example

```python
w1, w2 = 0.0, 0.0
lr = 0.1

for step in range(5):
    g1 = 2 * (w1 + 2 * w2 - 5)
    g2 = 4 * (w1 + 2 * w2 - 5)
    w1 -= lr * g1
    w2 -= lr * g2
    print(step, w1, w2)
```

## Practice Problems

1. Differentiate $x^3 + 2x^2 - 5x + 1$.
2. Compute the gradient of $f(x,y)=x^2+xy+y^2$.
3. Find the critical points of $f(x)=x^3-3x$.
4. Approximate $\sin(x)$ near $x=0$ using a Taylor expansion.
5. Solve a simple gradient descent step for $L(w)=(w-4)^2$.
