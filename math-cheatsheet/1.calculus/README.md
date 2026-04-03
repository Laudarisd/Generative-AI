# Calculus

Calculus is the language of change. In machine learning, it appears whenever we compute gradients, optimize a loss, model growth, or describe a physical process with differential equations.

This chapter is broader than a quick formula list. It is designed to cover the most useful material from late high school through bachelor-level introductory calculus and multivariable calculus.

## 0. Precalculus Foundations

Before differentiation and integration, it helps to be comfortable with:

- functions and graphs
- exponents and logarithms
- trigonometric functions
- sequences and series intuition

Common functions:

- polynomial: $x^2, x^3$
- rational: $\frac{1}{x}$
- exponential: $e^x$
- logarithmic: $\log x$
- trigonometric: $\sin x, \cos x$

These functions reappear throughout AI formulas.

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

### Worked Example

Evaluate:

```math
\lim_{x \to 2}(3x+1)
```

Since linear functions are continuous:

```math
\lim_{x \to 2}(3x+1)=3(2)+1=7
```

### Important Limit

```math
\lim_{h \to 0} \frac{\sin h}{h} = 1
```

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
- product rule: $(fg)' = f'g + fg'$
- quotient rule: $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$

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

### Worked Example

Differentiate:

```math
f(x)=x^3 + 2x^2 - 5x + 1
```

Solution:

```math
f'(x)=3x^2 + 4x - 5
```

### Trigonometric and Exponential Derivatives

```math
\frac{d}{dx}\sin x = \cos x
```

```math
\frac{d}{dx}\cos x = -\sin x
```

```math
\frac{d}{dx}a^x = a^x \ln a
```

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

### Worked Example

If:

```math
f(x,y)=x^2+xy+y^2
```

then:

```math
\frac{\partial f}{\partial x}=2x+y
```

```math
\frac{\partial f}{\partial y}=x+2y
```

So:

```math
\nabla f(x,y)=
\begin{bmatrix}
2x+y \\
x+2y
\end{bmatrix}
```

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

### Common Series Around 0

```math
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots
```

```math
\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots
```

```math
\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \dots
```

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

### Basic Antiderivatives

```math
\int x^n\,dx = \frac{x^{n+1}}{n+1}+C \quad (n \ne -1)
```

```math
\int e^x\,dx = e^x + C
```

```math
\int \frac{1}{x}\,dx = \ln|x| + C
```

### Worked Example

Compute:

```math
\int_0^2 (3x^2+1)\,dx
```

Solution:

```math
\int (3x^2+1)\,dx = x^3 + x
```

So:

```math
[x^3+x]_0^2 = (8+2)-0 = 10
```

## 7. Techniques of Integration

Important techniques:

- substitution
- integration by parts
- partial fractions
- trigonometric identities

### Substitution Example

Compute:

```math
\int 2x(x^2+1)^3\,dx
```

Let:

```math
u=x^2+1,\quad du=2x\,dx
```

Then:

```math
\int u^3\,du = \frac{u^4}{4}+C = \frac{(x^2+1)^4}{4}+C
```

### Integration by Parts

Formula:

```math
\int u\,dv = uv - \int v\,du
```

## 8. Applications of Derivatives

Derivatives are used to analyze:

- increasing and decreasing behavior
- maxima and minima
- concavity
- inflection points
- rates of change

### Critical Points

Critical points occur where:

```math
f'(x)=0
```

or where the derivative is undefined.

### Worked Example

Let:

```math
f(x)=x^3-3x
```

Then:

```math
f'(x)=3x^2-3=3(x^2-1)
```

Critical points:

```math
x=\pm1
```

Second derivative:

```math
f''(x)=6x
```

At $x=-1$, $f''(-1)<0$, so local maximum.

At $x=1$, $f''(1)>0$, so local minimum.

## 9. Sequences and Series

A sequence is an ordered list:

```math
\{a_n\}_{n=1}^{\infty}
```

A series is a sum:

```math
\sum_{n=1}^{\infty} a_n
```

Important examples:

- geometric series
- harmonic series
- p-series
- Taylor series

Geometric series:

```math
\sum_{n=0}^{\infty} ar^n
```

converges if $|r|<1$.

## 10. Parametric and Polar Calculus

Parametric form:

```math
x=f(t), \quad y=g(t)
```

Derivative:

```math
\frac{dy}{dx} = \frac{dy/dt}{dx/dt}
```

Polar area:

```math
A=\frac{1}{2}\int_\alpha^\beta r(\theta)^2\,d\theta
```

## 11. Multivariable Calculus

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

## 12. Vector Calculus

Important operators:

- gradient: $\nabla f$
- divergence: $\nabla \cdot \mathbf{F}$
- curl: $\nabla \times \mathbf{F}$

These appear in fluid dynamics, electromagnetics, conservation laws, and engineering AI.

## 13. Differential Equations

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

### ODE Example

Exponential decay:

```math
\frac{dy}{dt} = -ky
```

Solution:

```math
y(t)=Ce^{-kt}
```

## 14. Worked Gradient Example

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
6. Compute $\int_0^1 (2x+3)\,dx$.
7. Use substitution to integrate $\int 3x^2(x^3+1)^4\,dx$.
8. Explain why $\sqrt{d_k}$ scaling in attention is a calculus-friendly training trick rather than arbitrary notation.
