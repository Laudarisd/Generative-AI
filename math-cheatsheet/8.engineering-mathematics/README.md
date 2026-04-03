# Engineering Mathematics

Engineering mathematics connects pure math to real systems. It matters for control, signal processing, differential equations, numerical simulation, remote sensing, and physics-based AI.

## 1. Differential Equations

### Ordinary Differential Equations

```math
\frac{dy}{dt} = f(t,y)
```

### Partial Differential Equations

```math
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
```

These describe:

- heat flow
- wave propagation
- fluid motion
- diffusion

## 2. Fourier Series and Fourier Transform

Fourier transform:

```math
\mathcal{F}(f)(\xi) = \int_{-\infty}^{\infty} f(x)e^{-2\pi i x \xi}\,dx
```

Why this matters:

- frequency analysis
- signal processing
- image filtering
- spectral methods

## 3. Laplace Transform

```math
\mathcal{L}\{f(t)\} = \int_0^\infty e^{-st} f(t)\,dt
```

Useful for solving linear differential equations and analyzing systems.

## 4. Linear Systems and Transfer Functions

Many engineering systems can be approximated as linear systems around an operating point.

Transfer functions describe input-output behavior in the transform domain.

Why this matters:

- control systems
- filtering
- circuit analysis
- physical system modeling

## 5. Numerical Methods

Many real problems do not have closed-form solutions.

Important tools:

- Euler method
- Runge-Kutta methods
- finite difference methods
- finite element methods
- spectral methods

### Euler Method

```math
y_{n+1} = y_n + h f(t_n, y_n)
```

### Python Example

```python
def f(t, y):
    return -0.5 * y

t, y, h = 0.0, 2.0, 0.1
for _ in range(5):
    y = y + h * f(t, y)
    t += h
print(t, y)
```

### Runge-Kutta Intuition

Runge-Kutta methods improve over Euler by using several slope estimates within one step.

That usually gives better accuracy for the same step size.

## 6. Finite Difference Intuition

A derivative can be approximated numerically:

```math
f'(x) \approx \frac{f(x+h)-f(x)}{h}
```

### Python Example

```python
def f(x):
    return x**2

x = 2.0
h = 1e-3
approx = (f(x + h) - f(x)) / h
print(approx)
```

## 7. Signals and Systems

Concepts:

- time domain
- frequency domain
- transfer functions
- convolution

Convolution:

```math
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)\,d\tau
```

This appears in:

- CNNs
- filtering
- system response analysis

## 7.5 Sampling and Discretization

Continuous physical signals are often measured digitally.

Important idea:

- if you sample too coarsely, you may lose important information

This matters in:

- sensor systems
- audio models
- remote sensing
- control loops

## 8. Control and Stability

State-space form:

```math
\dot{x} = Ax + Bu
```

```math
y = Cx + Du
```

These ideas matter for:

- robotics
- autonomous systems
- scientific simulators
- energy systems

## 8.5 Stability Intuition

A stable system does not explode under small perturbations.

This matters in:

- simulation
- control
- recurrent computation
- iterative numerical schemes

Poor stability can make a mathematically correct model unusable in practice.

## 9. Probability in Engineering

Real measurements are noisy. Engineering mathematics often combines:

- stochastic models
- filtering
- uncertainty propagation

This links directly to Bayesian ML and state estimation.

## 9.5 Kalman Filter Intuition

The Kalman filter is a classic engineering tool for estimating hidden state from noisy observations.

It combines:

- a prediction step
- an update step

This is a beautiful example of probability, linear algebra, and systems theory working together.

## 10. Worked Example: First-Order Decay

Consider:

```math
\frac{dy}{dt}=-0.5y, \quad y(0)=2
```

Analytical solution:

```math
y(t)=2e^{-0.5t}
```

## 11. Why This Matters for AI

Engineering math is the bridge to:

- physics-informed neural networks
- neural operators
- digital twins
- climate and energy forecasting
- remote sensing

## Practice Problems

1. Explain the difference between an ODE and a PDE.
2. Apply one Euler step to a simple decay equation.
3. Describe one use of the Fourier transform in AI.
4. Explain what convolution means in words.
5. Give one engineering system that can be modeled in state-space form.
6. Approximate the derivative of $f(x)=x^2$ at $x=3$ using finite differences.
