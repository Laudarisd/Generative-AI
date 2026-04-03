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

## 4. Numerical Methods

Many real problems do not have closed-form solutions.

Important tools:

- Euler method
- Runge-Kutta methods
- finite difference methods
- finite element methods

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

## 5. Signals and Systems

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

## 6. Control and Stability

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

## 7. Probability in Engineering

Real measurements are noisy. Engineering mathematics often combines:

- stochastic models
- filtering
- uncertainty propagation

This links directly to Bayesian ML and state estimation.

## 8. Why This Matters for AI

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
