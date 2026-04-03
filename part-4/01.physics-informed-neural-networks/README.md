# Physics-Informed Neural Networks

Physics-informed neural networks, or **PINNs**, combine data fitting with physical constraints such as ordinary differential equations, partial differential equations, boundary conditions, and initial conditions.

This is one of the clearest examples of AI meeting engineering mathematics directly.

## 1. Core Idea

Instead of training only on labeled input-output pairs, a PINN also penalizes violations of known physics.

If the physical system satisfies:

```math
u_t + \mathcal{N}[u] = 0
```

then the network predicts a function $\hat{u}_\theta(t,x)$ and we build a residual:

```math
f_\theta(t,x) = \frac{\partial \hat{u}_\theta}{\partial t} + \mathcal{N}[\hat{u}_\theta]
```

A good model should make this residual small.

## 2. PINN Loss Function

A common total loss is:

```math
\mathcal{L} = \lambda_{data}\mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{phys} + \lambda_{bc}\mathcal{L}_{bc}
```

where:

- $\mathcal{L}_{data}$ fits observed data
- $\mathcal{L}_{phys}$ penalizes PDE or ODE residuals
- $\mathcal{L}_{bc}$ enforces boundary or initial conditions

For collocation points $(t_f^i, x_f^i)$:

```math
\mathcal{L}_{phys} = \frac{1}{N_f}\sum_{i=1}^{N_f}|f_\theta(t_f^i, x_f^i)|^2
```

## 3. Why PINNs Matter

PINNs are useful when:

- labeled simulation data is scarce
- physical laws are known
- we need differentiable surrogates
- interpolation inside a physical domain matters

Applications:

- fluid dynamics
- heat transfer
- material modeling
- inverse problems
- biomedical systems

## 4. Forward vs Inverse Problems

### Forward Problem

Predict the physical state when the governing equation and parameters are known.

### Inverse Problem

Infer unknown physical parameters from observations.

This is one reason PINNs are powerful: they are not only simulators, but parameter-discovery tools.

## 5. Example: 1D Heat Equation

Heat equation:

```math
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
```

Residual:

```math
f_\theta(t,x) = \frac{\partial \hat{u}_\theta}{\partial t} - \alpha \frac{\partial^2 \hat{u}_\theta}{\partial x^2}
```

The network is trained so that:

- observed temperatures are matched
- the PDE residual is small
- boundary conditions are respected

## 6. Automatic Differentiation

PINNs rely heavily on automatic differentiation.

That is how the model computes:

- first derivatives
- second derivatives
- mixed derivatives

without symbolic manual differentiation.

### Python Example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

tx = torch.tensor([[0.2, 0.5]], requires_grad=True)
u = model(tx)

grads = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_t = grads[:, 0:1]
u_x = grads[:, 1:2]

u_xx = torch.autograd.grad(u_x, tx, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
print(u_t, u_x, u_xx)
```

## 7. Strengths and Weaknesses

### Strengths

- uses physics as an inductive bias
- can be data efficient
- handles forward and inverse problems
- produces differentiable surrogate models

### Weaknesses

- optimization can be difficult
- balancing loss terms is nontrivial
- stiff systems can be hard
- performance may degrade in high-dimensional settings

## 8. Beyond PINNs

Related directions:

- neural operators
- operator learning
- Fourier neural operators
- DeepONets
- differentiable simulators

These methods often scale better for families of PDE problems.

## Problems to Think About

1. Why is automatic differentiation essential for PINNs?
2. What changes when the problem is inverse instead of forward?
3. Why might collocation point selection matter?
4. How would you explain PINNs to someone from classical numerical analysis?
5. When would a neural operator be preferable to a PINN?

## References

- Raissi, Perdikaris, and Karniadakis, *Physics Informed Deep Learning* and PINNs overview: https://maziarraissi.github.io/PINNs/
- Original PINNs paper: https://arxiv.org/abs/1711.10561
