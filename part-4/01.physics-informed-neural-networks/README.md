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

This is the key shift from ordinary supervised learning. The target is not only observed data. The target is also consistency with the governing equation.

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

The weighting terms matter a lot. If the data term dominates, the model may fit observations while violating physics. If the physics term dominates too much, the model may satisfy the equation but fit measured data poorly.

## 3. Why PINNs Matter

PINNs are useful when:

- labeled simulation data is scarce
- physical laws are known
- we need differentiable surrogates
- interpolation inside a physical domain matters
- inverse parameter estimation is needed

Applications include:

- fluid dynamics
- heat transfer
- wave propagation
- material modeling
- inverse problems
- biomedical systems
- geophysics

## 4. Forward vs Inverse Problems

### Forward Problem

Predict the physical state when the governing equation and parameters are known.

### Inverse Problem

Infer unknown physical parameters from observations.

This is one reason PINNs are powerful: they are not only simulators, but parameter-discovery tools.

For example, if thermal diffusivity $\alpha$ is unknown in a heat equation, the model may learn both:

- the solution field $u(t,x)$
- the physical parameter $\alpha$

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

## 6. Collocation Points

A PINN does not only evaluate the model at measurement locations. It also samples collocation points inside the domain where the PDE residual is penalized.

These points provide a way to tell the network:

- "not only match the observed values"
- "also behave correctly everywhere in the interior"

## 7. Automatic Differentiation

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

grad_u = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
u_t = grad_u[:, 0:1]
u_x = grad_u[:, 1:2]

u_xx = torch.autograd.grad(u_x, tx, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

print(u.shape, u_t.shape, u_xx.shape)
```

## 8. Boundary and Initial Conditions

Scientific problems usually come with conditions such as:

- temperature fixed at the boundary
- velocity zero at a wall
- initial displacement known at time zero

These can be encoded as extra loss terms.

Example initial condition loss:

```math
\mathcal{L}_{ic} = \frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}} |\hat{u}_\theta(0, x_i) - u_0(x_i)|^2
```

## 9. Why PINNs Are Attractive

PINNs offer several conceptual benefits:

- they combine physical knowledge and data
- they work well when observations are sparse
- they provide differentiable surrogates
- they can solve inverse problems in one framework

## 10. Why PINNs Are Hard

PINNs also have real limitations:

- optimization can be difficult
- balancing multiple loss terms is nontrivial
- higher-order derivatives can be unstable or expensive
- performance on very complex PDEs can be disappointing
- convergence can depend heavily on sampling strategy

## 11. Practical Workflow

A typical PINN workflow is:

1. define domain and governing equation
2. define boundary and initial conditions
3. choose collocation points
4. define neural network $\hat{u}_\theta$
5. compute PDE residual with autodiff
6. train on combined loss
7. validate against known solutions or simulation outputs

## 12. Tiny Residual Example for Heat Equation

```python
import torch

alpha = 0.1

def heat_residual(model, tx):
    tx = tx.clone().detach().requires_grad_(True)
    u = model(tx)
    grad_u = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    return u_t - alpha * u_xx
```

## 13. PINNs vs Ordinary Surrogates

| Feature | Standard Neural Surrogate | PINN |
| --- | --- | --- |
| uses observed data | yes | yes |
| uses known equations | no | yes |
| needs many labels | often yes | sometimes less |
| inverse parameter estimation | indirect | natural fit |

## 14. What To Watch In Practice

When training PINNs, monitor:

- physics residual magnitude
- boundary condition error
- data fit error
- gradient scale imbalance
- sensitivity to collocation sampling

## Problems to Think About

1. Why can a PINN be useful when measured data is sparse?
2. Why are collocation points different from ordinary labeled samples?
3. Why is automatic differentiation central to PINNs?
4. Why can inverse problems be more interesting than forward simulation alone?
5. Why can balancing the loss terms become difficult?

## Summary

PINNs are one of the strongest examples of scientific AI because they bring differential equations directly into the learning objective. They are appealing when physics is known but data is limited, though training them well requires much more care than ordinary supervised modeling.
