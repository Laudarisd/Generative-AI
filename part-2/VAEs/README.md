# VAE

Variational Autoencoders, or **VAEs**, are generative models that learn a compressed latent space and decode from it.

## 1. Core Idea

A VAE has:

- an encoder
- a latent variable $z$
- a decoder

The encoder maps input $x$ into a latent distribution.

The decoder reconstructs data from sampled latent vectors.

## 2. Objective

A common VAE objective combines:

- reconstruction loss
- KL regularization

```math
\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \| p(z))
```

The first term rewards accurate reconstruction.

The second term pushes the learned latent distribution toward a simple prior, often:

```math
p(z) = \mathcal{N}(0, I)
```

## 3. Why VAEs Matter

VAEs are useful because they:

- learn smooth latent spaces
- support controlled generation
- connect deep learning with probabilistic modeling

## 4. Intuition

The model is encouraged to:

- reconstruct data well
- keep the latent space regular and sample-friendly

## 5. Reparameterization Trick

Instead of sampling directly in a non-differentiable way, VAEs use:

```math
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

This keeps training differentiable.

## 6. Tiny PyTorch Example

```python
import torch
import torch.nn as nn

encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU())
mu_layer = nn.Linear(128, 16)
logvar_layer = nn.Linear(128, 16)
decoder = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 784))

x = torch.randn(4, 784)
h = encoder(x)
mu = mu_layer(h)
logvar = logvar_layer(h)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
recon = decoder(z)

print(mu.shape, recon.shape)
```

## 7. Reconstruction vs Regularization Tradeoff

If the KL term is too weak:

- latent space may become messy
- sampling quality may drop

If the KL term is too strong:

- reconstructions may become poor
- the model may ignore latent information

This tradeoff is central to understanding VAEs.

## 8. Example: Sampling New Data

```python
import torch

z = torch.randn(8, 16)
generated = decoder(z)
print(generated.shape)
```

Because the latent space is regularized, new points sampled from the prior can often decode into plausible outputs.

## 9. Strengths and Weaknesses

### Strengths

- principled latent-variable modeling
- smooth latent interpolation
- probabilistic interpretation

### Weaknesses

- output quality may be blurrier than GANs or diffusion models

## 10. VAE Use Cases

- anomaly detection
- representation learning
- controllable generation
- semi-supervised learning
- compression-style latent modeling

## Summary

VAEs are one of the clearest bridges between probabilistic modeling and neural generation.
