# Diffusion Models

Diffusion models are a class of generative models that learn to reverse a gradual noising process.

They became especially important in modern image generation systems.

## 1. Core Idea

Start with real data $x_0$.

Gradually add noise over many steps until the sample becomes nearly pure noise.

Then train a model to reverse that process.

## 2. Forward Process

```math
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
```

## 3. Direct Noising Formula

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
```

where:

- $\alpha_t = 1-\beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$

## 4. Training Objective

A common simplified objective is noise prediction:

```math
\mathcal{L}_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]
```

This means the model sees a noisy example $x_t$ and learns to predict the noise that was added. If it can predict the noise, it can help reconstruct a cleaner sample.

## 5. Why Diffusion Models Matter

They are strong for:

- image generation
- editing
- inpainting
- conditional synthesis

## 6. Tiny PyTorch Example: Noise Schedule

```python
import torch

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 2e-2, timesteps)

betas = linear_beta_schedule(1000)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

print(betas.shape, alpha_bar.shape)
```

## 7. UNet Backbone

Many diffusion systems use a UNet-like architecture because it handles multi-scale image structure well.

## 8. Sampling Intuition

At inference time:

1. start from noise
2. predict noise or a denoised estimate
3. update the sample slightly
4. repeat for many steps

This iterative refinement is one reason diffusion models produce high-quality images.

## 9. Common Variants

- DDPM
- DDIM
- latent diffusion
- classifier-free guidance systems

## 10. Classifier-Free Guidance Intuition

Classifier-free guidance helps trade off diversity and prompt alignment.

Higher guidance can make outputs follow prompts more strongly, but may reduce diversity.

## 11. Tiny Sampling Skeleton

```python
import torch

x = torch.randn(1, 3, 32, 32)

for t in reversed(range(5)):
    predicted_noise = torch.randn_like(x) * 0.1
    x = x - predicted_noise

print(x.shape)
```

This is not a full diffusion sampler, but it shows the iterative denoising pattern.

## 12. Strengths and Weaknesses

### Strengths

- high sample quality
- stable training compared with many GAN setups

### Weaknesses

- expensive sampling
- heavy compute demand

## 13. Why Latent Diffusion Helped

Operating directly in pixel space is expensive. Latent diffusion first compresses images into a lower-dimensional latent space, then runs diffusion there. This greatly reduces compute while preserving quality.

## Summary

Diffusion models are one of the central generative model families in modern AI, especially for image synthesis.
