# GAN

Generative Adversarial Networks, or **GANs**, are a class of generative models built from competition between two neural networks.

## 1. Core Idea

A GAN has:

- a **generator** $G$
- a **discriminator** $D$

The generator tries to create realistic samples.

The discriminator tries to tell real samples from fake samples.

## 2. Objective

Classic GAN objective:

```math
\min_G \max_D \ \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
```

where:

- $x$ is real data
- $z$ is latent noise
- $G(z)$ is a generated sample

This is a minimax game:

- the discriminator wants to maximize its ability to separate real and fake
- the generator wants to minimize that ability by producing convincing samples

## 3. Why GANs Were Important

GANs were influential because they showed strong sample quality for:

- image generation
- super-resolution
- image-to-image translation

## 4. Strengths

- sharp image outputs
- strong sample realism
- useful for image synthesis tasks

## 5. Weaknesses

- unstable training
- mode collapse
- difficult evaluation

### Mode Collapse

Mode collapse means the generator learns to produce only a narrow subset of possible outputs.

For example, instead of generating many different faces, it may generate many very similar faces.

## 6. Training Intuition

The typical loop alternates:

1. update the discriminator using real and generated samples
2. update the generator so generated samples fool the discriminator

This adversarial setup can create very strong gradients, but it can also become unstable if one side becomes much stronger than the other.

## 7. Tiny PyTorch Example

```python
import torch
import torch.nn as nn

generator = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh(),
)

discriminator = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

z = torch.randn(16, 100)
fake = generator(z)
score = discriminator(fake)

print(fake.shape, score.shape)
```

## 8. Example: One GAN Training Step Skeleton

```python
import torch
import torch.nn as nn

bce = nn.BCELoss()

real = torch.randn(16, 784)
z = torch.randn(16, 100)
fake = generator(z)

real_score = discriminator(real)
fake_score = discriminator(fake.detach())

real_loss = bce(real_score, torch.ones_like(real_score))
fake_loss = bce(fake_score, torch.zeros_like(fake_score))
d_loss = real_loss + fake_loss

g_score = discriminator(fake)
g_loss = bce(g_score, torch.ones_like(g_score))

print("D loss:", float(d_loss))
print("G loss:", float(g_loss))
```

## 9. Common Variants

- DCGAN
- Conditional GAN
- CycleGAN
- StyleGAN
- WGAN / WGAN-GP

## 10. GANs vs VAEs vs Diffusion

| Model | Main Strength | Main Weakness |
| --- | --- | --- |
| GAN | sharp samples | unstable training |
| VAE | structured latent space | blurrier outputs |
| Diffusion | high quality and stable training | slower sampling |

## Summary

GANs are historically one of the most important generative model families. They remain essential for understanding how adversarial training changed image generation research.
