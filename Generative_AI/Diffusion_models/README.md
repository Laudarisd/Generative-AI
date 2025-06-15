🧪 Generative AI: Diffusion Models
Welcome to the Diffusion Models repository — an educational and practical resource for exploring diffusion-based generative AI. This repo provides theoretical insights, mathematical foundations, and hands-on implementations to help you understand and build diffusion models from scratch.

📚 Overview
Diffusion models are a class of generative models that generate high-quality data (e.g., images, audio) by reversing a gradual noising process. They power state-of-the-art applications like Stable Diffusion and DALL·E 2, offering superior stability and fidelity compared to GANs.
This repository includes:

Core concepts and intuition behind diffusion models
Mathematical formulations with corrected equations
Python implementations using PyTorch
Code snippets for key algorithms
Comparisons with GANs and VAEs
Advanced topics like DDIM and Latent Diffusion
Visualizations and experiments


📌 Table of Contents

1. Introduction
2. Core Concepts
3. Mathematical Foundations
4. Algorithms and Code Examples
5. Advanced Topics
6. Setup & Installation



1. Introduction
Diffusion models work by corrupting data with noise over time and learning to reverse this process to generate new samples. They excel in:

Image Generation: Photorealistic images (Stable Diffusion)
Audio Synthesis: High-fidelity audio (DiffWave)
Video and 3D: Emerging applications in dynamic data

This repo is designed for researchers, students, and engineers curious about generative AI.

2. 🔍 Core Concepts
Forward Diffusion (Noising)
Data $( x_0 )$ is gradually noised over $( T )$ steps using a Markov chain:
$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$

$( \beta_t )$: Noise schedule (e.g., linear or cosine)
$( x_T )$: Nearly pure Gaussian noise

Reverse Diffusion (Denoising)
A neural network ( \epsilon_\theta(x_t, t) ) predicts the noise to recover 
$( x_0 ):p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$The model iteratively denoises from $( x_T )$ to $( x_0 )$.

3. 🧮 Mathematical Foundations
Forward Process
The noised sample at step ( t ) can be computed directly:$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$ Where:

( \alpha_t = 1 - \beta_t )
( \bar{\alpha}t = \prod{s=1}^t \alpha_s )

Training Objective
The simplified loss minimizes the difference between predicted and actual noise:[\mathcal{L}{\text{simple}} = \mathbb{E}{t, x_0, \epsilon} \left[ \left| \epsilon - \epsilon_\theta \left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right|^2 \right]]
Variational Lower Bound
The full objective optimizes:[\mathcal{L}{\text{vlb}} = \mathbb{E}q \left[ - \log p\theta(x_0) + \sum{t=1}^T D_{\text{KL}} \left( q(x_t \mid x_{t-1}, x_0) \mid\mid p_\theta(x_{t-1} \mid x_t) \right) \right]]

4. ⚙️ Algorithms and Code Examples
Denoising Diffusion Probabilistic Model (DDPM)
Proposed by Ho et al. (2020), DDPM trains a model to predict noise.
Code Snippet: DDPM Noise Scheduler
import torch

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alpha_bar(betas):
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

# Example usage
T = 1000
betas = linear_beta_schedule(T)
alpha_bar = get_alpha_bar(betas)

DDIM (Denoising Diffusion Implicit Models)
DDIM accelerates sampling by making the reverse process deterministic.
Code Snippet: DDIM Sampling
@torch.no_grad()
def ddim_sample(model, x_t, t, eta=0.0):
    alpha_bar_t = alpha_bar[t]
    alpha_bar_t_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
    epsilon = model(x_t, t)
    sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev))
    mean = torch.sqrt(alpha_bar_t_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_t_prev - sigma**2) * epsilon
    return mean + sigma * torch.randn_like(x_t)

Score-Based Models
These use the score function ( \nabla_x \log p(x_t) ) to guide denoising.

5. Advanced Topics

Classifier-Free Guidance: Improves sample quality using conditional generation.
Latent Diffusion: Operates in a compressed latent space (e.g., Stable Diffusion).
Text-to-Image Models: Conditions diffusion with CLIP embeddings.
Efficient Training:
Gradient checkpointing
Mixed precision (FP16)


UNet Architecture: Backbone with time embeddings.

Code Snippet: Simple UNet Time Embedding
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(1, dim)
        self.activation = nn.SiLU()

    def forward(self, t):
        t = t.unsqueeze(-1)
        return self.activation(self.linear(t))

# Example
time_emb = TimeEmbedding(dim=128)
t = torch.tensor([0.5])
emb = time_emb(t)


6. 🛠️ Setup & Installation
Clone the repository:
git clone https://github.com/yourusername/diffusion-models.git
cd diffusion-models

Install dependencies:
pip install -r requirements.txt

Sample requirements.txt:
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
matplotlib==3.7.2
tqdm==4.65.0
diffusers==0.18.0
transformers==4.30.2


7. 📊 Usage Examples
Train a DDPM Model
python scripts/train.py --dataset mnist --timesteps 1000 --batch-size 64

Generate Samples
python scripts/sample.py --model-path checkpoints/ddpm_mnist.pth --num-samples 100

Visualize Results
import matplotlib.pyplot as plt

def plot_samples(samples):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()


8. 🤝 Contributing
Contributions are encouraged! To contribute:

Fork the repo.
Create a new branch (git checkout - name my-feature).
Commit changes (git commit - m "Add feature").
Push to the branch (git push origin my-feature).
Open a pull request.

Please follow PEP 8 for Python code style.

9. 📚 References

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
Song, Y., & Ermon, S. (2021). Score-Based Generative Modeling Through Stochastic Differential Equations. ICLR.
Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML.
Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.


10. License 📜
This project is licensed under the MIT License.
