# Part 2: Models and Architectures

Part 2 moves from introductory concepts into concrete model families, practical architectures, and the building blocks that power modern generative AI systems.

If Part 1 answers "what is an LLM and why does it matter?", Part 2 answers "how do these models actually work, where are they used, and how are they implemented in practice?"

## What This Part Covers

This part focuses on:

- applied NLP tasks such as classification and representation learning
- transformer internals such as attention, feed-forward layers, normalization, and positional information
- prompt engineering as a practical interface layer for LLM systems
- core generative model families including GANs, VAEs, and diffusion models
- multimodal systems such as vision-language models
- the practical LLM landscape beyond just one model brand

## Recommended Reading Order

1. [Text Classification](Text_Classification/README.md)
2. [NLP](NLP/README.md)
3. [Transformer Overview](Transformer/README.md)
4. [Attention Mechanisms](Transformer/Attention_Mechanisms/README.md)
5. [Encoder-Decoder Structure](Transformer/Encoder_Decoder_Structure/README.md)
6. [Feed Forward Networks](Transformer/Feed_Forward_Networks/README.md)
7. [Layer Normalization and Residual Connection](Transformer/Layer_Normalization_and_Residual_Connection/README.md)
8. [Positional Encoding](Transformer/Positional_Encoding/README.md)
9. [Prompt Engineering](Prompt_Engineering/README.md)
10. [GAN](GAN/README.md)
11. [VAE](VAEs/README.md)
12. [Diffusion Models](Diffusion_models/README.md)
13. [VLM](VLM/README.md)
14. [LLMs Overview](LLMs/README.md)

## Chapter Map

| Chapter | Main Question | What You Should Learn |
| --- | --- | --- |
| [Text Classification](Text_Classification/README.md) | How do models assign labels to text? | features, losses, metrics, encoder-based models |
| [NLP](NLP/README.md) | What does language processing include beyond LLMs? | tasks, pipelines, preprocessing, representation choices |
| [Transformer Overview](Transformer/README.md) | Why did transformers dominate modern AI? | block structure, scaling, training intuition |
| [Prompt Engineering](Prompt_Engineering/README.md) | How do we control model behavior at inference time? | prompting patterns, constraints, evaluation |
| [GAN](GAN/README.md) | How does adversarial generation work? | generator-discriminator training, instability, variants |
| [VAE](VAEs/README.md) | How do probabilistic latent-variable generators work? | latent spaces, KL divergence, reparameterization |
| [Diffusion Models](Diffusion_models/README.md) | How does iterative denoising generate data? | forward process, reverse process, training objective |
| [VLM](VLM/README.md) | How are vision and language combined? | encoders, projectors, alignment, multimodal tasks |
| [LLMs Overview](LLMs/README.md) | How do real-world LLM families differ? | architecture style, openness, deployment tradeoffs |

## How To Use This Part

- Read the transformer chapters together, because the model only makes sense when attention, FFNs, normalization, and position handling are studied as one system.
- Read GANs, VAEs, and diffusion models comparatively, because each family solves generation differently.
- Read prompt engineering and LLM overview after the architecture chapters so the practical system-level decisions feel grounded.

## Practical Theme

A recurring theme in this part is that architecture choices create tradeoffs:

- encoder-heavy models are efficient for understanding tasks
- decoder-heavy models are natural for generation
- encoder-decoder models are strong for source-to-target mapping
- GANs often produce sharp outputs but can be unstable
- VAEs are probabilistic and interpretable but may be blurrier
- diffusion models are powerful but computationally expensive

## Code Philosophy

The code snippets in this part are intentionally small. They are not full production systems. Their purpose is to show:

- tensor shapes
- module interactions
- loss definitions
- data flow through architectures

You should be able to read the code and connect it back to the mathematical definitions.
