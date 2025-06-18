# Era of Generative AI

Welcome to the Era of Generative AI! 🚀

This repository is your gateway to the fascinating world of Generative Artificial Intelligence (AI), where innovation meets imagination. Dive into a realm where machines not only learn from data but also create new data that resembles the real world. From generating lifelike images to crafting human-like text, Generative AI is reshaping the boundaries of what's possible with artificial intelligence.

## Table of Contents

1. [About This Repository](#about-this-repository)
2. [What to Expect](#what-to-expect)
3. [Generative AI](#generative-ai)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
    - [Diffusion Models](#diffusion-models)
4. [Large Language Models (LLMs)](#large-language-models-llms)
5. [Relationship Between Generative AI and LLMs](#relationship-between-generative-ai-and-llms)
6. [Transformer Architecture](#transformer-architecture)
    - [Encoder-Decoder Structure](#encoder-decoder-structure)
    - [Attention Mechanisms](#attention-mechanisms)
    - [Positional Encoding](#positional-encoding)
    - [Feed-Forward Networks](#feed-forward-networks)
    - [Layer Normalization and Residual Connections](#layer-normalization-and-residual-connections)
7. [Training Process](#training-process)
8. [Key Mathematical Concepts](#key-mathematical-concepts)
    - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
    - [Multi-Head Attention](#multi-head-attention)
    - [Positional Encoding (Math)](#positional-encoding-math)
    - [Optimization Algorithms](#optimization-algorithms)
9. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
10. [Prompt Engineering](#prompt-engineering)
11. [Hyperparameter Tuning](#hyperparameter-tuning)
12. [Multimodal Applications](#multimodal-applications)
13. [LangChain](#langchain)
14. [Applications](#applications)
15. [Contribution Guidelines](#contribution-guidelines)
16. [References](#references)
17. [Reading Materials](#reading-materials)

## About This Repository

This repository is a comprehensive resource for exploring Generative AI, covering foundational models, advanced architectures, and practical applications. Here’s what you’ll find:

- **LLM Development**: Build Large Language Models (LLMs) from scratch, exploring text generation and understanding [Click Here](https://github.com/Laudarisd/Generative-AI/tree/main/LLMs).
- **Generative Models**: Dive into GANs, VAEs, and Diffusion Models for image, text, and audio synthesis ([code/generative_models/](code/generative_models/)).
- **Transformer Deep Dive**: Understand the Transformer architecture, the backbone of modern LLMs, with detailed explanations ([papers/attention_is_all_you_need.md](papers/attention_is_all_you_need.md)).
- **Practical Tools**: Use frameworks like LangChain for real-world applications ([code/langchain/](code/langchain/)).
- **Research Insights**: Explore cutting-edge papers and discussions ([papers/](papers/)).

## What to Expect

- **Code Samples and Tutorials**: Hands-on implementations of Generative AI models ([code/](code/)).
- **Research Insights**: Summaries and discussions of seminal papers like "Attention Is All You Need" ([papers/](papers/)).
- **Model Training and Evaluation**: Best practices for training, tuning, and evaluating models ([tutorials/](tutorials/)).

## Generative AI

Generative AI uses machine learning to create new data resembling the training set by learning its statistical distribution. Key models include:

### Generative Adversarial Networks (GANs)
GANs pit a generator against a discriminator: the generator creates fake data, and the discriminator evaluates its authenticity, improving realism over time. Used for image synthesis (e.g., DALL-E). Learn more: [code/generative_models/gans/](code/generative_models/gans/).

### Variational Autoencoders (VAEs)
VAEs encode data into a latent space and decode it to generate new samples by sampling a probability distribution. Great for image generation and data reconstruction. Explore: [code/generative_models/vaes/](code/generative_models/vaes/).

### Diffusion Models
Diffusion Models generate data by iteratively denoising random noise, inspired by thermodynamics. They excel in high-quality image generation (e.g., Stable Diffusion). Details: [code/generative_models/diffusion/](code/generative_models/diffusion/).

**Examples**:
- **Image Generation**: DALL-E creates images from text prompts.
- **Text Generation**: GPT-4 writes essays, code, and more.
- **Music Generation**: Jukedeck composes original music tracks.

## Large Language Models (LLMs)

LLMs are generative models specialized for language tasks, trained on vast text corpora to understand and generate human-like text. They leverage the Transformer architecture for parallel processing and long-range dependency capture. See implementations: [code/llms/](code/llms/).

**Examples**:
- **GPT-3/GPT-4 (OpenAI)**: Human-like text generation and conversation.
- **BERT (Google)**: Bidirectional context for search and NLP tasks.
- **T5 (Google)**: Text-to-text framework for versatile NLP applications.

## Relationship Between Generative AI and LLMs

Generative AI encompasses models for creating data across domains (images, text, audio), while LLMs are a subset focused on language generation and understanding. Both learn data distributions to generate novel content, but LLMs are tailored for text with Transformer-based architectures. Deep dive: [papers/generative_ai_vs_llms.md](papers/generative_ai_vs_llms.md).

## Transformer Architecture

Introduced in "Attention Is All You Need" (2017), the Transformer is the foundation of LLMs, using attention mechanisms for parallel sequence processing. Full explanation: [papers/attention_is_all_you_need.md](papers/attention_is_all_you_need.md).

### Encoder-Decoder Structure
- **Encoder**: Processes input sequences into a rich representation (6 layers in the base model).
- **Decoder**: Generates output sequences auto-regressively (6 layers). LLMs like GPT use only the decoder.

### Attention Mechanisms
Attention allows the model to focus on relevant parts of the input when generating output, capturing dependencies efficiently.

### Positional Encoding
Adds positional information to token embeddings, enabling the model to understand sequence order without recurrence.

### Feed-Forward Networks
Position-wise neural networks (FFNs) in each layer add non-linearity, with an inner dimension of 2048 and output of 512.

### Layer Normalization and Residual Connections
Residual connections (e.g., \( x + \text{Sublayer}(x) \)) and layer normalization stabilize training and improve gradient flow.

## Training Process

Training LLMs involves:
1. **Data Collection**: Large text corpus.
2. **Tokenization**: Split text into tokens.
3. **Model Initialization**: Set initial parameters.
4. **Forward Pass**: Compute predictions.
5. **Loss Calculation**: Measure prediction error.
6. **Backward Pass**: Compute gradients.
7. **Parameter Update**: Adjust parameters using an optimizer.
8. **Iteration**: Repeat until convergence. Tutorial: [tutorials/training_llms.md](tutorials/training_llms.md).

## Key Mathematical Concepts

### Scaled Dot-Product Attention
Computes attention scores as \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \), scaling by \( \sqrt{d_k} \) for stability. Details: [papers/attention_is_all_you_need.md](papers/attention_is_all_you_need.md#scaled-dot-product-attention).

### Multi-Head Attention
Runs multiple attention heads in parallel, capturing diverse relationships: \( \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \). See: [papers/attention_is_all_you_need.md](papers/attention_is_all_you_need.md#multi-head-attention).

### Positional Encoding (Math)
Encodes token positions using sine and cosine functions:
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
More: [papers/attention_is_all_you_need.md](papers/attention_is_all_you_need.md#positional-encoding).

### Optimization Algorithms
- **SGD**: Updates parameters via gradient descent on mini-batches.
- **Adam**: Adaptive learning rate optimizer, combining momentum and RMSProp: \( \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \). Guide: [tutorials/optimization.md](tutorials/optimization.md).

## Retrieval-Augmented Generation (RAG)

RAG enhances generation by retrieving relevant documents from a knowledge base, improving accuracy and context in responses. Example: [code/rag/](code/rag/).

## Prompt Engineering

Crafting effective prompts to guide LLMs, optimizing phrasing for desired outputs. Techniques: [tutorials/prompt_engineering.md](tutorials/prompt_engineering.md).

## Hyperparameter Tuning

Optimizing model hyperparameters (e.g., learning rate, batch size) to improve performance through systematic experimentation. Guide: [tutorials/hyperparameter_tuning.md](tutorials/hyperparameter_tuning.md).

## Multimodal Applications

Generative AI extends to multimodal tasks, combining text, images, audio, and video:
- **Text-to-Image**: DALL-E generates images from text (e.g., "a cat in a hat").
- **Text-to-Video**: Models create video clips from text prompts.
- **Text-to-Audio**: WaveNet generates speech or music. Explore: [code/multimodal/](code/multimodal/).

## LangChain

LangChain is a framework for building applications with LLMs, offering modular tools for workflows like text generation and chatbots. Features:
- **Modular Design**: Flexible integration of models.
- **Pipeline Support**: Multi-step task processing.
- **Ease of Use**: Simplified APIs. Try it: [code/langchain/](code/langchain/).

## Applications

Generative AI and LLMs power:
- **Text Generation**: Articles, stories, code.
- **Translation**: Multilingual text conversion.
- **Summarization**: Condensing documents.
- **Question Answering**: Accurate query responses. Examples: [code/applications/](code/applications/).
---
---
# 🧠 Transformer and LLM Model Overview

---

## 🔍 Model Overview

| Model          | Type            | Use Case                                      | Free        | Source                                                                         |
| -------------- | --------------- | --------------------------------------------- | ----------- | ------------------------------------------------------------------------------ |
| BERT           | Encoder         | Text classification, embeddings, NER          | ✅          | [BERT](https://huggingface.co/bert-base-uncased)                                  |
| RoBERTa        | Encoder         | Improved BERT; robust text understanding      | ✅          | [RoBERTa](https://huggingface.co/roberta-base)                                    |
| DistilBERT     | Encoder         | Lightweight BERT variant for fast inference   | ✅          | [DistilBERT](https://huggingface.co/distilbert-base-uncased)                      |
| ALBERT         | Encoder         | Efficient, parameter-sharing BERT             | ✅          | [ALBERT](https://huggingface.co/albert-base-v2)                                   |
| ELECTRA        | Encoder         | Pretraining alternative to BERT               | ✅          | [ELECTRA](https://huggingface.co/google/electra-base-discriminator)               |
| GPT-2          | Decoder         | Text generation, autocomplete                 | ✅          | [GPT-2](https://huggingface.co/gpt2)                                              |
| GPT-3.5 / 4    | Decoder         | Advanced generation, reasoning, coding        | ❌ API only | [OpenAI API](https://platform.openai.com/)                                        |
| T5             | Encoder-Decoder | Translation, summarization, QA                | ✅          | [T5](https://huggingface.co/t5-base)                                              |
| mT5            | Encoder-Decoder | Multilingual version of T5                    | ✅          | [mT5](https://huggingface.co/google/mt5-base)                                     |
| BART           | Encoder-Decoder | Summarization, text generation                | ✅          | [BART](https://huggingface.co/facebook/bart-base)                                 |
| Pegasus        | Encoder-Decoder | Summarization (Google, document-level)        | ✅          | [Pegasus](https://huggingface.co/google/pegasus-xsum)                             |
| CodeT5         | Encoder-Decoder | Code summarization, generation                | ✅          | [CodeT5](https://huggingface.co/Salesforce/codet5-base)                           |
| StarCoder2     | Decoder         | Code generation, permissive license           | ✅          | [StarCoder2](https://huggingface.co/bigcode/starcoder2-3b)                        |
| Code LLaMA     | Decoder         | Meta’s open model for coding                 | ✅          | [Code LLaMA](https://huggingface.co/codellama/CodeLlama-7b)                       |
| WizardCoder    | Decoder         | Instruction-following code assistant          | ✅          | [WizardCoder](https://huggingface.co/WizardLM/WizardCoder-1B-V1.0)                |
| Phi-2          | Decoder         | Lightweight, trained on textbook-quality data | ✅          | [Phi-2](https://huggingface.co/microsoft/phi-2)                                   |
| LLaMA 2        | Decoder         | Meta's general-purpose open LLM               | ✅          | [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b)                           |
| Mixtral        | Decoder (MoE)   | Mixture of Experts model from Mistral AI      | ✅          | [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)                     |
| Mistral        | Decoder         | Efficient 7B dense model                      | ✅          | [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)                       |
| DeepSeek Coder | Decoder         | Open coding model from DeepSeek               | ✅          | [DeepSeek Coder](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct) |
| Yi 34B         | Decoder         | Chinese + English LLM from 01.AI              | ✅          | [Yi-34B](https://huggingface.co/01-ai/Yi-34B)                                     |
| Gemini         | Decoder         | Google’s multimodal model (closed source)    | ❌          | N/A                                                                            |
| Claude         | Decoder         | Anthropic's chat model (closed source)        | ❌          | N/A                                                                            |
| Grok           | Decoder         | xAI (Elon Musk) — Twitter/X integration      | ❌          | N/A                                                                            |

---

## ✅ Model Comparison Summary

| Task                              | Recommended Models                            |
| --------------------------------- | --------------------------------------------- |
| Text classification               | BERT, RoBERTa, DistilBERT                     |
| Text generation (general)         | GPT-2, T5, BART, Mistral                      |
| Text generation (advanced)        | GPT-4 (paid), Claude (paid), Mixtral, LLaMA 2 |
| Summarization / Translation       | T5, Pegasus, BART, mT5                        |
| Code generation (free)            | StarCoder2, Code LLaMA, WizardCoder, DeepSeek |
| Code instruction/chat             | WizardCoder, CodeT5, DeepSeek Coder           |
| Lightweight model (V100 friendly) | Phi-2, Mistral 7B, CodeT5, StarCoder2-3B      |
| Multilingual support              | mT5, Yi-34B, Code LLaMA, LLaMA 2              |
| Multimodal (image+text)           | Gemini (❌), MiniGPT-4, OpenFlamingo (✅)     |

---

## 📌 Notes

- ✅ = Open-source and free to use commercially
- ❌ = Closed-source or paid API only
- "Decoder" models are typically used for text generation
- "Encoder" models are better for classification, embedding, etc.
- "Encoder-Decoder" models are flexible and good for translation, summarization, etc.

---
---


## Contribution Guidelines

We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get involved.

## References

- [Building LLMs from Scratch](https://www.kaggle.com/code/jayitabhattacharyya/building-llms-from-scratch-generative-ai-report)
- [LLMs from Scratch GitHub](https://github.com/rasbt/LLMs-from-scratch)

## Reading Materials

- [LLM Engineers Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [Attention Is All You Need Paper](https://arxiv.org/pdf/1706.03762) ([summary](papers/attention_is_all_you_need.md))

