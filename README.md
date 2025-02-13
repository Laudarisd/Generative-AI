# Era of Generative AI

Welcome to the Era of Generative AI! 🚀

This repository is your gateway to the fascinating world of Generative Artificial Intelligence (AI), where innovation meets imagination. Dive into a realm where machines not only learn from data but also create new data that resembles the real world. From generating lifelike images to crafting human-like text, Generative AI is reshaping the boundaries of what's possible with artificial intelligence.

## Table of Contents

1. [About This Repository](#about-this-repository)
2. [What to Expect](#what-to-expect)
3. [Generative AI](#generative-ai)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
4. [Large Language Models (LLMs)](#large-language-models-llms)
5. [Relationship between Generative AI and LLMs](#relationship-between-generative-ai-and-llms)
6. [Transformer Architecture](#transformer-architecture)
7. [Training Process](#training-process)
8. [Key Mathematical Concepts](#key-mathematical-concepts)
    - [Self-Attention](#self-attention)
    - [Positional Encoding](#positional-encoding)
    - [Optimization Algorithms](#optimization-algorithms)
9. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
10. [Prompt Engineering](#prompt-engineering)
11. [Hyperparameter Tuning](#hyperparameter-tuning)
12. [Language Change and Multimodal Applications](#language-change-and-multimodal-applications)
13. [LangChain](#langchain)
14. [Applications](#applications)
15. [Contribution Guidelines](#contribution-guidelines)
16. [References](#references)

## About This Repository

In this repository, we delve deep into the realm of Generative AI, exploring cutting-edge techniques and models that push the boundaries of creativity and innovation. Here's what you can expect:

- **LLM Models Development**: Embark on a journey to develop Language Model (LLM) models from scratch, unraveling the mysteries of human-like text generation and understanding.
- **GANs and Beyond**: Explore the mesmerizing world of Generative Adversarial Networks (GANs) and their variants, witnessing the evolution of image synthesis and creative expression.
- **Variational Autoencoders (VAEs)**: Discover the power of Variational Autoencoders (VAEs) in encoding and decoding data, exploring their applications in image generation and beyond.
- **Natural Language Processing (NLP)**: Engage in the intersection of Generative AI and Natural Language Processing (NLP), unlocking the potential of language models in text generation and understanding.

## What to Expect

- **Code Samples and Tutorials**: Get hands-on with our collection of code samples and tutorials, designed to guide you through the implementation of various Generative AI models.
- **Research Insights and Discussions**: Stay updated with the latest research insights and discussions in the field of Generative AI, exploring cutting-edge papers and engaging in thought-provoking discussions.
- **Model Training and Evaluation**: Learn best practices for training and evaluating Generative AI models, optimizing performance, and interpreting model outputs.

## Generative AI

Generative AI involves the use of advanced machine learning models, particularly generative models, to produce new data instances that resemble a given training dataset. These models learn the underlying statistical distributions of the training data and use this knowledge to generate novel data points. The two primary types of generative models are Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

### Generative Adversarial Networks (GANs)

GANs consist of two neural networks, a generator and a discriminator, that are trained simultaneously. The generator creates fake data instances, while the discriminator evaluates their authenticity. Through this adversarial process, the generator improves its ability to produce realistic data.

### Variational Autoencoders (VAEs)

VAEs encode input data into a latent space and then decode it to generate new data instances. This process involves learning a probability distribution over the latent space, which allows for the generation of new data by sampling from this distribution.

**Examples:**

- **Image Generation**: Models like DALL-E, developed by OpenAI, can generate images from textual descriptions, creating visually coherent and contextually appropriate images.
- **Text Generation**: GPT-4 (Generative Pre-trained Transformer 4) is capable of producing human-like text, writing essays, poems, and even computer code based on prompts.
- **Music Generation**: Jukedeck uses AI to compose original music tracks, offering a wide range of styles and moods.

## Large Language Models (LLMs)

Large Language Models (LLMs) are a subset of generative models specifically designed for understanding and generating human language. These models are trained on massive corpora of text data and can perform a wide range of language-related tasks.

LLMs are characterized by their extensive number of parameters, which allow them to capture complex patterns and nuances in language data. These models are typically based on the Transformer architecture, which facilitates parallel processing of input sequences and enhances the model's ability to manage long-range dependencies in text. LLMs are trained using unsupervised learning techniques on diverse and extensive text corpora, enabling them to generate coherent and contextually relevant text across various tasks.

**Examples:**

- **GPT-3 and GPT-4**: Developed by OpenAI, these models can generate human-like text, perform complex language tasks, and even engage in meaningful conversation.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Developed by Google, BERT is particularly effective at understanding the context of words in search queries, improving search engine performance.
- **T5 (Text-to-Text Transfer Transformer)**: Another model by Google, T5 treats every NLP task as a text-to-text problem, making it highly versatile in handling various language-related tasks.

## Relationship between Generative AI and LLMs

Large Language Models (LLMs) are indeed a part of Generative AI. While Generative AI encompasses a broad range of applications, including image, music, and text generation, LLMs focus specifically on the generation and understanding of human language. Both LLMs and other generative models share the underlying principle of learning from data to create new, original content. However, LLMs distinguish themselves by their specialized architecture and training processes tailored for language tasks.

In summary, Generative AI represents the broader category of AI models capable of creating new data, and LLMs are a specialized subset of these models focused on language. Together, they illustrate the remarkable potential of AI to mimic human creativity and understanding across various domains.

## Transformer Architecture

The foundation of most LLMs is the Transformer architecture, introduced by Vaswani et al. in 2017. The Transformer model's key innovation is the self-attention mechanism, which allows it to process input sequences in parallel, rather than sequentially.

### Key Components

- **Encoder-Decoder Structure**: While some models use the full encoder-decoder structure, LLMs like GPT utilize only the decoder part.
- **Self-Attention Mechanism**: Enables the model to focus on different parts of the input sequence when producing an output.
- **Positional Encoding**: Adds information about the position of each token in the sequence.

## Training Process

Training LLMs involves the following steps:

1. **Data Collection**: Gather a large and diverse corpus of text data.
2. **Tokenization**: Split the text into smaller units, called tokens.
3. **Model Initialization**: Initialize the model parameters.
4. **Forward Pass**: Compute the output of the model given an input sequence.
5. **Loss Calculation**: Measure the difference between the predicted output and the actual target using a loss function.
6. **Backward Pass**: Compute the gradients of the loss with respect to the model parameters.
7. **Parameter Update**: Adjust the model parameters using an optimization algorithm.
8. **Iteration**: Repeat the forward and backward passes for multiple epochs until the model converges.

## Key Mathematical Concepts

### Self-Attention

The self-attention mechanism computes a weighted sum of input representations, allowing the model to focus on relevant parts of the input sequence.

Given an input sequence \( X = (x_1, x_2, \ldots, x_n) \):

1. **Query, Key, and Value Matrices**: 
   \[
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V
   \]
   where \( W_Q \), \( W_K \), and \( W_V \) are learned weight matrices.

2. **Attention Scores**: 
   \[
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]
   where \( d_k \) is the dimensionality of the key vectors.

### Positional Encoding

To capture the order of tokens, positional encoding is added to the input embeddings.

For a position \( pos \) and dimension \( i \):
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

### Optimization Algorithms

Common optimization algorithms used to train LLMs include:

- **Stochastic Gradient Descent (SGD)**: Updates parameters using the gradient of the loss with respect to a mini-batch of data.
- **Adam**: An adaptive learning rate optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent.

The parameter update rule for Adam is:
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2
\]
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]
\[
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]
where \( g_t \) is the gradient at time step \( t \), \( \beta_1 \) and \( \beta_2 \) are decay rates, \( \alpha \) is the learning rate, and \( \epsilon \) is a small constant.

## RAG (Retrieval-Augmented Generation)

RAG combines the strengths of retrieval-based and generation-based models to enhance the performance and accuracy of AI systems. This approach involves retrieving relevant documents or information from a knowledge base and using it to generate more accurate and contextually relevant responses.

## Prompt Engineering

Prompt engineering involves designing and optimizing prompts to guide the behavior of language models effectively. This includes crafting precise and clear prompts that yield the desired output and understanding how different phrasing can influence the model's responses.

## Hyperparameter Tuning

Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance. This involves experimenting with different hyperparameter values and selecting the combination that yields the best results.

## Language Change and Multimodal Applications

LLMs are not only used in text generation but also play a significant role in multimodal applications. These applications include generating images, videos, and audio from textual descriptions, expanding the boundaries of creativity and utility of AI.

**Examples:**

- **Text-to-Image**: Models like DALL-E and CLIP, developed by OpenAI, can generate highly detailed and contextually appropriate images from textual descriptions.
- **Text-to-Video**: Emerging models are capable of generating short video clips based on text prompts, paving the way for automated video creation.
- **Text-to-Audio**: Models like Jukedeck and Google's WaveNet can generate music and realistic speech from textual descriptions, transforming the way we produce audio content.

## LangChain

LangChain is a powerful framework designed to facilitate the development of applications that utilize LLMs. It provides tools and abstractions to seamlessly integrate language models into various workflows, enabling efficient and effective use of these models in real-world applications.

**Key Features:**

- **Modular Design**: LangChain's modular architecture allows for flexible integration of different language models and components, making it easy to customize and extend functionalities.
- **Pipeline Support**: Supports the creation of processing pipelines that can handle complex tasks involving multiple steps, such as text generation, translation, and summarization.
- **Ease of Use**: Provides high-level APIs and utilities to simplify the development process, reducing the barrier to entry for leveraging LLMs in your applications.

**Example Use Cases:**

- **Automated Content Creation**: Using LangChain to build systems that generate articles, reports, or marketing content automatically.
- **Chatbots and Virtual Assistants**: Developing conversational agents that can understand and respond to user queries in a natural and engaging manner.
- **Multilingual Applications**: Creating tools that can translate content across multiple languages, making information accessible to a broader audience.

## Applications

LLMs are used in various NLP tasks, including:

- **Text Generation**: Creating coherent and contextually relevant text.
- **Translation**: Translating text from one language to another.
- **Summarization**: Condensing long texts into shorter summaries.
- **Question Answering**: Providing accurate answers to user queries.

## Contribution Guidelines

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) for more details.

## References

- https://www.kaggle.com/code/jayitabhattacharyya/building-llms-from-scratch-generative-ai-report
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/rasbt/LLMs-from-scratch

---
## Reading Materials

-https://github.com/PacktPublishing/LLM-Engineers-Handbook
