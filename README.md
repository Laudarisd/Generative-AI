# Era of Generative AI

A book-style repository for learning:

- AI and machine learning fundamentals
- LLMs and transformer concepts
- a standalone mathematics reference
- generative model families
- practical systems such as RAG, LangChain, and multimodal applications
- scientific and advanced AI topics such as PINNs, Bayesian ML, energy, and remote sensing

---

## Learning Path

This repository is organized in the order most readers actually need:

1. learn the fundamentals
2. build the required mathematics reference
3. understand how LLMs fit into generative AI
4. study transformer and model architecture topics
5. move into practical systems and applications
6. move into scientific and advanced AI topics

---

## Book Map

### Pre-Chapter: Fundamental Concepts

This section builds the base for the rest of the repository.

| Chapter | Topic                                            | Link                                                                             |
| ------- | ------------------------------------------------ | -------------------------------------------------------------------------------- |
| 1       | Statistics                                       | [Open](fundamental-concepts/1.statistics/README.md)                                 |
| 2       | Data Science                                     | [Open](fundamental-concepts/2.data-science/README.md)                               |
| 3       | Machine Learning and AI                          | [Open](fundamental-concepts/3.ml-and-ai/README.md)                                  |
| 4       | Generative AI                                    | [Open](fundamental-concepts/4.generative-ai/README.md)                              |
| 5       | Loss Functions, Optimization, and Regularization | [Open](fundamental-concepts/5.loss-functions-optimization-regularization/README.md) |
| 6       | Bayesian Thinking                                | [Open](fundamental-concepts/6.bayesian-thinking/README.md)                           |

What you learn here:

- probability and statistics basics
- data workflows and feature thinking
- supervised, unsupervised, semi-supervised, and reinforcement learning
- core algorithms and evaluation
- how losses, optimization, and regularization work
- Bayesian reasoning and uncertainty in machine learning

---

## Mathematics Reference

This section rebuilds the old mathematical cheat sheets as a dedicated standalone reference.

| Chapter | Topic | Link |
| --- | --- | --- |
| 0 | Algebra Foundations | [Open](math-cheatsheet/0.algebra-foundations/README.md) |
| 1 | Calculus | [Open](math-cheatsheet/1.calculus/README.md) |
| 2 | Linear Algebra | [Open](math-cheatsheet/2.linear-algebra/README.md) |
| 3 | Matrix Theory | [Open](math-cheatsheet/3.matrix-theory/README.md) |
| 4 | Real Analysis | [Open](math-cheatsheet/4.real-analysis/README.md) |
| 5 | Discrete Mathematics | [Open](math-cheatsheet/5.discrete-mathematics/README.md) |
| 6 | Optimization and Losses | [Open](math-cheatsheet/6.optimization-and-losses/README.md) |
| 7 | Machine Learning Mathematics | [Open](math-cheatsheet/7.machine-learning-mathematics/README.md) |
| 8 | Engineering Mathematics | [Open](math-cheatsheet/8.engineering-mathematics/README.md) |

Math index:

- [Math Cheat Sheet Overview](math-cheatsheet/README.md)

What you learn here:

- school-to-bachelor algebra foundations
- derivatives, gradients, Jacobians, and Hessians
- matrix decompositions, projections, rank, and conditioning
- optimization, losses, activations, and training math
- attention, embeddings, sequence probability, and transformer formulas
- engineering math for ODEs, PDEs, transforms, and numerical methods

---

## Part 1: Foundations for LLMs

This part explains the concepts behind language models in a cleaner chapter sequence.

| Chapter | Topic                      | Link                                                |
| ------- | -------------------------- | --------------------------------------------------- |
| 1       | LLM History                | [Open](part-1/01.llm-history/README.md)                |
| 2       | LLM vs Generative AI       | [Open](part-1/02.llm-vs-generative-ai/README.md)       |
| 3       | Understanding LLMs         | [Open](part-1/03.understanding-llms/README.md)         |
| 4       | Tokenization vs Embeddings | [Open](part-1/04.tokenization-vs-embeddings/README.md) |
| 5       | Encoding vs Decoding       | [Open](part-1/05.encoding-vs-decoding/README.md)       |
| 6       | Positional Encoding        | [Open](part-1/06.positional-encoding/README.md)        |

Part 1 index:

- [Part 1 Overview](part-1/README.md)

What you learn here:

- where LLMs came from
- how LLMs differ from broader generative AI
- how tokenization, embeddings, encoding, decoding, and positional encoding work
- how transformer-based language systems are organized

---

## Part 2: Models and Architectures

This part moves from concepts into model families and transformer building blocks.

### LLMs

- [LLMs Overview](part-2/LLMs/README.md)
- [Chatbot with LLM](part-2/LLMs/chatbot_with_LLM/README.md)

### Transformer Components

- [Attention Mechanisms](part-2/Transformer/Attention_Mechanisms/README.md)
- [Encoder-Decoder Structure](part-2/Transformer/Encoder_Decoder_Structure/README.md)
- [Feed Forward Networks](part-2/Transformer/Feed_Forward_Networks/README.md)
- [Layer Normalization and Residual Connection](part-2/Transformer/Layer_Normalization_and_Residual_Connection/README.md)
- [Positional Encoding](part-2/Transformer/Positional_Encoding/README.md)

### Generative Model Families

- [GAN](part-2/Generative_AI/GAN/README.md)
- [VAE](part-2/Generative_AI/VAEs/README.md)
- [Diffusion Models](part-2/Generative_AI/Diffusion_models/README.md)

### Prompting

- [Prompt Engineering](part-2/Prompt_Engineering/README.md)

---

## Part 3: Applications and Systems

This part is more practical and system-oriented.

### Retrieval and Knowledge

- [RAG](part-3/RAG/README.md)
- [Coding Assistance RAG](part-3/RAG/coding_assistance_rag/README.md)

### Frameworks and Apps

- [LangChain](part-3/LangChain/README.md)
- [Multimodel Application](part-3/Multimodel_Application/README.md)

### Training and Tuning

- [Hyperparameter Tuning](part-3/Hyperparameter_Tuning/README.md)

---

## Part 4: Scientific and Advanced AI

This part extends the repo toward mathematically heavier and domain-grounded AI.

| Chapter | Topic | Link |
| --- | --- | --- |
| 1 | Physics-Informed Neural Networks | [Open](part-4/01.physics-informed-neural-networks/README.md) |
| 2 | Bayesian Machine Learning | [Open](part-4/02.bayesian-machine-learning/README.md) |
| 3 | Energy and AI | [Open](part-4/03.energy-and-ai/README.md) |
| 4 | Remote Sensing and AI | [Open](part-4/04.remote-sensing-and-ai/README.md) |
| 5 | Scientific AI with Mathematics | [Open](part-4/05.scientific-ai-with-mathematics/README.md) |

Part 4 index:

- [Part 4 Overview](part-4/README.md)

What you learn here:

- how physical laws can be embedded into learning systems
- how uncertainty-aware Bayesian methods support science and engineering
- how AI interacts with energy systems and compute efficiency
- how geospatial and remote sensing AI differs from ordinary vision tasks
- how mathematics supports scientific modeling, simulation, and surrogate learning

---

## Paper Notes

Research summaries and reading notes:

- [Attention Is All You Need](papers/Attention%20is%20All%20You%20Need.md)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](papers/An%20Image%20is%20Worth%2016x16%20Words%20Transformers%20for%20Image%20Recognition%20at%20Scale.md)

---

## Recommended Reading Order

If you want the cleanest path through the repo:

1. [Statistics](fundamental-concepts/1.statistics/README.md)
2. [Data Science](fundamental-concepts/2.data-science/README.md)
3. [Machine Learning and AI](fundamental-concepts/3.ml-and-ai/README.md)
4. [Generative AI](fundamental-concepts/4.generative-ai/README.md)
5. [Loss Functions, Optimization, and Regularization](fundamental-concepts/5.loss-functions-optimization-regularization/README.md)
6. [Bayesian Thinking](fundamental-concepts/6.bayesian-thinking/README.md)
7. [Math Cheat Sheet Overview](math-cheatsheet/README.md)
8. [Algebra Foundations](math-cheatsheet/0.algebra-foundations/README.md)
9. [Calculus](math-cheatsheet/1.calculus/README.md)
10. [LLM History](part-1/01.llm-history/README.md)
11. [LLM vs Generative AI](part-1/02.llm-vs-generative-ai/README.md)
12. [Understanding LLMs](part-1/03.understanding-llms/README.md)
13. [Tokenization vs Embeddings](part-1/04.tokenization-vs-embeddings/README.md)
14. [Encoding vs Decoding](part-1/05.encoding-vs-decoding/README.md)
15. [Positional Encoding](part-1/06.positional-encoding/README.md)
16. Part 2 model and transformer chapters
17. Part 3 system and application chapters
18. [Part 4 Overview](part-4/README.md)

---

## What This Repo Is Trying To Be

Not just a note dump.

The target shape is:

- a guided book for learning generative AI
- a reusable mathematics reference
- a concept reference for LLMs and transformers
- a practical repository for model systems and applications
- a bridge between mathematics, architecture, implementation, and scientific AI

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Physics-Informed Neural Networks](https://maziarraissi.github.io/PINNs/)
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/)
- [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/)
- [Energy and AI (IEA)](https://www.iea.org/reports/energy-and-ai)
- [NASA Earthdata Remote Sensing Backgrounder](https://www.earthdata.nasa.gov/learn/backgrounders/remote-sensing)
- [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)
- [LLM Engineers Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
