# LLMs Overview

This chapter gives a model-family view of large language models inside Part 2.

Part 1 explained what LLMs are conceptually. This chapter focuses more on the ecosystem and practical architecture landscape.

## 1. What Makes a Model an LLM

An LLM is usually:

- large-scale
- trained on language or code
- built on neural sequence modeling
- commonly transformer-based

## 2. Major Architecture Styles

| Style | Main Strength | Example |
| --- | --- | --- |
| Encoder-only | understanding and embeddings | BERT |
| Decoder-only | generation and chat | GPT, Llama |
| Encoder-decoder | source-to-target tasks | T5, BART |

### Why this distinction matters

The architecture style influences:

- what objective is used during pretraining
- what downstream tasks are natural
- how inference works
- whether the model is better for embeddings, generation, or transformation

## 3. Why Decoder-Only Models Dominated Chat

Decoder-only models are especially natural for:

- text continuation
- code completion
- dialogue
- instruction following

That made them dominant in assistant-style products.

Their autoregressive objective matches the user experience of chat:

```text
prompt -> next token -> next token -> next token
```

## 4. Model Family Examples

- GPT family
- Llama
- Qwen
- Mistral
- T5
- BERT / RoBERTa

## 5. Open vs Closed Models

Some models are:

- open-weight
- API-only
- hybrid ecosystems with both open and closed variants

This affects:

- deployment
- customization
- cost
- privacy

## 6. Important Training Stages

Many real-world LLM systems involve multiple stages:

1. tokenizer design
2. large-scale pretraining
3. supervised fine-tuning or instruction tuning
4. preference alignment
5. evaluation and safety tuning
6. inference optimization

## 7. Important Deployment Questions

When choosing or comparing LLMs, engineers should think about:

- context length
- model size
- multilingual quality
- code capability
- latency
- memory use
- open-weight availability
- quantization support
- fine-tuning support

## 8. Mini Architecture Example

```python
import torch
import torch.nn as nn

vocab_size = 32000
d_model = 256

embedding = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)

tokens = torch.randint(0, vocab_size, (2, 5))
hidden = embedding(tokens)
logits = lm_head(hidden)

print("hidden:", hidden.shape)
print("logits:", logits.shape)
```

## 9. Practical Comparison

| Goal | Common Choice |
| --- | --- |
| embeddings / retrieval | encoder or embedding models |
| chat assistant | decoder-only LLM |
| translation / summarization | encoder-decoder |
| multimodal assistant | VLM or multimodal LLM |

## 10. Related Subprojects in This Folder

- [chatbot_with_LLM](chatbot_with_LLM/README.md)
- `LLM_codegenerator/`
- `Project-1.Large_Language_Model_from_Scratch/`

## 11. How To Read This Folder

This overview should be read together with:

- the transformer subchapters for architecture details
- prompt engineering for inference-time control
- VLM for multimodal extensions

## Summary

The LLM ecosystem is broader than one model brand. Understanding architecture style, openness, and intended use is more useful than memorizing names alone.
