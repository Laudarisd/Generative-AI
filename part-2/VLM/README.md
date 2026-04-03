# VLM

Vision-language models combine image understanding and language modeling.

## Typical Tasks

- image captioning
- visual question answering
- document understanding
- multimodal chat

## Common Structure

- vision encoder
- connector / projector
- language model

## 1. Why VLMs Matter

Many real-world tasks are not text-only. A useful AI system may need to:

- describe an image
- answer questions about a chart
- read text from a screenshot
- reason over a diagram and a prompt together

VLMs extend language models so they can work with visual inputs.

## 2. High-Level Pipeline

A common VLM pipeline is:

1. image -> vision encoder
2. image features -> projection layer
3. projected features -> language model token space
4. language model processes both visual and textual context
5. output text is generated autoregressively

## 3. Common Design Choices

### Dual Encoder

Image and text are encoded separately, then aligned in a shared embedding space.

Examples:

- CLIP-style systems

### Vision Encoder + LLM

A vision backbone produces image embeddings, and a projector maps them into the token space of a language model.

Examples:

- LLaVA-style systems

### Unified Multimodal Transformer

A single transformer processes mixed visual and textual tokens more directly.

## 4. Important Tasks

### Image Captioning

Generate a textual description of an image.

### Visual Question Answering

Answer questions like:

```text
What color is the car?
```

### OCR and Document AI

Read and reason over scanned pages, receipts, charts, and forms.

### Multimodal Chat

Take both image and text as context in one conversation.

## 5. Alignment Matters

One of the hardest parts of a VLM is alignment: making visual features and textual features live in compatible representational spaces.

Contrastive training is common for this:

```math
\text{similarity}(\text{image}, \text{text})
```

paired examples should score high, while mismatched pairs should score low.

## 6. Tiny PyTorch Example

```python
import torch
import torch.nn as nn

image_feat = torch.randn(4, 128)
text_feat = torch.randn(4, 128)
fusion = nn.Linear(256, 64)

print(fusion(torch.cat([image_feat, text_feat], dim=-1)).shape)
```

## 7. Example: Projecting Vision Features into an LLM

```python
import torch
import torch.nn as nn

batch = 2
num_patches = 8
vision_dim = 1024
llm_dim = 4096

vision_features = torch.randn(batch, num_patches, vision_dim)
projector = nn.Linear(vision_dim, llm_dim)
llm_ready = projector(vision_features)

print(llm_ready.shape)
```

## 8. Challenges in VLMs

- hallucination about visual content
- weak grounding in small details
- OCR failure
- visual reasoning errors
- high compute cost
- benchmark leakage risk

## 9. Evaluation

VLMs are evaluated using:

- caption quality metrics
- VQA accuracy
- retrieval accuracy
- OCR benchmarks
- human evaluation

## 10. Examples of VLM Families

- CLIP
- BLIP / BLIP-2
- Flamingo
- LLaVA
- Qwen-VL
- Gemini multimodal systems

## Summary

VLMs are one of the most important multimodal extensions of the LLM ecosystem.
