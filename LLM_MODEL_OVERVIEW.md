# 🧠 Transformer and LLM Model Overview

This document summarizes major transformer-based models, categorized by architecture (Encoder, Decoder, Encoder-Decoder), with use cases, licensing info, and source links.

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

## 🎨 Diffusion Models (for Image, Texture, 3D Generation)

| Model                | Type           | Use Case                                   | Free | Source                                                               |
| -------------------- | -------------- | ------------------------------------------ | ---- | -------------------------------------------------------------------- |
| Stable Diffusion 1.5 | Text-to-Image  | General-purpose image generation           | ✅   | [SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)         |
| Stable Diffusion XL  | Text-to-Image  | High-res image generation                  | ✅   | [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| ControlNet           | Image-to-Image | Conditioned generation (pose, depth, edge) | ✅   | [ControlNet](https://huggingface.co/lllyasviel/ControlNet)              |
| Kandinsky 2.2        | Text-to-Image  | Multilingual + multimodal generation       | ✅   | [Kandinsky](https://huggingface.co/kandinsky-community)                 |
| DeepFloyd IF         | Text-to-Image  | High-quality, staged generation            | ✅   | [DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)           |
| DreamBooth           | Fine-tuning    | Personalized image generation              | ✅   | [DreamBooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion) |
| Textual Inversion    | Fine-tuning    | Learn custom styles from images            | ✅   | [Textual Inversion](https://github.com/rinongal/textual_inversion)      |
| Latent Consistency   | Text-to-Image  | Fast, low-step generation                  | ✅   | [LCM](https://github.com/luo3300612/latent-consistency-model)           |
| RePaint              | Inpainting     | Image inpainting with diffusion            | ✅   | [RePaint](https://github.com/andreas128/RePaint)                        |
| Zero123              | Text/Img-to-3D | View synthesis, object rotation            | ✅   | [Zero123](https://github.com/autonomousvision/zero123)                  |
| DreamGaussian        | Image-to-3D    | Gaussian 3D asset creation from images     | ✅   | [DreamGaussian](https://github.com/USTC3DV/DreamGaussian)               |
| Shap-E               | Text-to-3D     | OpenAI’s 3D asset generator               | ✅   | [Shap-E](https://github.com/openai/shap-e)                              |

---

## 🧠 When to Use Diffusion Models

| Task                           | Recommended Models             |
| ------------------------------ | ------------------------------ |
| Realistic image generation     | Stable Diffusion, SDXL         |
| Custom texture/CMF generation  | ControlNet, DreamBooth, LCM    |
| Inpainting/editing             | RePaint, ControlNet            |
| Multilingual generation        | Kandinsky, DeepFloyd IF        |
| Fast generation (low step)     | LCM, Stable Diffusion 1.5      |
| Image-to-3D or text-to-3D      | Shap-E, Zero123, DreamGaussian |
| Style transfer/personalization | DreamBooth, Textual Inversion  |

---
