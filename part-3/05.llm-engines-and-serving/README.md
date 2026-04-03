# LLM Engines and Serving

An LLM engine is the runtime layer that loads a model, manages memory, tokenizes requests, schedules inference, and returns outputs.

## 1. Why an Engine Is Needed

A raw model checkpoint is not a product-ready system.

A serving engine must handle:

- model loading
- batching
- KV cache management
- request scheduling
- API formatting
- token streaming
- memory optimization

## 2. Common Engine Categories

### Local Desktop / Developer Engines

Examples:

- Ollama
- llama.cpp-based tools

### High-Throughput Server Engines

Examples:

- vLLM
- Text Generation Inference (TGI)

## 3. Ollama

Ollama is designed to make running models locally easier.

It is useful for:

- local experimentation
- small prototypes
- offline demos
- trying open-weight models quickly

Typical strengths:

- simple local workflow
- easy model pulling
- developer-friendly setup

Typical limitations:

- less optimized for large-scale multi-user serving than dedicated server engines

## 4. vLLM

vLLM is a high-performance inference and serving engine widely used for large-model serving.

It is especially known for:

- efficient KV cache handling
- strong throughput
- OpenAI-compatible API serving
- support for offline batch inference and online serving

## 5. Why Engines Matter for Cost

A better serving engine can significantly improve:

- tokens per second
- requests per second
- GPU utilization
- latency under concurrency

This means the same hardware can serve more users at lower cost.

## 6. KV Cache Intuition

During autoregressive generation, the model repeatedly reuses previous attention states. The KV cache stores these states so the model does not recompute everything from scratch.

That improves speed, but the cache also consumes memory.

## 7. Continuous Batching Intuition

A server may receive requests at different times. Instead of waiting for a full batch to form, some engines support continuous batching so requests can be merged dynamically for higher throughput.

## 8. Quantization and Serving

Serving engines often support lower-precision weights such as INT8 or 4-bit formats. This can reduce VRAM use and make local deployment possible, though throughput and quality tradeoffs must be considered.

## 9. Ollama CLI Examples

```bash
ollama pull llama3
ollama run llama3
```

## 10. vLLM Server Example

```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
```

## 11. Python Example: OpenAI-Compatible Call

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# response = client.chat.completions.create(
#     model="your-model-name",
#     messages=[{"role": "user", "content": "Explain LoRA in simple terms."}],
# )
```

## 12. When To Choose Ollama

Choose Ollama when:

- you want local experimentation
- you want a quick local demo
- simplicity matters more than peak throughput

## 13. When To Choose vLLM

Choose vLLM when:

- you need server-grade inference
- concurrent traffic matters
- GPU utilization matters
- you need an OpenAI-compatible serving layer

## 14. Other Serving Considerations

- quantization support
- tensor parallelism
- model download size
- startup time
- context window requirements
- structured output support
- metrics and observability

## 15. Real-World Design Question

A strong serving decision is not just "which model is best?" It is:

- which model fits hardware?
- which engine uses the hardware efficiently?
- which latency and concurrency targets matter?

## Summary

LLM engines are the operational layer between model weights and real applications. Ollama is strong for local development and quick setup, while vLLM is strong for production-style serving and throughput-oriented deployment.
