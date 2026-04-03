# Part 3: Systems, Adaptation, and Deployment

Part 3 moves from model architecture into real system design.

This part focuses on what happens after you understand transformers and LLM concepts:

- how model parameters affect memory, compute, and deployment
- how fine-tuning works in practice
- how retrieval systems differ from agentic retrieval systems
- how application frameworks such as LangChain help build production pipelines
- how local and server-grade LLM engines such as Ollama and vLLM are used
- how cloud training platforms organize data, jobs, storage, and endpoints
- how MCP servers expose tools, prompts, and resources to model clients
- how multimodal systems combine text, image, audio, and tools in real applications

## Recommended Reading Order

1. [Parameter Understanding](01.parameter-understanding/README.md)
2. [Fine-Tuning and Adaptation](02.fine-tuning-and-adaptation/README.md)
3. [RAG vs Agentic RAG](03.rag-vs-agentic-rag/README.md)
4. [LangChain Applications](04.langchain-applications/README.md)
5. [LLM Engines and Serving](05.llm-engines-and-serving/README.md)
6. [Training in the Cloud](06.training-in-the-cloud/README.md)
7. [MCP Servers and Tools](07.mcp-servers-and-tools/README.md)
8. [Multimodal Applications](08.multimodal-applications/README.md)

## Chapter Map

| Chapter | Main Question | What You Learn |
| --- | --- | --- |
| [Parameter Understanding](01.parameter-understanding/README.md) | Why are modern models so large and expensive? | parameter counts, memory cost, FLOPs intuition, practical examples |
| [Fine-Tuning and Adaptation](02.fine-tuning-and-adaptation/README.md) | How do we adapt base models efficiently? | full fine-tuning, PEFT, LoRA, QLoRA, quantization, training scripts |
| [RAG vs Agentic RAG](03.rag-vs-agentic-rag/README.md) | When is retrieval enough and when do we need agents? | pipeline design, orchestration, evaluation, tool-calling tradeoffs |
| [LangChain Applications](04.langchain-applications/README.md) | How do we build structured LLM apps quickly? | chains, agents, memory, tools, examples |
| [LLM Engines and Serving](05.llm-engines-and-serving/README.md) | How are models actually run locally or at scale? | Ollama, vLLM, batching, KV cache, OpenAI-compatible APIs |
| [Training in the Cloud](06.training-in-the-cloud/README.md) | How do teams train and deploy on cloud platforms? | AWS, Vertex AI, Azure ML, storage, jobs, endpoints |
| [MCP Servers and Tools](07.mcp-servers-and-tools/README.md) | How do tools connect to model clients? | JSON-RPC, tool schemas, resources, prompts, server examples |
| [Multimodal Applications](08.multimodal-applications/README.md) | How are real systems built across modalities? | image-text systems, voice systems, OCR, workflows, deployment |

## Theme of This Part

Part 1 explained ideas. Part 2 explained models. Part 3 explains systems.

That means the focus shifts from "what is attention?" to questions such as:

- how many GPUs will this require?
- should we fine-tune or use retrieval?
- should we deploy with Ollama or vLLM?
- when is an agent useful and when is it just extra complexity?
- how do tools and model clients actually communicate?

## Related Legacy Folders

The older folders in this part are kept for compatibility and now point into the chapter sequence:

- [RAG](RAG/README.md)
- [LangChain](LangChain/README.md)
- [Multimodel Application](Multimodel_Application/README.md)
- [Hyperparameter Tuning](Hyperparameter_Tuning/README.md)
