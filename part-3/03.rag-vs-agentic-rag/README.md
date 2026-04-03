# RAG vs Agentic RAG

Retrieval-Augmented Generation, or RAG, augments a language model with external information retrieval. Agentic RAG extends that idea by allowing the system to plan, choose tools, refine queries, and sometimes iterate before answering.

## 1. Classic RAG

A classic RAG pipeline usually looks like:

1. user asks a question
2. query is embedded
3. top-k relevant chunks are retrieved
4. retrieved context is inserted into the prompt
5. the model generates an answer

## 2. Core RAG Formula Intuition

A language model conditions on both the question $q$ and retrieved context $d$:

```math
P(y \mid q, d)
```

The quality of the final answer depends heavily on the quality of $d$.

## 3. RAG Components

- document ingestion
- chunking
- embeddings
- vector database
- retrieval
- reranking
- prompt augmentation
- generation
- evaluation

## 4. Why RAG Is Useful

- grounds answers in external documents
- handles private knowledge
- handles changing data better than static weights
- can provide citations or references

## 5. Limits of Classic RAG

Classic RAG is often not enough when:

- the query is ambiguous
- one retrieval step is insufficient
- tools beyond retrieval are required
- multiple sub-questions must be solved
- document coverage is weak

## 6. What Agentic RAG Adds

Agentic RAG may add:

- query rewriting
- decomposition into sub-questions
- tool calling
- web fallback
- multiple retrieval passes
- reasoning over intermediate results
- iterative answer refinement

## 7. Simple Comparison

| Aspect | Classic RAG | Agentic RAG |
| --- | --- | --- |
| Retrieval passes | usually one | one or many |
| Tool use | minimal | common |
| Planning | little or none | explicit or implicit |
| Complexity | lower | higher |
| Latency | lower | often higher |
| Reliability | easier to control | more moving parts |

## 8. When Classic RAG Is Enough

Use classic RAG when:

- the corpus is well-structured
- questions are straightforward
- latency matters a lot
- compliance and predictability matter

## 9. When Agentic RAG Makes Sense

Use agentic RAG when:

- tasks need multiple steps
- retrieval needs refinement
- you need calculators, search, SQL, or APIs
- workflow orchestration matters

## 10. Minimal Retrieval Example

```python
chunks = [
    "Transformers use self-attention.",
    "RAG combines retrieval with generation.",
    "LoRA is a PEFT technique.",
]
query = "What is RAG?"

scores = [chunk.lower().count("rag") for chunk in chunks]
ranked = sorted(zip(scores, chunks), reverse=True)
print(ranked[:2])
```

## 11. Agentic Pattern Example

```python
def agentic_rag(question):
    if "latest" in question.lower():
        return "Use retrieval plus web search tool"
    if "compare" in question.lower():
        return "Decompose into sub-questions, retrieve separately, then synthesize"
    return "Single retrieval pipeline is enough"

print(agentic_rag("Compare LoRA and QLoRA"))
```

## 12. Evaluation Dimensions

Evaluate both systems on:

- retrieval relevance
- answer correctness
- citation faithfulness
- latency
- cost
- tool-call success rate
- error recovery behavior

## 13. Practical Guidance

Start simple.

A common engineering mistake is to jump into agents before proving that plain retrieval, reranking, and prompt design are insufficient.

## 14. Coding Assistance RAG

A specialized example is available here:

- [Coding Assistance RAG](../RAG/coding_assistance_rag/README.md)

## Summary

RAG grounds generation with external knowledge. Agentic RAG adds planning and tool use when one retrieval step is not enough. In practice, simple RAG is often the correct first system, while agentic RAG is a second-stage design when task complexity justifies it.
