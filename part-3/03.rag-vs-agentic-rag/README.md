# RAG vs Agentic RAG

Retrieval-Augmented Generation (RAG) grounds LLM output using external data retrieval.
Agentic RAG extends this with planning, tool-use, and iterative retrieval when one pass is not enough.

---

## 1) Quick Answer: Is Chunking Included?

Yes, chunking is included in RAG.
It is one of the highest-impact design choices in retrieval quality.
Poor chunking can break otherwise good embeddings, retrievers, and prompts.

---

## 2) Classic RAG Pipeline

1. ingest source documents
2. clean and normalize text/code
3. split into chunks
4. create embeddings
5. index in vector database (optionally with keyword index)
6. embed user query
7. retrieve top-k candidates
8. rerank candidates (optional but recommended)
9. build grounded prompt with citations
10. generate answer
11. evaluate and monitor

Core intuition:

```math
P(y \mid q, d)
```

where `q` is the query and `d` is retrieved evidence.

---

## 3) Chunking Deep Dive

### Why chunking matters
- Too small: loses context and semantics.
- Too large: adds noise and wastes context window.
- Bad boundaries: split important logic or definitions.

### Common chunking strategies

1. Fixed-size chunking
- split by token count (for example 300-800 tokens)
- simple and fast baseline

2. Sliding-window chunking
- fixed size plus overlap (for example 20-30%)
- preserves continuity across boundaries

3. Recursive chunking
- split by natural structure first (heading, paragraph, sentence)
- fallback to token-length constraints

4. Semantic chunking
- split where meaning changes (embedding-based boundaries)
- higher quality, usually more expensive

5. Structure-aware chunking
- for markdown/docs: section-based
- for code: function/class/module boundaries
- for tables: row/column-preserving policies

### Practical chunk-size defaults
- knowledge docs: 400-800 tokens, 10-20% overlap
- API references: 200-500 tokens, low overlap
- source code: symbol-level chunks plus file metadata
- transcripts/logs: turn/window chunks with timestamps

---

## 4) Retrieval Terminology (Must Know)

- Corpus: all source documents
- Chunk: retrievable unit
- Embedding: dense vector representation
- Indexing: building searchable structure
- ANN: approximate nearest neighbor search
- Top-k retrieval: return k nearest chunks
- Similarity metric: cosine / dot product / L2
- Hybrid retrieval: vector + keyword/BM25
- Reranking: second-stage relevance sorting
- Query rewriting: improve retrieval query form
- Recall@k: whether relevant item appears in top-k
- Precision@k: fraction of top-k that are relevant
- nDCG: ranking quality with position-sensitive gain
- MRR: reciprocal-rank metric
- Groundedness: answer supported by retrieved evidence
- Citation faithfulness: citation actually supports claim
- Context packing: selecting/ordering chunks for prompt
- Hallucination: unsupported generated content
- Drift: corpus/distribution changes over time

---

## 5) Classic RAG vs Agentic RAG

| Aspect | Classic RAG | Agentic RAG |
| --- | --- | --- |
| Retrieval passes | Usually one | One or multiple |
| Tool use | Minimal | Frequent |
| Planning | Minimal | Explicit/implicit |
| Latency | Lower | Higher |
| Cost | Lower | Higher |
| Failure recovery | Limited | Better with retries/strategies |
| Operational complexity | Lower | Higher |

Use classic RAG first. Move to agentic RAG only when clear evidence shows single-pass retrieval is insufficient.

---

## 6) Practical Design Patterns

### Pattern A: High-precision FAQ assistant
- small trusted corpus
- strict retrieval threshold
- concise citations

### Pattern B: Enterprise knowledge copilot
- hybrid retrieval
- reranking
- metadata filtering (team, product, date)

### Pattern C: Code assistant RAG
- symbol-aware chunking
- file-path metadata
- exact identifier fallback (keyword search)

### Pattern D: Agentic troubleshooting assistant
- decomposition into sub-questions
- tool chain: docs + logs + metrics + SQL
- iterative retrieval and synthesis

---

## 7) Code Examples

### Example 1: Fixed-size chunking with overlap (simple)

```python
def chunk_text(text, size=500, overlap=100):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks
```

### Example 2: Structure-aware chunking for markdown headings

```python
import re

def split_markdown_by_heading(md_text):
    parts = re.split(r"\n(?=#{1,3}\s)", md_text)
    return [p.strip() for p in parts if p.strip()]
```

### Example 3: Hybrid retrieval score fusion (toy)

```python
# vec_score: cosine similarity in [0,1]
# kw_score: keyword/BM25 score normalized to [0,1]
def hybrid_score(vec_score, kw_score, alpha=0.7):
    return alpha * vec_score + (1 - alpha) * kw_score
```

### Example 4: Lightweight reranking prompt

```python
def rerank_prompt(query, passages):
    joined = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return (
        "Rank passages by relevance to query. Return only ids in order.\n"
        f"Query: {query}\n\nPassages:\n{joined}"
    )
```

### Example 5: Grounded answer prompt template

```python
def build_grounded_prompt(question, contexts):
    ctx = "\n\n".join(f"[C{i+1}] {c}" for i, c in enumerate(contexts))
    return f"""
You are a grounded assistant.
Use only the provided context.
If evidence is missing, say "I don't have enough evidence.".
Cite sources as [C1], [C2], ...

Question: {question}

Context:
{ctx}
""".strip()
```

### Example 6: Retrieval evaluation skeleton

```python
# qrels: dict[query_id] -> set(relevant_chunk_ids)
# preds: dict[query_id] -> list(retrieved_chunk_ids)
def recall_at_k(qrels, preds, k=5):
    vals = []
    for qid, rel in qrels.items():
        topk = set(preds.get(qid, [])[:k])
        vals.append(1.0 if len(rel & topk) > 0 else 0.0)
    return sum(vals) / max(1, len(vals))
```

---

## 8) Evaluation: What to Measure

### Retrieval metrics
- Recall@k
- Precision@k
- MRR
- nDCG

### Generation metrics
- factual correctness
- citation faithfulness
- groundedness
- answer completeness

### System metrics
- latency (p50/p95)
- cost per query
- timeout/error rates
- tool-call success rate (agentic)

---

## 9) Common Failure Modes and Fixes

1. Relevant info not retrieved
- improve chunking, query rewriting, hybrid retrieval, metadata filters

2. Retrieved context is noisy
- rerank, stricter top-k, thresholding, better indexing quality

3. Hallucinated answer despite good retrieval
- stronger grounded prompt, citation enforcement, answer verifier step

4. Context overflow
- context packing policy, compression, map-reduce summarization

5. Stale knowledge
- refresh indexing schedule, versioned corpus, cache invalidation policy

---

## 10) Minimal Agentic RAG Decision Flow

```python
def choose_rag_mode(question):
    q = question.lower()
    if any(k in q for k in ["compare", "multi-step", "investigate", "latest"]):
        return "agentic_rag"
    return "classic_rag"
```

Use this as a starting heuristic, then replace with telemetry-driven routing rules.

---

## 11) When to Stop Adding Complexity

Move from classic -> advanced only when metrics justify it:
- retrieval recall plateau with single-pass pipeline
- unresolved query classes requiring decomposition/tooling
- acceptable business value from higher latency/cost

---

## 12) Related Chapters

- [RAG Entry Page](../RAG/README.md)
- [Coding Assistance RAG](../RAG/coding_assistance_rag/README.md)
- [LLM Engines and Serving](../05.llm-engines-and-serving/README.md)
- [LangChain Applications](../04.langchain-applications/README.md)

## Summary

Chunking is not optional in RAG; it is foundational.
Start with classic RAG + good chunking + hybrid retrieval + reranking + grounded prompting.
Add agentic loops only when complexity is justified by measurable gains.
