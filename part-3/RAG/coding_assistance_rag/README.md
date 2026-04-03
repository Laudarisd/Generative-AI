# Coding Assistance RAG

This folder is kept as a supporting example under Part 3 because coding assistance is one of the clearest real-world use cases for retrieval-augmented generation.

Main related chapter:

- [RAG vs Agentic RAG](../../03.rag-vs-agentic-rag/README.md)

## 1. Why Coding Assistance Needs RAG

A coding assistant often needs knowledge that is not safe to leave purely to model memory:

- repository-specific code
- internal APIs
- architecture decisions
- style guides
- dependency versions
- documentation pages

A plain base model may know Python broadly, but it does not automatically know your repository.

## 2. What a Coding RAG System Retrieves

A coding-focused retriever may search over:

- README files
- source code
- docstrings
- API docs
- architecture notes
- issue threads
- test files

## 3. Typical Pipeline

1. load code and docs from the project
2. split them into retrievable chunks
3. embed those chunks
4. store embeddings in a vector database
5. retrieve context based on the user question
6. ask the model to answer using that code-aware context

## 4. Example User Questions

- Where is authentication handled?
- Which function writes metrics to the database?
- Show an example of how the API client is used.
- Why is this test failing?

## 5. Important Design Choices

### Chunking

Chunking source code is different from chunking prose. You usually want chunks aligned to:

- functions
- classes
- modules
- markdown sections

### Metadata

Useful metadata includes:

- file path
- symbol name
- language
- repository branch
- line or section references

### Retrieval Strategy

For code, hybrid retrieval is often useful:

- vector search for semantic similarity
- keyword search for exact identifiers

## 6. Tiny Example: Code Chunk Objects

```python
chunks = [
    {
        "path": "src/auth.py",
        "symbol": "login_user",
        "content": "def login_user(username, password): ...",
    },
    {
        "path": "src/db.py",
        "symbol": "save_metrics",
        "content": "def save_metrics(row): ...",
    },
]

for chunk in chunks:
    print(chunk["path"], chunk["symbol"])
```

## 7. Prompt Construction Example

```python
def build_code_rag_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        f"FILE: {c['path']}\nSYMBOL: {c['symbol']}\n{c['content']}" for c in retrieved_chunks
    )
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer using the retrieved code only."

print(build_code_rag_prompt("Where are metrics saved?", chunks[:1]))
```

## 8. Why This Can Beat Plain Prompting

A coding assistant with retrieval can:

- cite real files
- reduce hallucinated APIs
- answer repository-specific questions
- remain useful as the codebase changes

## 9. Limitations

It still needs:

- good indexing
- fresh repository snapshots
- careful handling of large files
- safe access controls for private code

## Summary

Coding assistance RAG is a concrete example of why retrieval matters. It grounds the model in the actual repository rather than asking it to guess from generic pretraining alone.
