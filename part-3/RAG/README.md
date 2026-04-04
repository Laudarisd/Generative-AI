# RAG

Retrieval-Augmented Generation, or RAG, is one of the most important practical system patterns in modern LLM engineering.

This legacy folder is kept because many readers look for "RAG" directly, but the maintained chapter sequence now lives here:

- [RAG vs Agentic RAG](../03.rag-vs-agentic-rag/README.md)

The maintained chapter now includes:

- chunking strategies and practical defaults
- retrieval terminology glossary
- classic vs agentic design tradeoffs
- hybrid retrieval and reranking examples
- grounded prompt templates with citation style
- retrieval evaluation metrics and failure-mode playbook

Related supporting example:

- [Coding Assistance RAG](coding_assistance_rag/README.md)

## What RAG Means

A RAG system combines two ideas:

1. retrieve relevant external information
2. generate an answer using that retrieved information as context

Instead of relying only on the model's internal parameters, RAG gives the model grounded context from:

- PDFs
- documentation
- knowledge bases
- databases
- web search
- internal company files

## Basic Pipeline

A standard pipeline usually looks like:

1. ingest source documents
2. split them into chunks
3. embed the chunks
4. store them in a vector database
5. embed the user query
6. retrieve top-k chunks
7. place retrieved chunks into the prompt
8. generate an answer

## Why RAG Matters

RAG is useful when:

- your data changes frequently
- your knowledge is private or proprietary
- you need citations or traceability
- you do not want to retrain the whole model

## Practical Weaknesses

RAG can still fail because of:

- bad chunking
- weak embeddings
- poor retrieval quality
- context window overflow
- weak reranking
- irrelevant retrieved passages

## How To Read This Folder

Use the numbered chapter for the full comparison and system discussion:

- [RAG vs Agentic RAG](../03.rag-vs-agentic-rag/README.md)

Use the example folder when you want a more application-oriented case:

- [Coding Assistance RAG](coding_assistance_rag/README.md)

## Summary

This folder now serves as an entry point. The deeper explanation has moved into the structured Part 3 chapter sequence, but the topic remains central because RAG is one of the cleanest ways to make LLMs useful on real private data.
