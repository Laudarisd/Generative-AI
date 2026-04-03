# NLP

Natural Language Processing, or NLP, is the field of building systems that can read, transform, search, classify, generate, and reason over human language.

Modern NLP includes both classic techniques and transformer-era methods. Large language models are part of NLP, but NLP is broader than LLMs alone.

## 1. What NLP Tries to Solve

Human language is difficult for machines because it is:

- ambiguous
- context-dependent
- noisy
- multilingual
- structured at many levels

A single sentence contains:

- characters
- words or subwords
- syntax
- semantics
- pragmatics
- discourse context

NLP tries to model these layers in computational form.

## 2. Common NLP Tasks

### Classification

Assign labels to text.

Examples:

- spam detection
- sentiment analysis
- topic classification

### Sequence Labeling

Predict a label for each token.

Examples:

- named entity recognition (NER)
- part-of-speech tagging

### Sequence-to-Sequence Learning

Map one text sequence to another.

Examples:

- translation
- summarization
- paraphrasing

### Retrieval

Find relevant documents or passages for a query.

### Question Answering

Answer questions using a context passage, a document collection, or a model's internal knowledge.

### Language Generation

Generate coherent text from a prompt.

## 3. Core NLP Pipeline

Although modern systems may be end-to-end, the core conceptual pipeline still looks like:

1. collect text data
2. clean and normalize the text
3. tokenize the text
4. convert text into numerical representations
5. train a model
6. evaluate task-specific metrics
7. deploy and monitor performance

## 4. Text Preprocessing

Common preprocessing choices include:

- lowercasing
- punctuation handling
- stop-word removal
- stemming
- lemmatization
- sentence splitting

These steps matter more for classical pipelines than for large pretrained transformer models, which often prefer minimally altered raw text.

## 5. Text Representations

### Bag of Words

Represents a document using token counts, ignoring order.

### TF-IDF

Represents token importance relative to the corpus.

### Static Embeddings

Examples:

- Word2Vec
- GloVe
- FastText

Each token gets one learned vector.

### Contextual Embeddings

Examples:

- ELMo
- BERT
- RoBERTa

The same token can receive different vectors in different contexts.

## 6. Language Modeling

One of the most important training ideas in NLP is language modeling.

For an autoregressive model:

```math
P(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_{<t})
```

This means the model learns to predict the next token given previous tokens.

Masked language modeling instead hides some tokens and predicts them from context.

## 7. Rule-Based NLP vs Statistical NLP vs Neural NLP

### Rule-Based NLP

Uses hand-written linguistic rules.

Advantages:

- interpretable
- useful in constrained domains

Limitations:

- brittle
- hard to scale

### Statistical NLP

Uses probabilities and feature-based learning.

Examples:

- n-gram language models
- HMMs
- CRFs

### Neural NLP

Uses learned dense representations and deep models.

Examples:

- RNNs
- LSTMs
- transformers

## 8. Important Metrics

Metrics depend on the task.

| Task | Common Metrics |
| --- | --- |
| classification | accuracy, precision, recall, F1 |
| NER / tagging | token F1, span F1 |
| translation | BLEU, COMET |
| summarization | ROUGE |
| generation | perplexity, human evaluation |
| retrieval | recall@k, MRR, nDCG |

## 9. Example: Basic Corpus Statistics

```python
from collections import Counter

texts = [
    "language models learn from language data",
    "modern nlp uses tokenization embeddings and transformers",
]

tokens = " ".join(texts).split()
counts = Counter(tokens)

print("vocab size:", len(counts))
print("most common:", counts.most_common(5))
```

## 10. Example: Named Entity Style Labels

A sequence labeling system may assign tags like:

```text
Sudip     B-PER
works     O
at        O
OpenAI    B-ORG
```

where:

- `B-PER` means beginning of a person entity
- `B-ORG` means beginning of an organization
- `O` means outside any entity

## 11. Example: Minimal Text Preprocessing

```python
import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

sample = "NLP, LLMs, and Transformers are AMAZING!"
print(normalize_text(sample))
```

## 12. How NLP Connects to LLMs

LLMs are a major part of modern NLP, but not every NLP problem requires an LLM.

Smaller approaches are often better when you need:

- low latency
- low cost
- interpretable models
- small on-device deployment
- domain-specific classification

LLMs become attractive when you need:

- open-ended generation
- flexible instruction following
- broad zero-shot capabilities
- strong few-shot transfer

## 13. Practical Mental Model

You can think of NLP as three layers:

1. represent language numerically
2. learn patterns from data
3. solve downstream tasks

Classic NLP focused heavily on engineered features and probabilistic models. Modern NLP places much more of the burden on learned representations and pretraining.

## Summary

NLP is the broader discipline that contains tokenization, statistical language modeling, representation learning, sequence labeling, retrieval, translation, summarization, and LLMs. To understand modern AI systems well, you need both the old NLP toolbox and the new transformer-based view.
