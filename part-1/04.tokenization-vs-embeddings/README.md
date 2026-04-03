# Tokenization vs Embeddings

## Why These Two Are Easy To Confuse

They appear next to each other in every LLM pipeline, but they are not the same thing.

- **Tokenization** turns raw text into discrete units
- **Embeddings** turn those units into dense vectors

---

## 1. Tokenization

Example sentence:

```text
Large language models are useful.
```

One tokenizer might produce:

```text
["Large", " language", " models", " are", " useful", "."]
```

### Why Tokenization Exists

- computers need a finite vocabulary
- rare words must still be handled
- token counts must stay manageable

### Python Example

```python
text = "Large language models are useful."
tokens = text.split()
print(tokens)
```

This is only a toy whitespace tokenizer, but it shows the basic idea.

---

## 2. Embeddings

After tokenization, each token id is mapped to a vector:

```math
e_i = E[t_i]
```

where:

- $t_i$ is the token id
- $E$ is the embedding matrix

If:

```math
E \in \mathbb{R}^{|\mathcal{V}| \times d_{model}}
```

then each token gets a vector of size $d_{model}$.

### Why Embeddings Matter

Embeddings let the model work in continuous space instead of with raw integer ids.

This is where linear algebra becomes essential.

If:

```math
X \in \mathbb{R}^{n \times d_{model}}
```

then:

- $n$ is the sequence length
- $d_{model}$ is the embedding dimension
- each token becomes one row in the matrix

That matrix is the actual object the transformer processes.

### Python Example

```python
import numpy as np

embedding_table = {
    "cat": np.array([0.2, 0.7, 0.1]),
    "dog": np.array([0.3, 0.6, 0.2]),
}

print(embedding_table["cat"])
```

---

## 3. Difference Summary

| Topic | Tokenization | Embeddings |
| --- | --- | --- |
| Input | raw text | token ids |
| Output | tokens / token ids | dense vectors |
| Main role | segment text | represent tokens numerically |

### Practical Intuition

Tokenization answers:

- what pieces should this text be split into?

Embeddings answer:

- how should each piece be represented for the neural network?

### Short Example

`cat` and `dog` are different tokens, but their embeddings may end up relatively close if the model learns they often appear in similar contexts.
