# Tokenization vs Embeddings

## Why These Two Are Easy To Confuse

They appear next to each other in every LLM pipeline, but they are not the same thing.

- **Tokenization** turns raw text into discrete units
- **Embeddings** turn those units into dense vectors

You can think of tokenization as segmentation and indexing, and embeddings as learned representation.

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

Another tokenizer might split differently.

That matters because tokenization is not universal. Different model families use different tokenizers.

---

## 2. Why Tokenization Exists

Raw text is not directly usable by the neural network.

We need:

- a finite vocabulary
- a repeatable mapping from text to ids
- a way to handle rare and unseen words
- token counts that stay computationally manageable

### Why Not Just Use Words?

If the vocabulary were only full words:

- rare words would explode vocabulary size
- new words would be hard to handle
- multilingual and code-heavy text would become awkward

That is why subword tokenization became dominant.

---

## 3. Common Tokenization Styles

### Word-Level

Split by words.

Simple, but too rigid for modern LLMs.

### Character-Level

Split into characters.

Flexible, but sequences become too long.

### Subword-Level

Split into reusable word pieces.

This is the most common approach in modern LLMs.

Popular algorithms include:

- BPE
- WordPiece
- unigram tokenization

---

## 4. Token IDs

After tokenization, tokens are mapped to integer ids.

Example:

```text
["Large", " language", " models"]
```

might become:

```text
[1042, 8821, 5510]
```

These ids are not meaningful numerically by themselves. They are just lookup keys.

---

## 5. Tiny Python Example: Toy Tokenization

```python
text = "Large language models are useful."
tokens = text.split()
token_to_id = {token: i for i, token in enumerate(sorted(set(tokens)))}
ids = [token_to_id[t] for t in tokens]

print(tokens)
print(ids)
```

This is only a toy whitespace tokenizer, but it shows the basic idea.

---

## 6. Embeddings

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

---

## 7. Why Embeddings Matter

Embeddings let the model work in continuous space instead of raw integer ids.

That matters because continuous vectors can express:

- similarity
- direction
- learned structure
- geometry useful for neural computation

This is where linear algebra becomes essential.

If:

```math
X \in \mathbb{R}^{n \times d_{model}}
```

then:

- $n$ is the sequence length
- $d_{model}$ is the embedding dimension
- each token becomes one row in the matrix

That matrix is what the transformer actually processes.

---

## 8. Input Embeddings vs Meaning

A beginner mistake is to think embeddings are literal dictionary meanings.

They are not.

Embeddings are learned vectors that become useful because they help the model predict or solve tasks.

That means:

- some semantic structure appears
- some syntactic structure appears
- some positional or task-specific behavior also appears

They are functional learned representations, not human definitions.

---

## 9. Contextual vs Non-Contextual Representations

Token embeddings at the input layer are usually static lookups.

But once the transformer processes them, the representations become contextual.

Example:

The word `bank` in:

- `river bank`
- `bank account`

may start from one token entry, but after contextual processing the internal representations differ.

This is one reason people sometimes loosely say "embeddings" to mean several different levels of representation.

---

## 10. Tiny Python Example: Toy Embedding Table

```python
import numpy as np

embedding_table = {
    "cat": np.array([0.2, 0.7, 0.1]),
    "dog": np.array([0.3, 0.6, 0.2]),
    "car": np.array([0.9, 0.1, 0.4]),
}

print(embedding_table["cat"])
```

---

## 11. Similarity Intuition

`cat` and `dog` are different tokens, but their embeddings may end up relatively close if the model learns they often appear in similar contexts.

That is one of the big conceptual wins of embeddings:

- raw symbols become geometry

### Tiny Cosine Similarity Example

```python
import numpy as np

cat = np.array([0.2, 0.7, 0.1])
dog = np.array([0.3, 0.6, 0.2])

sim = (cat @ dog) / (np.linalg.norm(cat) * np.linalg.norm(dog))
print(sim)
```

---

## 12. Why Tokenization Still Matters a Lot

People sometimes focus only on model size and ignore tokenizer choice, but tokenization affects:

- sequence length
- cost
- multilingual behavior
- code representation
- edge cases in formatting and punctuation

A poor tokenizer can make downstream modeling harder.

---

## 13. Difference Summary

| Topic | Tokenization | Embeddings |
| --- | --- | --- |
| Input | raw text | token ids |
| Output | tokens / token ids | dense vectors |
| Main role | segment text | represent tokens numerically |
| Nature | discrete | continuous |
| Main math view | vocabulary and indexing | vector space representation |

---

## 14. Practical Intuition

Tokenization answers:

- what pieces should this text be split into?

Embeddings answer:

- how should each piece be represented for the neural network?

You need both. Tokenization alone gives ids, but ids do not carry geometry. Embeddings alone cannot exist until text has been segmented into tokens.

---

## 15. Chapter Summary

- tokenization converts text into reusable discrete units
- token ids are lookup keys, not meaningful numeric features
- embeddings convert token ids into dense learned vectors
- transformers operate on embedding matrices, not raw text
- contextual processing later refines these representations far beyond the initial lookup

## Practice Questions

1. Why are raw token ids not enough for neural processing?
2. Why do subword tokenizers dominate modern LLMs?
3. What is the difference between a token id and an embedding vector?
4. Why can the same token behave differently in different contexts?
5. Why does tokenizer quality affect model efficiency?
