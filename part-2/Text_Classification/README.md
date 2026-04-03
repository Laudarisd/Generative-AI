# Text Classification

Text classification is one of the most practical tasks in machine learning and NLP. The goal is simple: read a piece of text and assign one or more labels to it.

Common applications include:

- spam vs non-spam email filtering
- positive / negative / neutral sentiment analysis
- topic tagging for articles
- intent detection for chatbots
- toxicity and abuse detection
- document routing in business workflows

## 1. Problem Setup

Given an input text $x$, predict a label $y$ from a label set $\mathcal{Y}$:

```math
\hat{y} = f_\theta(x)
```

If there are $K$ possible classes, the model often outputs a score vector:

```math
z = [z_1, z_2, \dots, z_K]
```

For multiclass classification, the scores are converted to probabilities using softmax:

```math
p(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
```

The predicted label is then:

```math
\hat{y} = \arg\max_k p(y = k \mid x)
```

## 2. Types of Text Classification

### Binary Classification

Two labels only.

Examples:

- spam / not spam
- toxic / not toxic

### Multiclass Classification

Exactly one class among many.

Examples:

- sports / politics / business / science

### Multi-Label Classification

A document can have multiple labels at once.

Examples:

- article tagged as both `AI` and `healthcare`

In that case, each class is often modeled with a sigmoid output instead of softmax.

## 3. Typical Pipeline

Most text classification systems follow these steps:

1. clean or normalize text
2. tokenize the text
3. convert tokens into features or embeddings
4. pass features into a classifier
5. compute a loss
6. optimize model parameters
7. evaluate on held-out data

## 4. Feature Representations

### Bag of Words

Represent text by token counts.

This is simple and often surprisingly competitive on small datasets.

### TF-IDF

TF-IDF gives more weight to words that are important in a document but not too common across the whole dataset.

```math
\mathrm{TF\mbox{-}IDF}(t, d) = \mathrm{TF}(t, d) \cdot \log \frac{N}{\mathrm{DF}(t)}
```

where:

- $\mathrm{TF}(t, d)$ is the term frequency of token $t$ in document $d$
- $\mathrm{DF}(t)$ is document frequency
- $N$ is the number of documents

### Word Embeddings

Each token gets a dense vector. Words with similar meanings often get similar vectors.

### Contextual Embeddings

Transformer encoders like BERT create different embeddings for the same word depending on its context.

Example:

- "bank" in "river bank"
- "bank" in "open a bank account"

## 5. Common Model Families

### Linear Models

Examples:

- logistic regression
- linear SVM

They are fast, interpretable, and work well with TF-IDF.

### CNN and RNN Classifiers

Older neural NLP systems used:

- CNNs for local phrase patterns
- RNNs / LSTMs / GRUs for sequence modeling

### Transformer Encoders

Models such as BERT, RoBERTa, and DeBERTa are now strong default choices because they build contextual token representations before classification.

## 6. Loss Functions

### Binary Cross-Entropy

For binary classification:

```math
\mathcal{L}_{BCE} = - \left[y \log \hat{p} + (1-y)\log(1-\hat{p})\right]
```

### Multiclass Cross-Entropy

For multiclass classification:

```math
\mathcal{L}_{CE} = - \sum_{k=1}^{K} y_k \log p_k
```

where $y_k$ is one-hot encoded.

## 7. Metrics

Accuracy alone is often not enough.

Important metrics include:

- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC for imbalanced datasets

Example:

If a spam detector marks everything as "not spam," it may get high accuracy on an imbalanced dataset while being useless in practice.

## 8. Practical Example: Sentiment Classification with TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

texts = [
    "this movie is amazing",
    "absolutely terrible acting",
    "great visual effects and strong ending",
    "boring and too long",
]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

test_texts = ["great acting", "terrible movie"]
pred = model.predict(vectorizer.transform(test_texts))

print(pred)
print(classification_report([1, 0], pred))
```

## 9. Practical Example: PyTorch Embedding Classifier

This example uses token embeddings, mean pooling, and a linear head.

```python
import torch
import torch.nn as nn

class TinyTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # [batch, seq_len, embed_dim]
        pooled = emb.mean(dim=1)         # [batch, embed_dim]
        logits = self.fc(pooled)         # [batch, num_classes]
        return logits

model = TinyTextClassifier(vocab_size=5000, embed_dim=64, num_classes=3)
tokens = torch.randint(0, 5000, (4, 12))
logits = model(tokens)

print("token shape:", tokens.shape)
print("logit shape:", logits.shape)
```

## 10. Example: BERT-Style Classification Head

A transformer encoder usually produces a representation for every token:

```math
H \in \mathbb{R}^{n \times d}
```

One common strategy is to take the special `[CLS]` token representation $h_{cls}$ and classify with:

```math
z = W h_{cls} + b
```

```python
import torch
import torch.nn as nn

batch_size = 2
seq_len = 6
hidden_dim = 768
num_classes = 4

encoder_output = torch.randn(batch_size, seq_len, hidden_dim)
cls_rep = encoder_output[:, 0, :]
classifier = nn.Linear(hidden_dim, num_classes)
logits = classifier(cls_rep)

print(logits.shape)
```

## 11. Challenges in Real Systems

- class imbalance
- noisy labels
- domain shift
- long documents
- rare classes
- multilingual inputs

Practical fixes include:

- weighted loss
- focal loss
- hierarchical labeling
- chunking long documents
- domain-specific fine-tuning

## 12. When to Use Which Approach

| Approach | Best Use Case | Tradeoff |
| --- | --- | --- |
| TF-IDF + logistic regression | small datasets, fast baselines | misses deep context |
| CNN / RNN | lightweight neural baselines | older and less flexible |
| BERT-like encoder | strong understanding tasks | heavier compute |
| LLM prompting | rapid prototyping, few-shot tasks | higher latency and cost |

## Summary

Text classification is a foundational application of NLP. It connects preprocessing, representation learning, losses, evaluation, and deployment tradeoffs in one practical workflow. In real projects, a simple TF-IDF baseline is often the right starting point, while encoder-based transformers become valuable when context and accuracy matter more than simplicity.
