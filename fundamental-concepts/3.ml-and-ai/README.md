# Machine Learning and AI

## AI vs ML

Artificial Intelligence is the broader field of building systems that act intelligently.

Machine Learning is the part of AI where models learn from data rather than following only hand-written rules.

Examples of AI beyond classic ML:

- search in chess engines
- rule-based expert systems
- symbolic planning
- robotics control logic

Examples of ML:

- fraud detection
- recommendation systems
- language models
- image recognition

---

## 1. The Core Learning Setup

A learning system tries to approximate a function:

```math
f_\theta(x)
```

where:

- $x$ is the input
- $\theta$ are learnable parameters
- $f_\theta(x)$ is the prediction

Training means finding parameter values that reduce error on real data.

---

## 2. Supervised Learning

In supervised learning, we use labeled examples:

```math
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
```

Typical objective:

```math
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_\theta(x_i), y_i)
```

### Typical Tasks

- classification
- regression
- ranking

### Python Example: Linear Regression

```python
from sklearn.linear_model import LinearRegression

X = [[500], [700], [900], [1100]]
y = [1200, 1500, 1800, 2200]

model = LinearRegression()
model.fit(X, y)

print(model.predict([[1000]]))
```

---

## 3. Unsupervised Learning

Here the data has no labels:

```math
\mathcal{D} = \{x_i\}_{i=1}^{N}
```

The goal is to discover structure.

### Common Tasks

- clustering
- dimensionality reduction
- anomaly detection
- density estimation

### Python Example: k-Means

```python
from sklearn.cluster import KMeans

X = [[1, 2], [1, 3], [8, 8], [9, 8]]

model = KMeans(n_clusters=2, random_state=0, n_init=10)
model.fit(X)

print(model.labels_)
print(model.cluster_centers_)
```

---

## 4. Semi-Supervised Learning

Semi-supervised learning combines a small labeled set and a large unlabeled set:

```math
\mathcal{D}_L = \{(x_i, y_i)\}, \quad \mathcal{D}_U = \{x_j\}
```

One common idea is:

```math
\mathcal{L} = \mathcal{L}_{sup} + \lambda \mathcal{L}_{unsup}
```

This matters when labels are expensive but raw data is abundant.

Example:

- a few thousand labeled medical scans
- hundreds of thousands of unlabeled scans

---

## 5. Reinforcement Learning

An agent interacts with an environment and receives rewards.

Objective:

```math
\max_\pi \mathbb{E}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
```

where:

- $\pi$ is the policy
- $r_t$ is reward at time $t$
- $\gamma$ is the discount factor

Examples:

- robotics
- games
- recommendation policies
- alignment-related optimization in some LLM pipelines

---

## 6. Important Algorithms

### 6.1 Linear Regression

```math
\hat{y} = w^T x + b
```

Used for continuous prediction.

### 6.2 Logistic Regression

```math
\hat{y} = \sigma(w^T x + b)
```

where:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

Used for binary classification.

### Python Example

```python
from sklearn.linear_model import LogisticRegression

X = [[0.1], [0.3], [0.8], [1.2]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[0.9]]))
print(model.predict_proba([[0.9]]))
```

### 6.3 Decision Tree

Decision trees split data with rule-based questions.

```python
from sklearn.tree import DecisionTreeClassifier

X = [[25, 500], [45, 700], [35, 650], [50, 800]]
y = [0, 1, 0, 1]

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

print(model.predict([[40, 720]]))
```

### 6.4 Random Forest

A random forest combines many trees to reduce overfitting and improve stability.

### 6.5 Support Vector Machine

SVMs try to maximize the margin between classes.

### 6.6 Naive Bayes

Naive Bayes uses probabilistic assumptions and is often strong on text baselines.

### 6.7 PCA

PCA reduces dimensions while keeping strong directions of variance.

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print(X_reduced)
```

---

## 7. Evaluation

Different problems need different metrics.

### Classification Metrics

- accuracy
- precision
- recall
- F1 score
- ROC-AUC

### Regression Metrics

- MAE
- MSE
- RMSE
- $R^2$

### Clustering Metrics

- inertia
- silhouette score

### Practical Note

For fraud detection, accuracy can be misleading because the positive class is rare. Precision and recall often matter more.

---

## 8. Optimization

Training updates parameters to reduce loss.

```math
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}
```

### Python Example

```python
w = 5.0
learning_rate = 0.1

for step in range(5):
    grad = 2 * (w - 1)
    w = w - learning_rate * grad
    print(step, w)
```

---

## 9. Neural Networks and Deep Learning

Deep learning stacks many learned transformations:

```math
h^{(1)} = \phi(W^{(1)}x + b^{(1)})
```

```math
h^{(2)} = \phi(W^{(2)}h^{(1)} + b^{(2)})
```

```math
\hat{y} = W^{(L)}h^{(L-1)} + b^{(L)}
```

Common activation:

```math
\mathrm{ReLU}(x) = \max(0, x)
```

Deep learning matters because it learns useful representations automatically from large data.

Continue to [Generative AI](../4.generative-ai/README.md).

---

## 10. Bias-Variance Tradeoff

One of the most important ideas in ML is the balance between **bias** and **variance**.

- **high bias**: the model is too simple and misses the true pattern
- **high variance**: the model fits the training data too closely and becomes unstable

### Intuition

- linear regression on a nonlinear problem may underfit
- a very deep tree on a tiny dataset may overfit

This tradeoff explains why model complexity must match the data and the task.

---

## 11. When to Use Which Algorithm

Professional work is rarely about memorizing formulas only. It is about choosing the right tool.

### Linear and Logistic Regression

Use when:

- you need a strong baseline
- interpretability matters
- the relationship is reasonably simple

### Trees and Random Forests

Use when:

- features are mostly tabular
- interactions are important
- you want strong performance without heavy preprocessing

### Gradient Boosting

Use when:

- tabular data quality is decent
- you want a high-performance classical model

Examples:

- XGBoost
- LightGBM
- CatBoost

### SVM

Use when:

- the dataset is not extremely large
- margins matter
- feature space is informative

### Neural Networks

Use when:

- data is large
- raw inputs are complex
- representation learning matters

Examples:

- images
- text
- audio
- multimodal data

---

## 12. Confusion Matrix and Error Analysis

Metrics are useful, but serious ML work also studies the mistakes.

A confusion matrix for binary classification tracks:

- true positives
- false positives
- true negatives
- false negatives

### Why it matters

Two models with the same accuracy can behave very differently.

In medical diagnosis:

- false negatives can be dangerous

In spam filtering:

- false positives may frustrate users

### Python Example

```python
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 0]
y_pred = [1, 0, 0, 1, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

---

## 13. Feature Scaling and Preprocessing

Some algorithms are very sensitive to feature scale.

Sensitive models:

- logistic regression
- SVM
- k-nearest neighbors
- neural networks

Less sensitive models:

- decision trees
- random forests
- many boosting methods

### Python Example

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[1000, 1], [1200, 0], [900, 1]], dtype=float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

---

## 14. From Classical ML to Generative AI

The path from classical ML to modern generative AI is easier to understand if you see the progression:

1. statistics teaches measurement and uncertainty
2. data science turns raw data into usable datasets
3. machine learning learns patterns from those datasets
4. deep learning learns rich representations automatically
5. generative AI learns to produce new content, not only predict labels

That is why these chapters are ordered this way. Each layer builds on the previous one.
