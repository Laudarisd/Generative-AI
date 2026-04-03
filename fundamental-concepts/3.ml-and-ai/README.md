# Machine Learning and AI

## AI vs ML

Artificial Intelligence is the broader field of building systems that act intelligently.

Machine Learning is the part of AI where models learn from data rather than following only hand-written rules.

---

## 1. Supervised Learning

In supervised learning, we use labeled examples:

```math
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
```

Typical objective:

```math
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(f_\theta(x_i), y_i)
```

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

## 2. Unsupervised Learning

Here the data has no labels:

```math
\mathcal{D} = \{x_i\}_{i=1}^{N}
```

### Python Example: k-Means

```python
from sklearn.cluster import KMeans

X = [[1, 2], [1, 3], [8, 8], [9, 8]]

model = KMeans(n_clusters=2, random_state=0, n_init=10)
model.fit(X)

print(model.labels_)
```

---

## 3. Key Algorithms

### Logistic Regression

```math
\hat{y} = \sigma(w^T x + b)
```

where:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

### Python Example

```python
from sklearn.linear_model import LogisticRegression

X = [[0.1], [0.3], [0.8], [1.2]]
y = [0, 0, 1, 1]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[0.9]]))
```

### Decision Tree

Decision trees split data with simple rule-based questions.

```python
from sklearn.tree import DecisionTreeClassifier

X = [[25, 500], [45, 700], [35, 650], [50, 800]]
y = [0, 1, 0, 1]

model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

print(model.predict([[40, 720]]))
```

### PCA

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

## 4. Optimization

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

Continue to [Generative AI](../4.generative-ai/README.md).
