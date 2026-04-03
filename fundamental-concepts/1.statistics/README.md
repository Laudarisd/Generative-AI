# Statistics

## Why Statistics Comes First

Machine learning is built on data, and statistics is the language we use to describe data, uncertainty, and patterns. Before a model predicts anything, we need to understand what the data looks like and how confident we should be in what we observe.

---

## 1. Population, Sample, and Variables

- **Population**: the full set you care about
- **Sample**: the part you actually measured
- **Variable**: a measurable quantity such as age, latency, price, or label

### Practical Example

If you want to understand all customers of a company, the population is all customers. If you only have last quarter's transaction data, that is a sample.

---

## 2. Descriptive Statistics

### Mean

The mean is the average:

```math
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
```

where:

- $N$ is the number of observations
- $x_i$ is the $i$th value

### Median

The median is the middle value after sorting.

Why it matters:

- the mean reacts strongly to outliers
- the median is often more robust

### Variance

Variance measures spread:

```math
\mathrm{Var}(x) = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
```

### Standard Deviation

```math
\sigma = \sqrt{\mathrm{Var}(x)}
```

### Python Example

```python
import numpy as np

latencies = np.array([100, 110, 90, 120, 80])

print("mean:", latencies.mean())
print("median:", np.median(latencies))
print("variance:", latencies.var())
print("std:", latencies.std())
```

---

## 3. Correlation and Probability

Covariance:

```math
\mathrm{Cov}(x, y) = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_x)(y_i - \mu_y)
```

Correlation:

```math
\rho_{x,y} = \frac{\mathrm{Cov}(x,y)}{\sigma_x \sigma_y}
```

Conditional probability:

```math
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
```

Bayes' rule:

```math
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
```

### Python Example

```python
import numpy as np

study_hours = np.array([1, 2, 3, 4, 5])
scores = np.array([50, 55, 65, 72, 80])

print(np.corrcoef(study_hours, scores)[0, 1])
```

Continue to [Data Science](../2.data-science/README.md).
