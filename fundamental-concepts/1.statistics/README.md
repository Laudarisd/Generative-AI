# Statistics

## Why Statistics Comes First

Machine learning is built on data, and statistics is the language we use to describe data, variation, uncertainty, and evidence. Before a model predicts anything useful, we need to know what the data looks like, how noisy it is, and whether the observed pattern is likely real.

This chapter covers the statistical ideas that show up again and again in machine learning and generative AI.

---

## 1. Population, Sample, and Variables

- **Population**: the full set you care about
- **Sample**: the subset you actually measured
- **Variable**: a measurable quantity such as age, price, latency, or class label
- **Feature**: a variable used as model input
- **Target**: the value the model tries to predict

### Practical Example

If a company has 2 million users but you only analyze 50,000 user records from the last quarter, then:

- the 2 million users form the population
- the 50,000 records form the sample

This matters because conclusions drawn from a sample may not perfectly match the full population.

---

## 2. Descriptive Statistics

Descriptive statistics summarize what the data looks like.

### Mean

The mean is the average:

```math
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
```

where:

- $N$ is the number of observations
- $x_i$ is the $i$th observation

### Median

The median is the middle value after sorting.

Why median matters:

- the mean is sensitive to outliers
- the median is more robust when extreme values exist

### Mode

The mode is the most frequent value.

Useful example:

- most common product category purchased
- most common class label in a dataset

### Variance

Variance measures spread:

```math
\mathrm{Var}(x) = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
```

### Standard Deviation

```math
\sigma = \sqrt{\mathrm{Var}(x)}
```

### Range

```math
\mathrm{Range} = x_{max} - x_{min}
```

### Python Example

```python
import numpy as np

latencies = np.array([100, 110, 90, 120, 80])

print("mean:", latencies.mean())
print("median:", np.median(latencies))
print("variance:", latencies.var())
print("std:", latencies.std())
print("range:", latencies.max() - latencies.min())
```

---

## 3. Quantiles, Percentiles, and Outliers

### Percentile

The 90th percentile is the value below which 90% of the data lies.

### Quartiles

- Q1: 25th percentile
- Q2: median
- Q3: 75th percentile

### Interquartile Range

```math
IQR = Q3 - Q1
```

This is often used to detect outliers.

### Practical Example

In latency analysis:

- average latency might look fine
- but p95 latency may still be very poor

That is why percentiles matter in production systems.

### Python Example

```python
import numpy as np

values = np.array([3, 4, 5, 6, 7, 8, 100])

print("q1:", np.percentile(values, 25))
print("median:", np.percentile(values, 50))
print("q3:", np.percentile(values, 75))
print("p95:", np.percentile(values, 95))
```

---

## 4. Covariance and Correlation

Covariance tells us whether two variables move together:

```math
\mathrm{Cov}(x, y) = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_x)(y_i - \mu_y)
```

Correlation normalizes covariance:

```math
\rho_{x,y} = \frac{\mathrm{Cov}(x,y)}{\sigma_x \sigma_y}
```

Interpretation:

- close to `+1`: strong positive linear relationship
- close to `-1`: strong negative linear relationship
- close to `0`: weak linear relationship

### Practical Example

Hours studied and exam scores often show positive correlation.

### Important Warning

Correlation does not imply causation.

Two variables may move together without one causing the other.

### Python Example

```python
import numpy as np

study_hours = np.array([1, 2, 3, 4, 5])
scores = np.array([50, 55, 65, 72, 80])

print("covariance:", np.cov(study_hours, scores, bias=True)[0, 1])
print("correlation:", np.corrcoef(study_hours, scores)[0, 1])
```

---

## 5. Probability Foundations

Probability lets us reason under uncertainty.

### Conditional Probability

```math
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
```

This means: probability of event $A$ given that event $B$ already happened.

### Independence

Events are independent when:

```math
P(A \cap B) = P(A)P(B)
```

### Expected Value

For a discrete variable $X$:

```math
\mathbb{E}[X] = \sum_x x \cdot P(X=x)
```

Expected value is the long-run average outcome.

### Variance of a Random Variable

```math
\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]
```

---

## 6. Common Probability Distributions

### Bernoulli Distribution

Used for yes/no outcomes.

Examples:

- clicked or not clicked
- spam or not spam

### Binomial Distribution

Counts successes across repeated Bernoulli trials.

Example:

- number of successful conversions in 100 visits

### Normal Distribution

Bell-shaped and common in measurement noise.

### Categorical Distribution

Used when one outcome is selected from multiple classes.

Example:

- positive, neutral, negative sentiment

### Python Example

```python
import numpy as np

normal_samples = np.random.normal(loc=0.0, scale=1.0, size=5)
bernoulli_samples = np.random.binomial(n=1, p=0.7, size=5)

print("normal:", normal_samples)
print("bernoulli:", bernoulli_samples)
```

---

## 7. Sampling and Estimation

In real work, we rarely know the full population exactly. We estimate it from samples.

Examples of estimates:

- average revenue per user
- average token length
- conversion rate
- model accuracy on a test set

### Sample Mean

```math
\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i
```

This is used to estimate the true population mean.

### Standard Error

```math
SE = \frac{\sigma}{\sqrt{N}}
```

As sample size increases, standard error decreases.

---

## 8. Hypothesis Testing and Significance

Statistics is not only about measuring averages. It is also about deciding whether an effect is likely real.

Common setup:

- **Null hypothesis**: no real effect
- **Alternative hypothesis**: there is a real effect

### Practical Example

You launch a new product UI and click-through rate increases from 5% to 6%. Statistics helps decide whether that lift is likely real or just random noise.

### p-Value Intuition

A p-value answers:

- if the null hypothesis were true, how surprising would the observed result be?

This is useful in experiments, but should not be treated as magic.

---

## 9. Why Statistics Matters for ML

Statistics appears throughout ML and AI:

- data cleaning
- exploratory analysis
- feature analysis
- uncertainty estimation
- evaluation and confidence
- A/B testing
- probabilistic modeling
- Bayesian methods

If you understand statistics well, machine learning formulas stop looking arbitrary.

## Small End-to-End Example

```python
import numpy as np

values = np.array([12, 15, 14, 10, 18, 16, 14])

print("mean:", values.mean())
print("std:", values.std())
print("min:", values.min())
print("max:", values.max())
print("p75:", np.percentile(values, 75))
```

Continue to [Data Science](../2.data-science/README.md).

---

## 10. Confidence Intervals and Estimation

Point estimates are useful, but in practice we also want a range of plausible values.

A confidence interval gives an interval that would contain the true parameter in repeated sampling under the modeling assumptions.

For a mean, a simple large-sample interval is:

```math
\bar{x} \pm z \cdot \frac{\sigma}{\sqrt{N}}
```

where:

- $\bar{x}$ is the sample mean
- $z$ is a critical value such as $1.96$ for an approximate 95% interval
- $\sigma / \sqrt{N}$ is the standard error

### Practical Example

Suppose average model latency is 180 ms. A confidence interval gives a more honest statement than a single number:

- point estimate: 180 ms
- interval estimate: maybe 180 +/- 12 ms

That helps teams reason about uncertainty rather than pretending the estimate is exact.

### Python Example

```python
import numpy as np

values = np.array([172, 181, 177, 190, 185, 179, 183, 176])

mean = values.mean()
std = values.std(ddof=1)
n = len(values)
se = std / np.sqrt(n)
z = 1.96

lower = mean - z * se
upper = mean + z * se

print("mean:", mean)
print("95% CI:", (lower, upper))
```

---

## 11. Sampling Bias and Data Quality

Statistics is not only about formulas. It is also about whether the data was collected in a trustworthy way.

Common issues:

- **sampling bias**: the sample is not representative of the population
- **survivorship bias**: only successful cases remain visible
- **measurement error**: values are noisy or incorrect
- **label noise**: targets are wrong or inconsistent

### Practical Example

If you estimate average user satisfaction only from users who answered a feedback survey, the result may be biased. People with strong positive or negative opinions are more likely to respond.

In ML this matters because a model trained on biased data can look strong offline but fail in production.

---

## 12. Central Limit Theorem and Why It Matters

The Central Limit Theorem, or **CLT**, explains why averages are so important.

It says that under broad conditions, the distribution of the sample mean becomes approximately normal as the sample size grows.

This is why confidence intervals, z-tests, and many practical approximations work well in large datasets.

### Intuition

- one observation may be noisy
- the average of many observations is more stable
- repeated averages tend to form a bell-shaped distribution

### Python Example

```python
import numpy as np

rng = np.random.default_rng(42)
sample_means = []

for _ in range(1000):
    sample = rng.exponential(scale=2.0, size=50)
    sample_means.append(sample.mean())

print("mean of sample means:", np.mean(sample_means))
print("std of sample means:", np.std(sample_means))
```

Even though the raw data here is exponential, the distribution of sample means becomes much more normal-looking.

---

## 13. A/B Testing Intuition

A/B testing is one of the most practical uses of statistics in product and ML work.

Typical setup:

- group A sees the old version
- group B sees the new version
- you compare metrics such as click-through rate, revenue, or retention

### Example

If a prompt rewrite increases task completion from 41% to 45%, statistics helps answer:

- is that gain likely real?
- how large is the effect?
- is the improvement worth shipping?

### Python Example

```python
import numpy as np

control = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
variant = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])

control_rate = control.mean()
variant_rate = variant.mean()
lift = variant_rate - control_rate

print("control_rate:", control_rate)
print("variant_rate:", variant_rate)
print("absolute_lift:", lift)
```

This tiny example is not a full significance test, but it shows the workflow: measure, compare, then judge uncertainty.
