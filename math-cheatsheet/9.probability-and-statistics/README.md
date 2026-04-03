# Probability and Statistics Mathematics

Probability and statistics are core mathematical languages for machine learning. They explain uncertainty, randomness, data variability, estimation, confidence, and probabilistic modeling.

This chapter is more mathematical than the earlier fundamentals chapter. The goal here is formula fluency, worked examples, and problem-solving intuition.

## 1. Sample Space and Events

The sample space $\Omega$ is the set of all possible outcomes.

Examples:

- coin flip: $\Omega = \{H, T\}$
- dice roll: $\Omega = \{1,2,3,4,5,6\}$

An event is a subset of the sample space.

## 2. Probability Axioms

Core rules:

```math
0 \le P(A) \le 1
```

```math
P(\Omega)=1
```

```math
P(A \cup B)=P(A)+P(B)-P(A \cap B)
```

If $A$ and $B$ are disjoint:

```math
P(A \cup B)=P(A)+P(B)
```

## 3. Conditional Probability and Independence

Conditional probability:

```math
P(A \mid B)=\frac{P(A \cap B)}{P(B)}
```

Independence:

```math
P(A \cap B)=P(A)P(B)
```

### Worked Example

From a standard deck, what is the probability of drawing a king given that the card is a face card?

There are 12 face cards and 4 kings, so:

```math
P(\text{king} \mid \text{face}) = \frac{4}{12} = \frac{1}{3}
```

## 4. Bayes' Rule

```math
P(A \mid B)=\frac{P(B \mid A)P(A)}{P(B)}
```

This is one of the most important formulas in AI because it updates beliefs after evidence.

### Worked Example

Suppose:

- disease prevalence: $1\%$
- positive test if diseased: $99\%$
- false positive rate: $5\%$

Then:

```math
P(D \mid +)=\frac{P(+ \mid D)P(D)}{P(+)}
```

where:

```math
P(+)=0.99(0.01)+0.05(0.99)=0.0594
```

So:

```math
P(D \mid +)=\frac{0.99 \cdot 0.01}{0.0594} \approx 0.1667
```

Even a strong test can produce a surprisingly modest posterior if the base rate is low.

## 5. Random Variables

A random variable maps outcomes to numbers.

### Discrete Random Variable

Example:

- number of heads in 2 flips

### Continuous Random Variable

Example:

- temperature
- latency
- pixel intensity after normalization

## 6. Expectation and Variance

Expected value:

```math
\mathbb{E}[X]=\sum_x xP(X=x)
```

or for continuous variables:

```math
\mathbb{E}[X]=\int x f(x)\,dx
```

Variance:

```math
\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]
```

Standard deviation:

```math
\sigma = \sqrt{\mathrm{Var}(X)}
```

### Worked Example

Let $X$ be a fair die roll.

```math
\mathbb{E}[X]=\frac{1+2+3+4+5+6}{6}=3.5
```

## 7. Common Distributions

### Bernoulli

```math
P(X=1)=p,\quad P(X=0)=1-p
```

### Binomial

```math
P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
```

### Poisson

```math
P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}
```

### Normal

```math
X \sim \mathcal{N}(\mu,\sigma^2)
```

### Exponential

Often used for waiting times.

### Categorical

Used in multi-class modeling.

## 8. Joint, Marginal, and Conditional Distributions

Joint distribution:

```math
P(X,Y)
```

Marginalization:

```math
P(X)=\sum_y P(X,Y)
```

Conditional:

```math
P(X \mid Y)=\frac{P(X,Y)}{P(Y)}
```

These are central in graphical models, HMMs, Bayesian networks, and latent-variable models.

## 9. Covariance and Correlation

Covariance:

```math
\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
```

Correlation:

```math
\rho_{X,Y} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
```

Why this matters:

- PCA
- feature analysis
- multicollinearity
- uncertainty structure

## 10. Estimation

An estimator is a rule for inferring unknown quantities from data.

Example sample mean:

```math
\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i
```

Properties people care about:

- bias
- variance
- consistency

## 11. Law of Large Numbers and Central Limit Theorem

### Law of Large Numbers

As sample size grows, the sample mean approaches the true mean.

### Central Limit Theorem

Under broad conditions:

```math
\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}
```

approaches a standard normal distribution.

This matters for:

- confidence intervals
- test statistics
- practical approximation

## 12. Confidence Intervals

Approximate confidence interval for a mean:

```math
\bar{x} \pm z \frac{\sigma}{\sqrt{n}}
```

### Worked Example

If:

- sample mean = 50
- standard deviation = 10
- sample size = 25
- $z=1.96$

then:

```math
50 \pm 1.96 \cdot \frac{10}{5}
= 50 \pm 3.92
```

So the approximate 95% interval is:

```math
[46.08, 53.92]
```

## 13. Hypothesis Testing

Basic structure:

- null hypothesis $H_0$
- alternative hypothesis $H_1$
- compute a statistic
- compute p-value or rejection region

This matters in experimentation, A/B testing, and scientific claims.

## 14. Likelihood and Log-Likelihood

Likelihood:

```math
L(\theta)=P(D \mid \theta)
```

Log-likelihood:

```math
\ell(\theta)=\log P(D \mid \theta)
```

Why logs help:

- products become sums
- numerical stability improves
- derivatives are easier to compute

This is fundamental in MLE and many ML losses.

## 15. Entropy and Cross-Entropy

Entropy:

```math
H(P)=-\sum_i P(i)\log P(i)
```

Cross-entropy:

```math
H(P,Q)=-\sum_i P(i)\log Q(i)
```

KL divergence:

```math
D_{KL}(P\|Q)=\sum_i P(i)\log\frac{P(i)}{Q(i)}
```

These appear in:

- classification
- language modeling
- distillation
- variational inference

## 16. Maximum Likelihood Estimation

MLE chooses:

```math
\hat{\theta}_{MLE} = \arg\max_\theta P(D \mid \theta)
```

or equivalently:

```math
\hat{\theta}_{MLE} = \arg\max_\theta \log P(D \mid \theta)
```

## 17. Python Examples

```python
import numpy as np

# expectation of a fair die
values = np.array([1, 2, 3, 4, 5, 6])
probs = np.ones(6) / 6
expectation = np.sum(values * probs)
print("expectation:", expectation)

# simple confidence interval
sample_mean = 50
sample_std = 10
n = 25
z = 1.96
margin = z * sample_std / np.sqrt(n)
print("95% CI:", (sample_mean - margin, sample_mean + margin))
```

## 18. Practice Problems

1. Compute the expected value of a fair coin coded as 1 for heads and 0 for tails.
2. Explain the difference between correlation and causation.
3. Derive the mean of a Bernoulli random variable.
4. Compute one simple Bayesian update from prior, likelihood, and evidence.
5. Explain why log-likelihood is preferred over raw likelihood in many models.
