# Bayesian Machine Learning

Bayesian machine learning treats uncertainty as a first-class object. Instead of learning only one best parameter value, it reasons about distributions over unknown quantities.

## 1. Core Principle

Bayes' rule:

```math
P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}P(D)
```

where:

- $P(\theta)$ is the prior
- $P(D \mid \theta)$ is the likelihood
- $P(\theta \mid D)$ is the posterior

In ML:

- $\theta$ can be model parameters
- $D$ is observed data

## 2. Why Bayesian ML Matters

Bayesian ML is useful when:

- uncertainty matters
- data is limited
- decisions are high stakes
- prior knowledge is valuable

Applications:

- medical modeling
- active learning
- Bayesian optimization
- model calibration
- scientific inference

## 3. MLE vs MAP vs Full Bayesian Inference

Maximum likelihood:

```math
\theta_{MLE} = \arg\max_\theta P(D \mid \theta)
```

Maximum a posteriori:

```math
\theta_{MAP} = \arg\max_\theta P(\theta \mid D)
```

Full Bayesian inference reasons over the entire posterior distribution instead of collapsing to one point estimate.

## 4. Naive Bayes

Naive Bayes uses:

```math
P(y \mid x_1,\dots,x_n) \propto P(y)\prod_{i=1}^{n}P(x_i \mid y)
```

It assumes conditional independence of features given the class.

Even though the assumption is strong, it often works well for text classification.

### Python Example

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["free money now", "project meeting tomorrow", "claim your prize", "team update"]
labels = [1, 0, 1, 0]

vec = CountVectorizer()
X = vec.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

print(model.predict(vec.transform(["free prize"])))
```

## 5. Bayesian Linear Regression

Instead of fixed weights, Bayesian linear regression places a distribution over weights.

This gives:

- prediction mean
- prediction uncertainty

That is often more useful than a single point prediction.

## 6. Gaussian Processes

Gaussian processes, or **GPs**, define distributions over functions.

They are especially useful for:

- small-to-medium datasets
- uncertainty-aware regression
- Bayesian optimization

A GP is specified by:

- mean function
- kernel / covariance function

## 7. Bayesian Optimization

Bayesian optimization is used when evaluations are expensive.

Examples:

- hyperparameter tuning
- scientific experiment design
- simulation calibration

High-level loop:

1. fit a surrogate model, often a GP
2. use an acquisition function to choose the next experiment
3. evaluate the true objective
4. update the surrogate

## 8. Bayesian Deep Learning

Deep learning can be made more uncertainty-aware via:

- variational inference
- Monte Carlo dropout
- ensemble approximations
- Laplace approximations

## 9. Practical Example: Beta-Binomial Update

For binary outcomes:

```math
\text{Posterior} = \mathrm{Beta}(\alpha + \text{heads}, \beta + \text{tails})
```

### Python Example

```python
alpha, beta = 2, 2
data = [1, 1, 0, 1, 0, 1]

heads = sum(data)
tails = len(data) - heads

posterior_alpha = alpha + heads
posterior_beta = beta + tails

print(posterior_alpha, posterior_beta)
print(posterior_alpha / (posterior_alpha + posterior_beta))
```

## Problems to Think About

1. Why is posterior uncertainty useful in scientific applications?
2. What is the difference between MAP and full Bayesian inference?
3. Why do Gaussian processes become expensive at scale?
4. When is Naive Bayes still a good baseline?
5. Why is Bayesian optimization useful for costly training runs?

## References

- Probabilistic Machine Learning resources: https://probml.github.io/pml-book/
- Gaussian Processes for Machine Learning: https://gaussianprocess.org/gpml/chapters/
