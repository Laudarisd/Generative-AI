# Bayesian Machine Learning

Bayesian machine learning treats uncertainty as a first-class object. Instead of learning only one best parameter value, it reasons about distributions over unknown quantities.

This is important because in many scientific and engineering settings, a model should not only answer "what is the prediction?" but also "how certain is that prediction?"

## 1. Core Principle

Bayes' rule:

```math
P(\theta \mid D) = \frac{P(D \mid \theta)P(\theta)}P(D)
```

where:

- $P(\theta)$ is the prior
- $P(D \mid \theta)$ is the likelihood
- $P(\theta \mid D)$ is the posterior
- $P(D)$ is the evidence or marginal likelihood

In ML:

- $\theta$ can be model parameters
- $D$ is observed data

## 2. Why Bayesian ML Matters

Bayesian ML is useful when:

- uncertainty matters
- data is limited
- decisions are high stakes
- prior knowledge is valuable
- model calibration matters

Applications include:

- medical modeling
- active learning
- Bayesian optimization
- model calibration
- scientific inference
- robotics
- reliability analysis

## 3. Prior, Likelihood, Posterior Intuition

### Prior

What we believe before seeing the current dataset.

### Likelihood

How probable the data is under a parameter choice.

### Posterior

Updated belief after combining prior knowledge with observed data.

This is conceptually powerful because it turns learning into belief updating rather than only optimization.

## 4. MLE vs MAP vs Full Bayesian Inference

Maximum likelihood:

```math
\theta_{MLE} = \arg\max_\theta P(D \mid \theta)
```

Maximum a posteriori:

```math
\theta_{MAP} = \arg\max_\theta P(\theta \mid D)
```

Full Bayesian inference reasons over the entire posterior distribution instead of collapsing to one point estimate.

That means the output is not just one best parameter value, but a distribution expressing uncertainty.

## 5. Predictive Distribution

A Bayesian model makes predictions by integrating over parameter uncertainty:

```math
P(y_* \mid x_*, D) = \int P(y_* \mid x_*, \theta) P(\theta \mid D) d\theta
```

This integral is often analytically hard, which is why approximation methods matter.

## 6. Naive Bayes

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

## 7. Bayesian Linear Regression

Instead of fixed weights, Bayesian linear regression places a distribution over weights.

This gives:

- prediction mean
- prediction uncertainty

That is often more useful than a single point prediction.

If ordinary linear regression is:

```math
y = Xw + \epsilon
```

Bayesian linear regression treats $w$ itself as random with a prior distribution.

## 8. Gaussian Processes

Gaussian processes, or **GPs**, define distributions over functions.

They are especially useful for:

- small-to-medium datasets
- uncertainty-aware regression
- Bayesian optimization

A GP is specified by:

- mean function
- kernel / covariance function

The kernel encodes assumptions such as smoothness and correlation across input space.

## 9. Bayesian Optimization

Bayesian optimization is used when evaluations are expensive.

Examples:

- hyperparameter tuning
- scientific experiment design
- simulator calibration
- engineering design search

A common loop is:

1. fit a surrogate model such as a GP
2. compute an acquisition function
3. choose the next point to evaluate
4. update the posterior with the new observation

## 10. Approximate Inference

Exact Bayesian inference is often hard in modern ML.

Common approximation approaches include:

- Laplace approximation
- variational inference
- Markov chain Monte Carlo (MCMC)
- Monte Carlo dropout as a rough approximation in deep nets

## 11. Why Bayesian ML Fits Science So Well

Science often needs:

- calibrated uncertainty
- transparent assumptions
- integration of prior knowledge
- robust decisions under limited data

Those needs align naturally with Bayesian thinking.

## 12. Tiny Posterior Updating Example

Suppose we model a coin bias with a Beta prior. After observing heads and tails, the posterior updates analytically.

```python
alpha_prior = 2
beta_prior = 2

heads = 8
tails = 2

alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

print(alpha_post, beta_post)
```

This is one of the simplest examples of Bayesian updating.

## 13. Practical Questions to Ask

When using Bayesian ML, ask:

- what prior assumptions are reasonable?
- what uncertainty matters for the decision?
- do we need full posterior inference or just MAP?
- how expensive is approximate inference?

## Problems to Think About

1. Why is a posterior distribution often more informative than a point estimate?
2. What is the practical difference between MLE and MAP?
3. Why are Gaussian processes attractive for expensive small-data problems?
4. Why is uncertainty central in scientific modeling?
5. When is Bayesian inference too expensive to use directly?

## Summary

Bayesian machine learning reframes learning as uncertainty-aware inference. It is valuable whenever predictions alone are not enough and the system must also express confidence, incorporate prior knowledge, and support decisions under uncertainty.
