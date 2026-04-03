# Scientific AI with Mathematics

Scientific AI is the broader area where machine learning is used not only for text, images, and apps, but for scientific discovery, simulation, forecasting, and engineering decision-making.

## 1. What Makes Scientific AI Different

In many mainstream AI tasks, success means strong prediction or generation.

In scientific AI, we often need more:

- physical consistency
- uncertainty estimation
- interpretability
- sample efficiency
- extrapolation under structure
- robustness to simulation or measurement noise

Scientific AI is often used when the outputs interact with real systems, not only benchmarks.

## 2. Mathematical Ingredients

Scientific AI typically combines:

- calculus
- linear algebra
- differential equations
- optimization
- probability
- numerical analysis
- statistics

That is why the `math-cheatsheet/` section exists separately.

## 3. Model Families in Scientific AI

Important families include:

- PINNs
- neural operators
- Gaussian processes
- surrogate models
- hybrid mechanistic + data-driven models
- graph neural networks for scientific systems
- sequence models for scientific time series

## 4. Example: Surrogate Modeling

Suppose a full simulator is expensive. A neural network surrogate can learn:

```math
\hat{u} = f_\theta(\text{parameters}, \text{boundary conditions}, \text{inputs})
```

This can accelerate design loops and uncertainty studies.

A surrogate is useful when the true simulator is:

- slow
- expensive
- difficult to differentiate through
- needed inside optimization loops

## 5. Error, Stability, and Generalization

Scientific AI needs a stricter mindset:

- predictive accuracy alone may not be enough
- physically impossible outputs can be unacceptable
- uncertainty may matter more than raw average performance

Important questions:

- is the model stable?
- does it conserve important quantities?
- does it extrapolate sensibly?
- how wrong can it be under distribution shift?

## 6. Tiny Surrogate Example

```python
import numpy as np
from sklearn.neural_network import MLPRegressor

X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
y = np.sin(X).ravel()

model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=5000, random_state=0)
model.fit(X, y)

print(model.predict([[1.2]]))
```

This is only a toy surrogate, but the principle is real.

## 7. Why Numerical Methods Matter

Many scientific workflows involve discretization, simulation, and numerical error.

That means scientific AI often lives beside tools such as:

- finite difference methods
- finite element methods
- spectral methods
- Monte Carlo simulation
- optimization solvers

The ML model may replace, accelerate, or assist these methods rather than fully replacing science.

## 8. Hybrid Mechanistic + Data-Driven Systems

A useful scientific AI system is often hybrid.

Examples:

- learn the part of the dynamics that a mechanistic model misses
- use a simulator to generate synthetic training data
- use a neural model inside a larger physics solver
- combine Bayesian uncertainty with mechanistic constraints

## 9. Scientific Objectives Are Often Multi-Criteria

In ordinary ML, the objective may be mostly one scalar metric.

In scientific AI, we may care about several things at once:

- accuracy
- physical consistency
- uncertainty calibration
- stability
- interpretability
- computational efficiency

That is why scientific AI design is often more constrained and more mathematically explicit.

## 10. Where Scientific AI Appears

- climate and weather modeling
- fluid dynamics
- materials science
- molecular modeling
- energy systems
- geoscience
- engineering optimization
- digital twins

## 11. Example: Learning Inside an Optimization Loop

Suppose a design team needs to test thousands of parameter combinations, but the real simulator is too expensive.

A learned surrogate can provide fast approximations, which are then used by an optimizer to search for better designs.

That turns scientific AI into a bridge between:

- simulation
- prediction
- optimization

## 12. Practical Questions to Ask

When evaluating a scientific AI system, ask:

- what physical law should the model respect?
- what uncertainty should be reported?
- what happens outside the training regime?
- how expensive is the original simulator?
- is the learned model replacing science or supporting it?

## Chapter Problems

1. Why is scientific AI stricter than ordinary benchmark modeling?
2. What is the difference between a simulator and a surrogate?
3. Why are PDEs central in many scientific AI applications?
4. Why is uncertainty especially important in science and engineering?
5. When would a mechanistic model still be preferred over a learned model?

## References

- PINNs overview: https://maziarraissi.github.io/PINNs/
- Probabilistic ML resources: https://probml.github.io/pml-book/

## Summary

Scientific AI is the area where machine learning becomes deeply entangled with mathematics, modeling assumptions, and real physical systems. It is not only about making predictions. It is about making useful, constrained, uncertainty-aware predictions in domains where science and engineering already provide strong structure.
