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

## 2. Mathematical Ingredients

Scientific AI typically combines:

- calculus
- linear algebra
- differential equations
- optimization
- probability
- numerical analysis

That is why the new `math-cheatsheet/` section exists separately.

## 3. Model Families in Scientific AI

- PINNs
- neural operators
- Gaussian processes
- surrogate models
- hybrid mechanistic + data-driven models
- graph neural networks for scientific systems

## 4. Example: Surrogate Modeling

Suppose a full simulator is expensive. A neural network surrogate can learn:

```math
\hat{u} = f_\theta(\text{parameters}, \text{boundary conditions}, \text{inputs})
```

This can accelerate design loops and uncertainty studies.

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

## 7. Chapter Problems

1. Why is scientific AI stricter than ordinary benchmark modeling?
2. What is the difference between a simulator and a surrogate?
3. Why are PDEs central in many scientific AI applications?
4. Why is uncertainty especially important in science and engineering?
5. When would a mechanistic model still be preferred over a learned model?

## References

- PINNs overview: https://maziarraissi.github.io/PINNs/
- Probabilistic ML resources: https://probml.github.io/pml-book/
