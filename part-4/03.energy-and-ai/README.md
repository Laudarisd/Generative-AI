# Energy and AI

Energy and AI is a two-way topic:

- AI consumes energy through training, inference, and data centers
- AI can also improve energy systems through forecasting, optimization, and control

This chapter needs both engineering realism and mathematical thinking.

## 1. Why Energy and AI Matters

Important questions:

- how much electricity do AI systems consume?
- how do data centers affect power grids?
- where can AI reduce waste or emissions?
- how should we optimize energy-aware AI systems?

## 2. AI for Energy Systems

Common applications:

- load forecasting
- renewable generation forecasting
- demand response
- battery management
- fault detection
- smart building control

### Example

If a utility can forecast short-term electricity demand better, it can schedule generation more efficiently and reduce reserve waste.

## 3. Mathematical Formulation

A simple forecasting setup:

```math
\hat{y}_{t+1} = f_\theta(x_t, x_{t-1}, \dots)
```

Typical loss:

```math
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
```

For control or scheduling, optimization may appear:

```math
\min_x \ c^\top x
```

subject to physical and operational constraints.

## 4. AI's Own Energy Footprint

The other side of the topic is AI infrastructure itself:

- training runs
- inference serving
- cooling
- networking
- storage

Important operational ideas:

- performance per watt
- utilization efficiency
- carbon-aware scheduling
- smaller models vs larger models

## 5. Energy-Aware Modeling

Not every problem should use the biggest possible model.

Good engineering questions:

- does a smaller model achieve the target accuracy?
- can quantization reduce serving energy?
- can batching improve hardware efficiency?
- can retrieval reduce repeated compute?

## 6. Tiny Forecasting Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[100], [110], [120], [130], [140]])
y = np.array([108, 118, 128, 137, 149])

model = LinearRegression()
model.fit(X, y)

print(model.predict([[150]]))
```

This toy example is tiny, but it reflects the idea of demand forecasting.

## 7. Problems and Research Directions

- efficient training
- green inference systems
- AI for grid resilience
- energy optimization under uncertainty
- coupling AI with physics-based grid models

## Problems to Think About

1. Why is forecasting important for renewable-heavy grids?
2. What does performance per watt mean in practice?
3. When should a team prioritize a smaller model over a larger one?
4. How can optimization and forecasting work together in energy systems?
5. Why is AI both a consumer and an optimizer of energy?

## References

- IEA Energy and AI report: https://www.iea.org/reports/energy-and-ai
