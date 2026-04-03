# Energy and AI

Energy and AI is a two-way topic:

- AI consumes energy through training, inference, and data centers
- AI can also improve energy systems through forecasting, optimization, and control

This chapter needs both engineering realism and mathematical thinking.

## 1. Why Energy and AI Matters

Important questions include:

- how much electricity do AI systems consume?
- how do data centers affect power grids?
- where can AI reduce waste or emissions?
- how should we optimize energy-aware AI systems?
- how can forecasting and control reduce operational inefficiency?

## 2. AI for Energy Systems

Common applications include:

- load forecasting
- renewable generation forecasting
- demand response
- battery management
- fault detection
- smart building control
- grid anomaly detection
- energy market forecasting

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

This is where AI and operations research often meet.

## 4. AI's Own Energy Footprint

The other side of the topic is AI infrastructure itself:

- training runs
- inference serving
- cooling
- networking
- storage

Important operational ideas include:

- performance per watt
- utilization efficiency
- carbon-aware scheduling
- smaller models vs larger models
- batching and hardware efficiency

## 5. Why Efficiency Is Not Optional

A model that is 1% better but 5 times more expensive to serve may be a poor engineering decision depending on the application.

Energy-aware AI asks practical questions:

- can we quantize?
- can we batch more efficiently?
- can we schedule jobs during cleaner grid periods?
- can we use a smaller model with similar utility?

## 6. Energy-Aware Modeling

Not every problem should use the biggest possible model.

Good engineering questions:

- does a smaller model achieve the target accuracy?
- can quantization reduce serving energy?
- can batching improve hardware efficiency?
- can retrieval reduce repeated compute?

## 7. Tiny Forecasting Example

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

## 8. Optimization Example Intuition

Suppose a battery can charge or discharge over time. Then a scheduler may optimize cost while respecting:

- storage limits
- charge/discharge rates
- demand constraints
- electricity price forecasts

The resulting system is not only predictive. It is predictive plus prescriptive.

## 9. AI for Renewable Integration

Renewables such as wind and solar introduce uncertainty.

AI can help by forecasting:

- solar irradiance
- wind generation
- demand peaks
- storage requirements

Better forecasts help grids absorb renewables more efficiently.

## 10. Data Center Perspective

Large AI systems also affect the grid because data centers draw substantial power and can create concentrated demand.

That makes this topic important from both directions:

- AI as an optimizer of energy systems
- AI as an energy-intensive system that must be managed carefully

## 11. Problems and Research Directions

- efficient training
- green inference systems
- AI for grid resilience
- energy optimization under uncertainty
- coupling AI with physics-based grid models
- carbon-aware scheduling for compute workloads

## Problems to Think About

1. Why is forecasting important for renewable-heavy grids?
2. What does performance per watt mean in practice?
3. When should a team prioritize a smaller model over a larger one?
4. How can optimization and forecasting work together in energy systems?
5. Why is AI both a consumer and an optimizer of energy?

## References

- IEA Energy and AI report: https://www.iea.org/reports/energy-and-ai

## Summary

Energy and AI is not only about training cost. It is about how learning systems interact with real infrastructure, real constraints, and real environmental tradeoffs. That makes it both a systems topic and a mathematical optimization topic.
