# Problem 2: Sensor Drift Detection (Statistics)

## Problem
Detect when a sensor's behavior slowly shifts away from normal calibration.

## Typical Data
- timestamp
- sensor_id
- measured_value
- optional reference_value

## Approach
1. Build baseline using a stable historical period.
2. Apply statistical detectors: z-score, EWMA, CUSUM.
3. Trigger alerts when persistent shift is detected (not just one noisy point).

## Why Statistical First?
For drift detection, interpretable and low-latency statistical methods are often strong first-line monitoring tools.

## Metrics
- False alarm rate per day/week
- Mean time to detection
- Detection delay after known shift

## Starter Code (CUSUM + EWMA)
```python
import numpy as np
import pandas as pd

x = df["measured_value"].to_numpy()
mu = np.mean(x[:2000])          # baseline mean
sigma = np.std(x[:2000]) + 1e-8

# EWMA
alpha = 0.1
ewma = np.zeros_like(x, dtype=float)
ewma[0] = x[0]
for i in range(1, len(x)):
    ewma[i] = alpha * x[i] + (1 - alpha) * ewma[i - 1]

# CUSUM
k = 0.5 * sigma
h = 5.0 * sigma
s_pos, s_neg = 0.0, 0.0
alarms = []
for i, v in enumerate(x):
    s_pos = max(0.0, s_pos + (v - mu - k))
    s_neg = min(0.0, s_neg + (v - mu + k))
    if s_pos > h or s_neg < -h:
        alarms.append(i)
        s_pos, s_neg = 0.0, 0.0

print("num_alarms:", len(alarms))
```

## Extension
Use ARIMA residual monitoring when the signal has strong autocorrelation/seasonality.
