# Problem 4: Pipeline Anomaly Detection (Hybrid AI + Statistical Rules)

## Problem
Detect leaks/blockage anomalies from multivariate sensor streams (pressure, flow, temperature).

## Typical Data
- timestamp
- pressure_in, pressure_out
- flow_rate
- temperature
- optional valve state / pump state

## Hybrid Strategy
- AI model learns complex normal patterns (LSTM Autoencoder).
- Statistical layer applies robust thresholding and CUSUM for change confirmation.

## Approach
1. Train LSTM autoencoder on normal periods.
2. Compute reconstruction error as anomaly score.
3. Apply adaptive threshold + CUSUM to reduce false alarms.
4. Add simple physical checks (pressure drop-flow consistency).

## Metrics
- Event-level recall
- False alarms per day
- Mean detection delay

## Starter Code (Error + Threshold)
```python
import numpy as np

# reconstruction_error: per-window anomaly score from model
scores = np.array(reconstruction_error)
threshold = np.percentile(scores[:2000], 99.5)  # baseline quantile
pred = (scores > threshold).astype(int)

# optional alert smoothing: require 3 hits in 5 windows
alerts = np.zeros_like(pred)
for i in range(4, len(pred)):
    if pred[i-4:i+1].sum() >= 3:
        alerts[i] = 1
```

## Extra Physics Rule Example
- If `flow_rate` decreases sharply but pressure gradient increases unusually, raise anomaly confidence.
