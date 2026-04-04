# Brief Q&A + Code Examples (Fast Revision)

Use this as a quick interview sheet: 1-2 line answers.

## 1) Core ML (Brief)
- Bias vs variance: Bias = underfit from simple assumptions; variance = overfit from data sensitivity.
- Overfitting fix: Check leakage, add regularization, simplify model, use early stopping and better data.
- Classification vs regression: Discrete label vs continuous value prediction.
- Cross-validation: Repeated train/val splits to estimate generalization stability.
- L1 vs L2: L1 promotes sparsity; L2 shrinks weights smoothly.
- Gradient descent: Move parameters opposite gradient to minimize loss.
- SGD: Mini-batch gradient descent with noisy but efficient updates.
- Precision/recall/F1: Positive correctness / positive coverage / balanced harmonic mean.
- ROC-AUC: Threshold-independent ranking quality.
- Data leakage: Future/test information accidentally used in training.
- Normalization vs standardization: Range scaling vs zero-mean unit-variance scaling.
- Curse of dimensionality: Distance and sample efficiency degrade in high dimensions.

## 2) Statistics + Anomaly (Brief)
- Probability vs likelihood: Data given parameters vs parameters given observed data.
- MLE: Choose parameters maximizing observed data likelihood.
- Bayesian inference: Posterior is updated belief after seeing data.
- p-value: Probability of equally/more extreme data under null.
- Convex vs non-convex: Single global basin vs multiple local minima/saddles.
- Hessian: Second-derivative matrix showing curvature/conditioning.
- Calibration: Predicted probability should match true frequency.
- Anomaly detection: Find observations that strongly deviate from expected behavior.
- Anomaly metrics: Precision, recall, F1, PR-AUC, false alarm rate, detection delay.
- CUSUM: Cumulative sum change detector for small persistent mean shifts.
- CUSUM vs EWMA: CUSUM catches sustained shifts quickly; EWMA smooths and tracks trend drift.
- ARIMA (often mistyped RMIA): AR + I + MA model for univariate time-series.
- ARIMA anomaly method: Fit ARIMA, monitor residuals, flag statistically large residuals.
- Autoencoder anomaly detection: Train on normal samples; high reconstruction error => anomaly.
- GAN anomaly detection: Learn normal data distribution; poorly generated/reconstructed samples are anomalous.
- 1D-CNN anomaly detection: Good for vibration/sensor windows and local temporal pattern shifts.
- LOF: Lower local density than neighbors => outlier.
- One-Class SVM: Learn normal frontier; outside-boundary points => anomalies.
- Robust Covariance (Elliptic Envelope): Gaussian-like density model; low-probability points are flagged.
- Foundation-model anomaly detection: Use pretrained signal representations + lightweight anomaly head.

## 3) Deep Learning + LLM (Brief)
- CNN: Convolutional model for spatial patterns.
- RNN/LSTM/GRU: Sequential models; LSTM/GRU improve long-range gradient flow.
- Transformer: Attention-based sequence model with parallel token processing.
- Attention formula: softmax(QK^T / sqrt(d_k))V.
- Encoder vs decoder: Encoder understands context; decoder generates autoregressively.
- Encoder-only vs decoder-only vs enc-dec: Understanding vs generation vs seq2seq mapping.
- Causal mask: Blocks attention to future tokens in generation.
- Cross-attention: Decoder attends to encoder outputs.
- BatchNorm vs LayerNorm: BatchNorm uses batch stats; LayerNorm normalizes per token/sample features.
- Optimizer vs activation: Optimizer updates weights; activation adds nonlinearity.
- Adam vs AdamW: AdamW decouples weight decay and is preferred in many modern setups.
- Dropout: Randomly drop activations during training to reduce co-adaptation.
- Residual connection: Skip path that stabilizes deep optimization.
- LoRA/QLoRA: Parameter-efficient tuning with low-rank adapters (QLoRA adds quantized base).
- RAG: Retrieve evidence, then generate with grounded context.
- Hallucination reduction: Retrieval, constraints, verification, better prompts/data.

## 4) Time-Series (Brief)
- Stationarity: Statistical properties do not change over time.
- Autocorrelation: Correlation with lagged self.
- Seasonality: Repeating periodic pattern.
- Forecasting horizon: Future window being predicted.
- Sliding window: Convert sequence into supervised samples via rolling windows.
- Non-stationary handling: Differencing, transforms, rolling retrain, drift monitoring.
- ESN: Fixed recurrent reservoir, train only readout for efficient temporal modeling.

## 5) Production + Reliability (Brief)
- Offline good, online bad: Usually drift, skew, integration bugs, or monitoring gaps.
- Real-time design: Define latency SLO, optimize model/runtime, add fallback path.
- Drift handling: Detect shift, alert, retrain/recalibrate, validate, redeploy.
- Reliability: Canary, rollback, observability, runbooks, on-call ownership.
- Safety-critical AI: Uncertainty estimates, conservative thresholds, human override.

## 6) Leadership (Brief)
- Conflict: Align goals, clarify evidence, decide by agreed criteria.
- Failure: Own quickly, communicate clearly, fix root cause, prevent repeat.
- Prioritization: Impact x risk x effort with stakeholder alignment.
- Mentoring: Small milestones, frequent feedback, increasing ownership.

---

## 7) Code Examples

### A) Minimal PyTorch training loop
```python
import torch

model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

### B) Mixed precision + gradient clipping
```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### C) Early stopping
```python
best, wait, patience = float("inf"), 0, 5
for epoch in range(epochs):
    train_one_epoch()
    val_loss = validate()
    if val_loss < best:
        best, wait = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            break
```

### D) BatchNorm vs LayerNorm
```python
import torch.nn as nn

cnn = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
seq = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU())
```

### E) Decoder causal mask
```python
import torch
T = 8
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
```

### F) Simple z-score anomaly detector
```python
import numpy as np

x = np.array(series)
z = (x - x.mean()) / (x.std() + 1e-8)
anomaly_idx = np.where(np.abs(z) > 3.0)[0]
```

### G) Simple CUSUM (mean shift)
```python
import numpy as np

x = np.array(series)
target, k, h = x.mean(), 0.5, 5.0
s_pos = s_neg = 0.0
alarms = []
for i, v in enumerate(x):
    s_pos = max(0.0, s_pos + (v - target - k))
    s_neg = min(0.0, s_neg + (v - target + k))
    if s_pos > h or s_neg < -h:
        alarms.append(i)
        s_pos = s_neg = 0.0
```

### H) ARIMA forecast (statsmodels)
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=7)
residuals = model.resid
```

### I) Autoencoder anomaly score (PyTorch)
```python
import torch
import torch.nn as nn

ae = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 64))
x = torch.randn(128, 64)  # normal training batch
recon = ae(x)
err = ((x - recon) ** 2).mean(dim=1)  # sample-wise reconstruction error
anomaly_mask = err > err.mean() + 3 * err.std()
```

### J) LOF example (scikit-learn)
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
pred = lof.fit_predict(X)   # -1 anomaly, 1 normal
scores = -lof.negative_outlier_factor_
```

### K) One-Class SVM example (scikit-learn)
```python
from sklearn.svm import OneClassSVM

oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
oc.fit(X_train_normal)
pred = oc.predict(X_test)   # -1 anomaly, 1 normal
```
