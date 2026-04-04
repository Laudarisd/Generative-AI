# Problem 1: Equipment Failure Prediction (AI)

## Problem
Predict whether a machine will fail in the next 24 hours using recent sensor history.

## Typical Data
- timestamp
- machine_id
- temperature, pressure, vibration, RPM
- maintenance_flag
- failure_next_24h (label)

## Approach
1. Build lag/rolling features (for example last 1h, 6h, 24h averages).
2. Train baseline (`XGBoost`/`RandomForest`) and compare with sequence model (LSTM).
3. Handle class imbalance (class weights or focal loss).
4. Optimize threshold based on business cost (false negatives usually more costly).

## Metrics
- PR-AUC (important for imbalance)
- Recall at fixed precision
- Lead-time recall (how early failures are detected)

## Starter Code (Tabular Baseline)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
from xgboost import XGBClassifier

# df has precomputed features + target
X = df.drop(columns=["failure_next_24h"])
y = df["failure_next_24h"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum())
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.35).astype(int)

print("PR-AUC:", average_precision_score(y_test, proba))
print(classification_report(y_test, pred, digits=4))
```

## Production Notes
- Monitor drift in key sensors and prediction score distribution.
- Keep model retraining cadence aligned to maintenance cycles.
