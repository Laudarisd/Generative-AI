# Problem 3: Building Energy Forecasting (AI + Physics)

## Problem
Forecast next-day building energy consumption while respecting physical behavior.

## Typical Data
- weather: outdoor temperature, humidity, solar radiation
- operation: occupancy, HVAC schedule, setpoint
- target: power/energy consumption (kW/kWh)

## Hybrid AI + Physics Idea
Use a data-driven model for accuracy, then enforce soft physical consistency:
- higher cooling load when outdoor temperature rises (in cooling season)
- energy should stay within realistic operational range

## Approach
1. Train baseline regressor (XGBoost/LSTM).
2. Add physics-inspired features (degree days, thermal lag).
3. Add monotonicity penalty or post-check rules.

## Metrics
- MAE, RMSE, MAPE
- physics violation rate (custom)

## Starter Code (Physics-aware penalty, PyTorch)
```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y, temp_outdoor in loader:
    optimizer.zero_grad()
    y_hat = model(x).squeeze(-1)

    # data fit loss
    mse = ((y_hat - y) ** 2).mean()

    # simple physics-inspired penalty:
    # if outdoor temp increases, predicted load should not decrease too much
    temp_diff = temp_outdoor[1:] - temp_outdoor[:-1]
    pred_diff = y_hat[1:] - y_hat[:-1]
    physics_penalty = torch.relu(-(pred_diff * torch.sign(temp_diff))).mean()

    loss = mse + 0.1 * physics_penalty
    loss.backward()
    optimizer.step()
```

## Production Notes
- Retrain by season.
- Track both forecast error and physics-violation trend.
