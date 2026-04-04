# Problem 5: Demand Forecasting - ARIMA vs LSTM (Stat + AI)

## Problem
Forecast daily demand and choose the right model family based on data behavior and deployment constraints.

## Typical Data
- date
- demand
- optional exogenous variables: temperature, holiday_flag, promotion

## Model Selection Logic
- ARIMA/SARIMA: strong baseline for linear, structured, seasonal univariate series.
- LSTM/Transformer: better for nonlinear multivariate dependencies with larger data.

## ARIMA(p,d,q) Refresher
- `p`: number of lagged observations (AR)
- `d`: differencing order to improve stationarity (I)
- `q`: number of lagged forecast errors (MA)

## Metrics
- MAE / RMSE / MAPE
- rolling backtest stability across time windows

## Starter Code (ARIMA baseline)
```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

train = series[:-30]
test = series[-30:]

model = ARIMA(train, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=len(test))

print("MAE:", mean_absolute_error(test, forecast))
```

## Starter Code (LSTM skeleton)
```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=1, h=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
```

## Practical Recommendation
Always start with ARIMA/SARIMA as a baseline before moving to deep sequence models.
