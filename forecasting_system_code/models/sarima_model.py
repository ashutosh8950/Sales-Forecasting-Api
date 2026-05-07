"""
SARIMA Forecasting Model
Seasonal ARIMA with automatic order selection via AIC grid search.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings
warnings.filterwarnings("ignore")


def _aic_grid_search(series: pd.Series, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2),
                     seasonal_period=52):
    """Minimal AIC-based grid search for (p,d,q)(P,D,Q,s) orders."""
    best_aic, best_order, best_seasonal = np.inf, (1, 1, 1), (1, 1, 1, seasonal_period)

    for p, d, q in product(range(*p_range), range(*d_range), range(*q_range)):
        for P, D, Q in product([0, 1], [0, 1], [0, 1]):
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=50)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, seasonal_period)
            except Exception:
                continue

    return best_order, best_seasonal


class SARIMAModel:
    """
    SARIMA model wrapper.
    Uses weekly data (s=52 for yearly seasonality).
    Falls back to simple ARIMA if grid search fails.
    """

    def __init__(self, state: str, seasonal_period: int = 52):
        self.state = state
        self.seasonal_period = seasonal_period
        self.order = None
        self.seasonal_order = None
        self.fitted = None

    def fit(self, series: pd.Series, fast: bool = True):
        """
        Fit SARIMA. If fast=True uses fixed (1,1,1)(1,1,1,s) order.
        Set fast=False for AIC grid search (slower but potentially better).
        """
        if fast:
            self.order = (1, 1, 1)
            self.seasonal_order = (1, 1, 1, self.seasonal_period)
        else:
            self.order, self.seasonal_order = _aic_grid_search(
                series, seasonal_period=self.seasonal_period
            )

        try:
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted = model.fit(disp=False, maxiter=100)
        except Exception:
            # fallback: plain ARIMA(1,1,1)
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False)
            self.fitted = model.fit(disp=False)

        return self

    def predict(self, steps: int = 8) -> np.ndarray:
        forecast = self.fitted.forecast(steps=steps)
        return np.maximum(forecast.values, 0)

    def evaluate(self, val_series: pd.Series) -> dict:
        n = len(val_series)
        preds = self.predict(steps=n)
        actuals = val_series.values
        mae  = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}
