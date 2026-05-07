"""
Facebook Prophet Forecasting Model
Handles trend + seasonality + holidays automatically.
"""

import numpy as np
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


class ProphetModel:
    """
    Wrapper around Facebook Prophet.
    Adds US holidays and weekly/yearly seasonality.
    """

    def __init__(self, state: str):
        self.state = state
        self.model = None
        self.last_date = None

    def fit(self, df: pd.DataFrame):
        """
        df must have columns ['Date', 'sales'].
        """
        prophet_df = df[["Date", "sales"]].rename(columns={"Date": "ds", "sales": "y"})
        prophet_df = prophet_df.dropna(subset=["y"])

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
        )
        self.model.add_country_holidays(country_name="US")
        self.model.fit(prophet_df)
        self.last_date = prophet_df["ds"].max()
        return self

    def predict(self, steps: int = 8) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=steps, freq="W")
        forecast = self.model.predict(future)
        preds = forecast.tail(steps)["yhat"].values
        return np.maximum(preds, 0)

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        prophet_df = val_df[["Date", "sales"]].rename(columns={"Date": "ds", "sales": "y"})
        forecast = self.model.predict(prophet_df[["ds"]])
        preds   = np.maximum(forecast["yhat"].values, 0)
        actuals = val_df["sales"].values
        mae  = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}
