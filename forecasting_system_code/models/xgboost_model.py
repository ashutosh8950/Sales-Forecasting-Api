"""
XGBoost Forecasting Model with Lag Features
Tree-based gradient boosting using engineered temporal features.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "lag_1", "lag_4", "lag_8", "lag_13",
    "rolling_mean_4", "rolling_std_4",
    "rolling_mean_13", "rolling_std_13",
    "week_of_year", "month", "quarter",
    "day_of_week", "holiday_flag",
]


class XGBoostModel:
    """
    XGBoost model that uses pre-engineered lag/rolling/temporal features.
    Recursive multi-step forecasting for future periods.
    """

    def __init__(self, state: str):
        self.state = state
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.last_known = None   # last rows of training data for recursive forecast

    def _prepare(self, df: pd.DataFrame):
        df = df.dropna(subset=FEATURE_COLS + ["sales"])
        X = df[FEATURE_COLS].values
        y = df["sales"].values
        return X, y

    def fit(self, df: pd.DataFrame):
        X, y = self._prepare(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        # Keep the last known rows for recursive forecasting
        self.last_known = df.tail(30).copy()
        return self

    def predict(self, steps: int = 8) -> np.ndarray:
        """
        Recursive multi-step forecast.
        Each step appends predicted value and recomputes lag/rolling features.
        """
        import holidays as hol
        us_holidays = hol.US(years=range(2019, 2026))

        history = self.last_known[["Date", "sales"]].copy()
        preds = []

        last_date = history["Date"].max()

        for i in range(steps):
            next_date = last_date + pd.Timedelta(weeks=1)

            # Build feature row from rolling history
            s = history["sales"]
            row = {
                "lag_1":          s.iloc[-1]                         if len(s) >= 1  else 0,
                "lag_4":          s.iloc[-4]                         if len(s) >= 4  else s.mean(),
                "lag_8":          s.iloc[-8]                         if len(s) >= 8  else s.mean(),
                "lag_13":         s.iloc[-13]                        if len(s) >= 13 else s.mean(),
                "rolling_mean_4": s.iloc[-4:].mean()                 if len(s) >= 2  else s.mean(),
                "rolling_std_4":  s.iloc[-4:].std()                  if len(s) >= 2  else 0,
                "rolling_mean_13":s.iloc[-13:].mean()                if len(s) >= 4  else s.mean(),
                "rolling_std_13": s.iloc[-13:].std()                 if len(s) >= 4  else 0,
                "week_of_year":   next_date.isocalendar().week,
                "month":          next_date.month,
                "quarter":        (next_date.month - 1) // 3 + 1,
                "day_of_week":    next_date.dayofweek,
                "holiday_flag":   int(any(
                    next_date + pd.Timedelta(days=d) in us_holidays for d in range(-3, 4)
                )),
            }

            X_new = np.array([[row[c] for c in FEATURE_COLS]])
            X_scaled = self.scaler.transform(X_new)
            pred = float(self.model.predict(X_scaled)[0])
            pred = max(pred, 0)
            preds.append(pred)

            # Append prediction to history
            new_row = pd.DataFrame({"Date": [next_date], "sales": [pred]})
            history = pd.concat([history, new_row], ignore_index=True)
            last_date = next_date

        return np.array(preds)

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        val_df = val_df.dropna(subset=FEATURE_COLS + ["sales"])
        if val_df.empty:
            return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
        X = val_df[FEATURE_COLS].values
        X_scaled = self.scaler.transform(X)
        preds   = np.maximum(self.model.predict(X_scaled), 0)
        actuals = val_df["sales"].values
        mae  = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}
