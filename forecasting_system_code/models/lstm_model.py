"""
LSTM Deep Learning Forecasting Model
Sequence-based time series forecasting using stacked LSTM layers.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 13   # ~3 months of weekly data as input window


def _build_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


class LSTMModel:
    """
    Stacked LSTM with dropout for weekly sales forecasting.
    Uses sliding-window sequences of length LOOKBACK.
    Recursive multi-step prediction for future horizons.
    """

    def __init__(self, state: str, lookback: int = LOOKBACK):
        self.state = state
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_sequence = None

    def _build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss="mse")
        return model

    def fit(self, series: pd.Series, epochs: int = 80, batch_size: int = 16):
        values = series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values).flatten()

        X, y = _build_sequences(scaled, self.lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = self._build_model()
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ]

        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )

        # Save last lookback window for recursive forecasting
        self.last_sequence = scaled[-self.lookback:].copy()
        return self

    def predict(self, steps: int = 8) -> np.ndarray:
        seq = self.last_sequence.copy()
        preds_scaled = []

        for _ in range(steps):
            x_input = seq[-self.lookback:].reshape(1, self.lookback, 1)
            pred = self.model.predict(x_input, verbose=0)[0, 0]
            preds_scaled.append(pred)
            seq = np.append(seq, pred)

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(preds, 0)

    def evaluate(self, val_series: pd.Series) -> dict:
        """
        Evaluate on validation series using recursive one-step prediction
        seeded from training data.
        """
        seq = self.last_sequence.copy()
        preds_scaled = []

        val_scaled = self.scaler.transform(
            val_series.values.reshape(-1, 1)
        ).flatten()

        for actual_val in val_scaled:
            x_input = seq[-self.lookback:].reshape(1, self.lookback, 1)
            pred = self.model.predict(x_input, verbose=0)[0, 0]
            preds_scaled.append(pred)
            seq = np.append(seq, actual_val)  # teacher forcing

        preds   = np.maximum(
            self.scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten(), 0
        )
        actuals = val_series.values
        mae  = np.mean(np.abs(preds - actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}
