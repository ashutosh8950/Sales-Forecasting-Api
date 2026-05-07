"""
Data Preprocessing Module
Handles loading, cleaning, resampling, and feature engineering for sales forecasting.
"""

import pandas as pd
import numpy as np
import holidays
import warnings
warnings.filterwarnings("ignore")


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw Excel data and standardise it."""
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Total": "sales"})
    df = df[["State", "Date", "sales"]].sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample irregular dates to weekly frequency (Sunday anchor = 'W').
    Missing weeks are forward-filled then zero-filled.
    """
    frames = []
    us_holidays = holidays.US(years=range(2019, 2025))

    for state, grp in df.groupby("State"):
        grp = grp.set_index("Date").sort_index()
        grp = grp[~grp.index.duplicated(keep="first")]
        grp = grp.resample("W").sum()          # sum within week
        grp["sales"] = grp["sales"].replace(0, np.nan)
        grp["sales"] = grp["sales"].interpolate(method="time")  # interpolate gaps
        grp["sales"] = grp["sales"].bfill().fillna(0)
        grp["State"] = state
        frames.append(grp.reset_index())

    df_weekly = pd.concat(frames, ignore_index=True)
    return df_weekly


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      - Lag features: t-1, t-7, t-30 (in weeks: lag_1, lag_7, lag_30)
      - Rolling mean/std (4-week and 12-week)
      - Temporal: week_of_year, month, quarter, day_of_week
      - Holiday flag (US federal holidays)
    """
    us_holidays = holidays.US(years=range(2019, 2025))
    frames = []

    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date").copy()
        s = grp["sales"]

        # Lag features (weekly data: lag_1 = prev week, lag_4 = ~month ago)
        grp["lag_1"] = s.shift(1)
        grp["lag_4"] = s.shift(4)    # ~1 month
        grp["lag_8"] = s.shift(8)    # ~2 months
        grp["lag_13"] = s.shift(13)  # ~3 months / quarter

        # Rolling stats (min_periods to avoid too many NaNs)
        grp["rolling_mean_4"] = s.shift(1).rolling(4, min_periods=2).mean()
        grp["rolling_std_4"]  = s.shift(1).rolling(4, min_periods=2).std()
        grp["rolling_mean_13"] = s.shift(1).rolling(13, min_periods=4).mean()
        grp["rolling_std_13"]  = s.shift(1).rolling(13, min_periods=4).std()

        # Temporal features
        grp["week_of_year"] = grp["Date"].dt.isocalendar().week.astype(int)
        grp["month"]        = grp["Date"].dt.month
        grp["quarter"]      = grp["Date"].dt.quarter
        grp["day_of_week"]  = grp["Date"].dt.dayofweek  # 6 = Sunday (week-end anchor)

        # Holiday flag: 1 if the week contains a US public holiday
        grp["holiday_flag"] = grp["Date"].apply(
            lambda d: int(any(d + pd.Timedelta(days=i) in us_holidays for i in range(-3, 4)))
        )

        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    return result


def train_val_split(df: pd.DataFrame, val_weeks: int = 8):
    """
    Time-series-aware split: last `val_weeks` per state go to validation.
    No data leakage — validation is strictly after training.
    """
    train_frames, val_frames = [], []
    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date")
        train_frames.append(grp.iloc[:-val_weeks])
        val_frames.append(grp.iloc[-val_weeks:])
    return pd.concat(train_frames), pd.concat(val_frames)


def get_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    return df[df["State"] == state].sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_clean("data/sales_data.xlsx")
    df_weekly = resample_weekly(df)
    df_feat = add_features(df_weekly)
    train, val = train_val_split(df_feat)
    print(f"Train rows: {len(train)} | Val rows: {len(val)}")
    print(df_feat.columns.tolist())
    print(df_feat.head(3))
