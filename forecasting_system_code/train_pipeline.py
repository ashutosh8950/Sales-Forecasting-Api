"""
Model Training & Selection Pipeline
Trains all models per state, evaluates on validation set, selects best by RMSE.
Persists trained models and results to disk.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessing import load_and_clean, resample_weekly, add_features, train_val_split, get_state_series
from models.sarima_model import SARIMAModel
from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel

import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = Path("outputs/trained_models")
RESULTS_FILE = Path("outputs/model_comparison.json")
VAL_WEEKS = 8


def train_state(state: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train all models for a single state; return metrics dict."""
    train_s = get_state_series(train_df, state)
    val_s   = get_state_series(val_df,   state)

    metrics = {}

    # ─── SARIMA ────────────────────────────────────────────────────────────────
    try:
        sarima = SARIMAModel(state=state, seasonal_period=13)  # 13-week quarterly
        sarima.fit(train_s.set_index("Date")["sales"])
        m = sarima.evaluate(val_s.set_index("Date")["sales"])
        metrics["SARIMA"] = m
        joblib.dump(sarima, MODELS_DIR / f"{state}_sarima.pkl")
        print(f"  [SARIMA]  RMSE={m['rmse']:,.0f}  MAPE={m['mape']:.1f}%")
    except Exception as e:
        print(f"  [SARIMA]  FAILED: {e}")
        metrics["SARIMA"] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    # ─── Prophet ───────────────────────────────────────────────────────────────
    try:
        prophet = ProphetModel(state=state)
        prophet.fit(train_s[["Date", "sales"]])
        m = prophet.evaluate(val_s[["Date", "sales"]])
        metrics["Prophet"] = m
        joblib.dump(prophet, MODELS_DIR / f"{state}_prophet.pkl")
        print(f"  [Prophet] RMSE={m['rmse']:,.0f}  MAPE={m['mape']:.1f}%")
    except Exception as e:
        print(f"  [Prophet] FAILED: {e}")
        metrics["Prophet"] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    # ─── XGBoost ───────────────────────────────────────────────────────────────
    try:
        xgb = XGBoostModel(state=state)
        xgb.fit(train_s)
        m = xgb.evaluate(val_s)
        metrics["XGBoost"] = m
        joblib.dump(xgb, MODELS_DIR / f"{state}_xgboost.pkl")
        print(f"  [XGBoost] RMSE={m['rmse']:,.0f}  MAPE={m['mape']:.1f}%")
    except Exception as e:
        print(f"  [XGBoost] FAILED: {e}")
        metrics["XGBoost"] = {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    # ─── Select best model ─────────────────────────────────────────────────────
    valid = {k: v for k, v in metrics.items() if not np.isnan(v["rmse"])}
    best_model = min(valid, key=lambda k: valid[k]["rmse"]) if valid else "Prophet"
    print(f"  ✓ Best model for {state}: {best_model}\n")

    return {"metrics": metrics, "best_model": best_model}


def run_pipeline(data_path: str = "data/sales_data.xlsx"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TIME SERIES FORECASTING PIPELINE")
    print("=" * 60)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading & preprocessing data ...")
    df_raw    = load_and_clean(data_path)
    df_weekly = resample_weekly(df_raw)
    df_feat   = add_features(df_weekly)
    train_df, val_df = train_val_split(df_feat, val_weeks=VAL_WEEKS)
    print(f"  States: {df_feat['State'].nunique()}  |  "
          f"Train: {len(train_df)} rows  |  Val: {len(val_df)} rows")

    # ── Train all states ───────────────────────────────────────────────────────
    print("\n[2/3] Training models per state ...")
    all_results = {}
    states = sorted(df_feat["State"].unique())

    for state in states:
        print(f"\n── {state} ──")
        result = train_state(state, train_df, val_df)
        all_results[state] = result

    # ── Save results ───────────────────────────────────────────────────────────
    print("\n[3/3] Saving results ...")
    def _serialise(obj):
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=_serialise)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL SELECTION SUMMARY")
    print("=" * 60)
    rows = []
    for state, res in all_results.items():
        best = res["best_model"]
        m    = res["metrics"].get(best, {})
        rows.append({
            "State": state,
            "Best Model": best,
            "RMSE": m.get("rmse"),
            "MAE":  m.get("mae"),
            "MAPE%": m.get("mape"),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv("outputs/model_selection_summary.csv", index=False)
    print(f"\nResults saved → {RESULTS_FILE}")
    print("Pipeline complete ✓")
    return all_results


if __name__ == "__main__":
    run_pipeline()
