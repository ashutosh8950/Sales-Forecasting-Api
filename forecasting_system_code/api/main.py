"""
Forecasting REST API
FastAPI service exposing prediction endpoints.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-ready REST API for weekly sales forecasting. "
        "Trains SARIMA, Prophet, and XGBoost models per US state "
        "and serves predictions for the next N weeks."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR  = Path("outputs/trained_models")
RESULTS_FILE = Path("outputs/model_comparison.json")

# ─── Response schemas ─────────────────────────────────────────────────────────
class WeeklyForecast(BaseModel):
    week: int
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_horizon_weeks: int
    forecasts: List[WeeklyForecast]
    model_metrics: Optional[dict] = None


class ModelInfo(BaseModel):
    state: str
    best_model: str
    metrics: dict


class HealthResponse(BaseModel):
    status: str
    total_states_available: int
    models_dir: str
    timestamp: str


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Models not trained yet. Run train_pipeline.py first."
        )
    with open(RESULTS_FILE) as f:
        return json.load(f)


def _load_model(state: str, model_name: str):
    path = MODELS_DIR / f"{state}_{model_name.lower()}.pkl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {path}. Train the pipeline first."
        )
    return joblib.load(path)


def _future_dates(steps: int, freq: str = "W") -> List[str]:
    today = datetime.today()
    # Align to next Sunday (weekly anchor)
    days_until_sunday = (6 - today.weekday()) % 7 or 7
    start = today + timedelta(days=days_until_sunday)
    dates = pd.date_range(start=start, periods=steps, freq="W")
    return [d.strftime("%Y-%m-%d") for d in dates]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Sales Forecasting API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/forecast/{state}",
            "/forecast/batch",
            "/models",
            "/models/{state}",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    results = _load_results()
    return HealthResponse(
        status="ok",
        total_states_available=len(results),
        models_dir=str(MODELS_DIR),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/models", tags=["Model Info"])
def list_models():
    """List all trained states and their best model."""
    results = _load_results()
    return {
        state: {
            "best_model": v["best_model"],
            "rmse": v["metrics"].get(v["best_model"], {}).get("rmse"),
            "mape": v["metrics"].get(v["best_model"], {}).get("mape"),
        }
        for state, v in results.items()
    }


@app.get("/models/{state}", response_model=ModelInfo, tags=["Model Info"])
def state_model_info(state: str):
    """Get detailed model comparison for a specific state."""
    results = _load_results()
    state_title = state.title().replace("_", " ")
    if state_title not in results:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    res = results[state_title]
    return ModelInfo(state=state_title, best_model=res["best_model"], metrics=res["metrics"])


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecasting"])
def forecast_state(
    state: str,
    weeks: int = Query(default=8, ge=1, le=52, description="Number of weeks to forecast"),
    model: Optional[str] = Query(
        default=None,
        description="Override model selection. Options: SARIMA, Prophet, XGBoost"
    ),
):
    """
    Forecast weekly sales for a given US state.

    - **state**: State name (e.g., `California`, `Texas`)
    - **weeks**: Forecast horizon in weeks (default 8, max 52)
    - **model**: Optionally override the auto-selected best model
    """
    results = _load_results()
    state_title = state.title().replace("_", " ")

    if state_title not in results:
        available = list(results.keys())
        raise HTTPException(
            status_code=404,
            detail=f"State '{state_title}' not found. Available: {available}"
        )

    res = results[state_title]
    model_name = model.strip() if model else res["best_model"]

    # Map user-supplied name to file name
    model_map = {"sarima": "sarima", "prophet": "prophet", "xgboost": "xgboost"}
    model_key = model_map.get(model_name.lower(), model_name.lower())

    loaded_model = _load_model(state_title, model_key)
    preds = loaded_model.predict(steps=weeks)
    dates = _future_dates(steps=weeks)

    forecasts = [
        WeeklyForecast(week=i + 1, date=dates[i], predicted_sales=round(float(preds[i]), 2))
        for i in range(weeks)
    ]

    model_metrics = res["metrics"].get(model_name, None)

    return ForecastResponse(
        state=state_title,
        model_used=model_name,
        forecast_horizon_weeks=weeks,
        forecasts=forecasts,
        model_metrics=model_metrics,
    )


@app.get("/forecast/batch/all", tags=["Forecasting"])
def forecast_all_states(
    weeks: int = Query(default=8, ge=1, le=52, description="Weeks to forecast")
):
    """Forecast next N weeks for ALL states using their best models."""
    results = _load_results()
    output = {}

    for state, res in results.items():
        try:
            model_key = res["best_model"].lower()
            loaded_model = _load_model(state, model_key)
            preds = loaded_model.predict(steps=weeks)
            dates = _future_dates(steps=weeks)
            output[state] = {
                "best_model": res["best_model"],
                "forecasts": [
                    {"week": i + 1, "date": dates[i], "predicted_sales": round(float(preds[i]), 2)}
                    for i in range(weeks)
                ],
            }
        except Exception as e:
            output[state] = {"error": str(e)}

    return {"forecast_horizon_weeks": weeks, "results": output}


# ─── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
