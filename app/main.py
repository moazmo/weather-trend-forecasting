"""Weather Trend Forecasting API - FastAPI Application."""
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ============================================
# Model Definition (must match training)
# ============================================
class WeatherMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================
# Load Model & Artifacts
# ============================================
def load_artifacts():
    checkpoint = torch.load(MODELS_DIR / "global_weather_mlp.pt", map_location="cpu", weights_only=False)
    model = WeatherMLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")
    encoder = joblib.load(MODELS_DIR / "country_encoder.joblib")
    stats = {row["country"]: row for _, row in __import__("pandas").read_csv(MODELS_DIR / "country_stats.csv").iterrows()}
    
    return model, scaler, encoder, stats


MODEL, SCALER, ENCODER, COUNTRY_STATS = load_artifacts()
COUNTRIES = sorted(ENCODER.classes_.tolist())
FEATURE_COLS = [
    "country_encoded", "latitude", "longitude", "month", "day_of_month", "day_of_week",
    "day_of_year", "quarter", "is_weekend", "month_sin", "month_cos", "day_sin", "day_cos",
    "day_of_year_sin", "day_of_year_cos", "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "temp_lag_7", "temp_lag_14", "temp_lag_30", "temp_rolling_mean_7", "temp_rolling_mean_14", "temp_rolling_std_7"
]


# ============================================
# Prediction Logic
# ============================================
def predict_forecast(country: str, start_date: str, days: int = 7) -> list[dict]:
    stats = COUNTRY_STATS.get(country)
    if not stats:
        raise ValueError(f"Country '{country}' not found")
    
    temp_history = [float(stats["temp_mean"])] * 30
    forecasts = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    
    for _ in range(days):
        features = {
            "country_encoded": int(stats["country_encoded"]),
            "latitude": stats["latitude"], "longitude": stats["longitude"],
            "month": current.month, "day_of_month": current.day, "day_of_week": current.weekday(),
            "day_of_year": current.timetuple().tm_yday, "quarter": (current.month - 1) // 3 + 1,
            "is_weekend": int(current.weekday() >= 5),
            "month_sin": np.sin(2 * np.pi * current.month / 12),
            "month_cos": np.cos(2 * np.pi * current.month / 12),
            "day_sin": np.sin(2 * np.pi * current.weekday() / 7),
            "day_cos": np.cos(2 * np.pi * current.weekday() / 7),
            "day_of_year_sin": np.sin(2 * np.pi * current.timetuple().tm_yday / 365),
            "day_of_year_cos": np.cos(2 * np.pi * current.timetuple().tm_yday / 365),
            "temp_lag_1": temp_history[-1], "temp_lag_2": temp_history[-2], "temp_lag_3": temp_history[-3],
            "temp_lag_7": temp_history[-7], "temp_lag_14": temp_history[-14], "temp_lag_30": temp_history[-30],
            "temp_rolling_mean_7": np.mean(temp_history[-7:]),
            "temp_rolling_mean_14": np.mean(temp_history[-14:]),
            "temp_rolling_std_7": np.std(temp_history[-7:])
        }
        
        X = np.array([[features[c] for c in FEATURE_COLS]])
        X_scaled = SCALER.transform(X)
        with torch.no_grad():
            pred = MODEL(torch.FloatTensor(X_scaled)).item()
        
        forecasts.append({"date": current.strftime("%Y-%m-%d"), "day": current.strftime("%A"), "temp": round(pred, 1)})
        temp_history.append(pred)
        current += timedelta(days=1)
    
    return forecasts


# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Weather Trend Forecasting", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ForecastRequest(BaseModel):
    country: str
    start_date: str  # YYYY-MM-DD


class ForecastResponse(BaseModel):
    country: str
    forecast: list[dict]
    summary: dict


@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE_DIR / "app" / "templates" / "index.html").read_text()


@app.get("/api/countries")
async def get_countries():
    return {"countries": COUNTRIES}


@app.post("/api/forecast", response_model=ForecastResponse)
async def get_forecast(req: ForecastRequest):
    try:
        forecast = predict_forecast(req.country, req.start_date)
        temps = [f["temp"] for f in forecast]
        trend = "↗ Warming" if temps[-1] > temps[0] else "↘ Cooling" if temps[-1] < temps[0] else "→ Stable"
        return {
            "country": req.country,
            "forecast": forecast,
            "summary": {"min": min(temps), "max": max(temps), "avg": round(np.mean(temps), 1), "trend": trend}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy", "countries": len(COUNTRIES)}
