"""V2 Weather Trend Forecasting API - Location-Based Model."""
from datetime import datetime, timedelta
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
V2_MODELS_DIR = BASE_DIR / "models"
V1_MODELS_DIR = BASE_DIR.parent / "models"  # For fallback country stats


# ============================================
# Model Definition
# ============================================
class LocationMLP(nn.Module):
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
    # V2 Location Model
    checkpoint = torch.load(V2_MODELS_DIR / "location_model.pt", map_location="cpu", weights_only=False)
    model = LocationMLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_cols = checkpoint["feature_cols"]
    
    # V2 Scaler
    scaler = joblib.load(V2_MODELS_DIR / "location_scaler.joblib")
    
    # Location stats with climate zones
    stats_df = pd.read_csv(V2_MODELS_DIR / "location_stats.csv")
    stats = {row["country"]: row.to_dict() for _, row in stats_df.iterrows()}
    
    # Climate zones
    with open(V2_MODELS_DIR / "climate_zones.json", "r") as f:
        climate_data = json.load(f)
    
    return model, scaler, stats, stats_df, feature_cols, climate_data


MODEL, SCALER, COUNTRY_STATS, STATS_DF, FEATURE_COLS, CLIMATE_DATA = load_artifacts()
COUNTRIES = sorted(COUNTRY_STATS.keys())


# ============================================
# Geographic Utilities
# ============================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def find_nearest_country(lat: float, lon: float) -> tuple[str, float]:
    min_dist, nearest = float('inf'), None
    for _, row in STATS_DF.iterrows():
        dist = haversine(lat, lon, row['latitude'], row['longitude'])
        if dist < min_dist:
            min_dist, nearest = dist, row['country']
    return nearest, round(min_dist, 1)


def get_climate_zone(lat: float) -> tuple[str, int]:
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        return 'Tropical', 0
    elif abs_lat <= 35:
        return 'Subtropical', 1
    elif abs_lat <= 55:
        return 'Temperate', 2
    elif abs_lat <= 66.5:
        return 'Continental', 3
    return 'Polar', 4


# ============================================
# Prediction Logic (V2 - Location-Based)
# ============================================
def predict_forecast(lat: float, lon: float, start_date: str, days: int = 7) -> list[dict]:
    # Get climate info for location
    climate_zone, climate_zone_encoded = get_climate_zone(lat)
    hemisphere_encoded = 1 if lat >= 0 else 0
    abs_latitude = abs(lat)
    latitude_normalized = abs_latitude / 90.0
    
    # Find nearest country for month-specific temps
    nearest_country, _ = find_nearest_country(lat, lon)
    stats = COUNTRY_STATS.get(nearest_country, {})
    
    # Initialize temperature history
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    month_key = f"temp_mean_month_{start_dt.month}"
    month_avg = stats.get(month_key, stats.get("temp_mean", 20.0))
    temp_history = [float(month_avg)] * 30
    
    forecasts = []
    current = start_dt
    
    for _ in range(days):
        # Build feature vector matching training order
        features = {
            # Geographic
            'latitude': lat,
            'longitude': lon,
            'abs_latitude': abs_latitude,
            'latitude_normalized': latitude_normalized,
            'hemisphere_encoded': hemisphere_encoded,
            'climate_zone_encoded': climate_zone_encoded,
            
            # Temporal
            'month': current.month,
            'day_of_month': current.day,
            'day_of_week': current.weekday(),
            'day_of_year': current.timetuple().tm_yday,
            'quarter': (current.month - 1) // 3 + 1,
            'is_weekend': int(current.weekday() >= 5),
            
            # Cyclical
            'month_sin': np.sin(2 * np.pi * current.month / 12),
            'month_cos': np.cos(2 * np.pi * current.month / 12),
            'day_sin': np.sin(2 * np.pi * current.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * current.weekday() / 7),
            'day_of_year_sin': np.sin(2 * np.pi * current.timetuple().tm_yday / 365),
            'day_of_year_cos': np.cos(2 * np.pi * current.timetuple().tm_yday / 365),
            
            # Lag features
            'temp_lag_1': temp_history[-1],
            'temp_lag_2': temp_history[-2],
            'temp_lag_3': temp_history[-3],
            'temp_lag_7': temp_history[-7],
            'temp_lag_14': temp_history[-14],
            'temp_lag_30': temp_history[-30],
            
            # Rolling stats
            'temp_rolling_mean_7': np.mean(temp_history[-7:]),
            'temp_rolling_mean_14': np.mean(temp_history[-14:]),
            'temp_rolling_std_7': np.std(temp_history[-7:])
        }
        
        X = np.array([[features[c] for c in FEATURE_COLS]])
        X_scaled = SCALER.transform(X)
        with torch.no_grad():
            pred = MODEL(torch.FloatTensor(X_scaled)).item()
        
        forecasts.append({
            "date": current.strftime("%Y-%m-%d"),
            "day": current.strftime("%A"),
            "temp": round(pred, 1)
        })
        temp_history.append(pred)
        current += timedelta(days=1)
    
    return forecasts, climate_zone, nearest_country


# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Weather Trend Forecasting V2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LocationRequest(BaseModel):
    lat: float
    lon: float
    start_date: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/nearest")
async def get_nearest(lat: float, lon: float):
    country, distance = find_nearest_country(lat, lon)
    climate_zone, _ = get_climate_zone(lat)
    hemisphere = "Northern" if lat >= 0 else "Southern"
    return {
        "country": country,
        "distance_km": distance,
        "climate_zone": climate_zone,
        "hemisphere": hemisphere
    }


@app.post("/api/forecast")
async def get_forecast(req: LocationRequest):
    try:
        forecast, climate_zone, nearest_country = predict_forecast(req.lat, req.lon, req.start_date)
        temps = [f["temp"] for f in forecast]
        trend = "↗ Warming" if temps[-1] > temps[0] else "↘ Cooling" if temps[-1] < temps[0] else "→ Stable"
        hemisphere = "Northern" if req.lat >= 0 else "Southern"
        
        return {
            "location": {"lat": req.lat, "lon": req.lon},
            "nearest_country": nearest_country,
            "climate_zone": climate_zone,
            "hemisphere": hemisphere,
            "forecast": forecast,
            "summary": {
                "min": min(temps),
                "max": max(temps),
                "avg": round(np.mean(temps), 1),
                "trend": trend
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.0", "model": "location-based", "countries": len(COUNTRIES)}
