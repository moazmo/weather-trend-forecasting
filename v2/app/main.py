"""V2 Weather Trend Forecasting API - Transformer Model."""
from datetime import datetime, timedelta
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import json
import math

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
DATA_DIR = BASE_DIR.parent / "data" / "processed"


# ============================================
# Transformer Model Definition
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class WeatherTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=4, dropout=0.2, seq_len=30, pred_len=7):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, pred_len))
    
    def forward(self, x):
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])


# ============================================
# Load Model & Artifacts
# ============================================
def load_artifacts():
    # Transformer Model
    checkpoint = torch.load(V2_MODELS_DIR / "transformer_model.pt", map_location="cpu", weights_only=False)
    model = WeatherTransformer(
        input_dim=checkpoint['input_dim'],
        d_model=checkpoint['d_model'],
        nhead=checkpoint['nhead'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
        seq_len=checkpoint['seq_len'],
        pred_len=checkpoint['pred_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_cols = checkpoint['feature_cols']
    seq_len = checkpoint['seq_len']
    pred_len = checkpoint['pred_len']
    
    # Scaler
    scaler = joblib.load(V2_MODELS_DIR / "transformer_scaler.joblib")
    
    # Location stats
    stats_df = pd.read_csv(V2_MODELS_DIR / "location_stats.csv")
    stats = {row["country"]: row.to_dict() for _, row in stats_df.iterrows()}
    
    # Historical data
    historical = {}
    hist_path = DATA_DIR / "weather_cleaned.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path, parse_dates=["date"])
        for _, row in hist_df.iterrows():
            key = (row["country"], row["date"].strftime("%Y-%m-%d"))
            historical[key] = round(row["temperature_celsius"], 1)
    
    return model, scaler, stats, stats_df, feature_cols, seq_len, pred_len, historical


MODEL, SCALER, COUNTRY_STATS, STATS_DF, FEATURE_COLS, SEQ_LEN, PRED_LEN, HISTORICAL_DATA = load_artifacts()
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
    if abs_lat <= 23.5: return 'Tropical', 0
    elif abs_lat <= 35: return 'Subtropical', 1
    elif abs_lat <= 55: return 'Temperate', 2
    elif abs_lat <= 66.5: return 'Continental', 3
    return 'Polar', 4


# ============================================
# LSTM Prediction
# ============================================
def predict_forecast(lat: float, lon: float, start_date: str) -> tuple:
    nearest_country, _ = find_nearest_country(lat, lon)
    stats = COUNTRY_STATS.get(nearest_country, {})
    climate_zone, climate_zone_encoded = get_climate_zone(lat)
    hemisphere_encoded = 1 if lat >= 0 else 0
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Build 30-day sequence for LSTM input
    real_history = []
    for i in range(SEQ_LEN, 0, -1):
        hist_date = (start_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        key = (nearest_country, hist_date)
        if key in HISTORICAL_DATA:
            real_history.append(HISTORICAL_DATA[key])
    
    # Use real data if available, else monthly average
    if len(real_history) == SEQ_LEN:
        temp_history = real_history
        using_real_lags = True
    else:
        month_key = f"temp_mean_month_{start_dt.month}"
        month_avg = stats.get(month_key, stats.get("temp_mean", 20.0))
        temp_history = [float(month_avg)] * SEQ_LEN
        using_real_lags = False
    
    # Build feature sequence
    sequence = []
    for i in range(SEQ_LEN):
        day = start_dt - timedelta(days=SEQ_LEN - i)
        features = {
            'latitude': lat, 'longitude': lon,
            'abs_latitude': abs(lat), 'latitude_normalized': abs(lat) / 90.0,
            'hemisphere_encoded': hemisphere_encoded, 'climate_zone_encoded': climate_zone_encoded,
            'month': day.month, 'day_of_month': day.day, 'day_of_week': day.weekday(),
            'day_of_year': day.timetuple().tm_yday, 'quarter': (day.month - 1) // 3 + 1,
            'is_weekend': int(day.weekday() >= 5),
            'month_sin': np.sin(2 * np.pi * day.month / 12),
            'month_cos': np.cos(2 * np.pi * day.month / 12),
            'day_sin': np.sin(2 * np.pi * day.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * day.weekday() / 7),
            'day_of_year_sin': np.sin(2 * np.pi * day.timetuple().tm_yday / 365),
            'day_of_year_cos': np.cos(2 * np.pi * day.timetuple().tm_yday / 365),
            'temperature_celsius': temp_history[i]
        }
        sequence.append([features[c] for c in FEATURE_COLS])
    
    # Scale and predict
    X = np.array([sequence])
    X_scaled = SCALER.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    with torch.no_grad():
        preds = MODEL(torch.FloatTensor(X_scaled)).numpy()[0]
    
    # Build forecast response
    forecasts = []
    for i in range(PRED_LEN):
        date = start_dt + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        actual = HISTORICAL_DATA.get((nearest_country, date_str))
        
        forecast_item = {"date": date_str, "day": date.strftime("%A"), "temp": round(float(preds[i]), 1)}
        if actual is not None:
            forecast_item["actual"] = actual
        forecasts.append(forecast_item)
    
    return forecasts, climate_zone, nearest_country, using_real_lags


# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Weather Trend Forecasting V2", version="2.2.0-lstm")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LocationRequest(BaseModel):
    lat: float
    lon: float
    start_date: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/countries")
async def get_countries():
    return [{"name": row["country"], "lat": row["latitude"], "lon": row["longitude"],
             "climate_zone": row["climate_zone"]} for _, row in STATS_DF.iterrows()]


@app.get("/api/nearest")
async def get_nearest(lat: float, lon: float):
    country, distance = find_nearest_country(lat, lon)
    stats = COUNTRY_STATS.get(country, {})
    climate_zone, _ = get_climate_zone(lat)
    return {"country": country, "distance_km": distance, "climate_zone": climate_zone,
            "hemisphere": "Northern" if lat >= 0 else "Southern",
            "country_lat": stats.get("latitude"), "country_lon": stats.get("longitude")}


@app.post("/api/forecast")
async def get_forecast(req: LocationRequest):
    try:
        forecast, climate_zone, nearest_country, using_real_lags = predict_forecast(req.lat, req.lon, req.start_date)
        temps = [f["temp"] for f in forecast]
        trend = "↗ Warming" if temps[-1] > temps[0] else "↘ Cooling" if temps[-1] < temps[0] else "→ Stable"
        
        actuals = [f.get("actual") for f in forecast if f.get("actual") is not None]
        preds_with_actual = [f["temp"] for f in forecast if f.get("actual") is not None]
        
        summary = {
            "min": min(temps), "max": max(temps), "avg": round(np.mean(temps), 1), "trend": trend,
            "has_actual": len(actuals) > 0, "actual_days": len(actuals), "using_real_lags": using_real_lags,
            "model": "transformer"
        }
        if actuals:
            summary["mae"] = round(np.mean([abs(a - p) for a, p in zip(actuals, preds_with_actual)]), 2)
        
        return {"location": {"lat": req.lat, "lon": req.lon}, "nearest_country": nearest_country,
                "climate_zone": climate_zone, "hemisphere": "Northern" if req.lat >= 0 else "Southern",
                "forecast": forecast, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.3-transformer", "model": "transformer", "countries": len(COUNTRIES),
            "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "historical_records": len(HISTORICAL_DATA)}
