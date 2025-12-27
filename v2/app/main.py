"""V4 Weather Trend Forecasting API - Advanced Transformer with GRN + Open-Meteo Integration."""
from datetime import datetime, timedelta
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from functools import lru_cache
import math

import httpx
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================
# Advanced Transformer Model Definition
# ============================================
class GatedResidualNetwork(nn.Module):
    """GRN: Allows model to learn to skip irrelevant inputs."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return self.layer_norm(gate * output + self.skip(x))


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


class AdvancedWeatherTransformer(nn.Module):
    """Enhanced Transformer with Gated Residual Networks for weather forecasting."""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, dropout=0.2, seq_len=30, pred_len=7):
        super().__init__()
        self.d_model = d_model
        self.input_grn = GatedResidualNetwork(input_dim, d_model * 2, d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(d_model // 2, pred_len))
    
    def forward(self, x):
        x = self.input_grn(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_grn(x[:, -1, :])
        return self.output_head(x)


# ============================================
# Open-Meteo API Integration
# ============================================
@lru_cache(maxsize=500)
def fetch_open_meteo_history(lat: float, lon: float, end_date: str, days: int = 30) -> list[dict] | None:
    """Fetch historical weather data from Open-Meteo Archive API."""
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days)
        
        params = {
            "latitude": round(lat, 2), "longitude": round(lon, 2),
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": (end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max,precipitation_sum,cloud_cover_mean",
            "timezone": "auto"
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        history = []
        for i, date in enumerate(dates):
            history.append({
                'date': date,
                'temperature_celsius': daily.get("temperature_2m_mean", [20.0] * len(dates))[i] or 20.0,
                'humidity': daily.get("relative_humidity_2m_mean", [50.0] * len(dates))[i] or 50.0,
                'pressure_mb': daily.get("surface_pressure_mean", [1013.0] * len(dates))[i] or 1013.0,
                'wind_kph': (daily.get("wind_speed_10m_max", [10.0] * len(dates))[i] or 10.0),
                'precip_mm': daily.get("precipitation_sum", [0.0] * len(dates))[i] or 0.0,
                'cloud': daily.get("cloud_cover_mean", [50.0] * len(dates))[i] or 50.0,
                'uv_index': 5.0
            })
        
        return history if len(history) >= days else None
    except Exception as e:
        print(f"⚠️ Open-Meteo API error: {e}")
        return None


# ============================================
# Load Model & Artifacts
# ============================================
def load_artifacts():
    # Advanced Transformer Model
    checkpoint = torch.load(V2_MODELS_DIR / "advanced_transformer.pt", map_location="cpu", weights_only=False)
    model = AdvancedWeatherTransformer(
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
    scaler = joblib.load(V2_MODELS_DIR / "advanced_scaler.joblib")
    
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
            historical[key] = {
                'temperature_celsius': round(row["temperature_celsius"], 1),
                'humidity': row.get("humidity", 50.0),
                'pressure_mb': row.get("pressure_mb", 1013.0),
                'wind_kph': row.get("wind_kph", 10.0),
                'precip_mm': row.get("precip_mm", 0.0),
                'cloud': row.get("cloud", 50.0),
                'uv_index': row.get("uv_index", 5.0)
            }
    
    return model, scaler, stats, stats_df, feature_cols, seq_len, pred_len, historical


MODEL, SCALER, COUNTRY_STATS, STATS_DF, FEATURE_COLS, SEQ_LEN, PRED_LEN, HISTORICAL_DATA = load_artifacts()
COUNTRIES = sorted(COUNTRY_STATS.keys())

DEFAULT_WEATHER = {'humidity': 50.0, 'pressure_mb': 1013.0, 'wind_kph': 10.0, 'precip_mm': 0.0, 'cloud': 50.0, 'uv_index': 5.0}


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
# Advanced Transformer Prediction
# ============================================
def predict_forecast(lat: float, lon: float, start_date: str) -> tuple:
    nearest_country, _ = find_nearest_country(lat, lon)
    stats = COUNTRY_STATS.get(nearest_country, {})
    climate_zone, climate_zone_encoded = get_climate_zone(lat)
    hemisphere_encoded = 1 if lat >= 0 else 0
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    data_source = "unknown"
    
    # Strategy 1: Try internal dataset first
    weather_history = []
    for i in range(SEQ_LEN, 0, -1):
        hist_date = (start_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        key = (nearest_country, hist_date)
        if key in HISTORICAL_DATA:
            weather_history.append(HISTORICAL_DATA[key])
    
    if len(weather_history) == SEQ_LEN:
        data_source = "internal_dataset"
    else:
        # Strategy 2: Fetch from Open-Meteo API
        api_history = fetch_open_meteo_history(lat, lon, start_date, SEQ_LEN)
        if api_history and len(api_history) >= SEQ_LEN:
            weather_history = api_history[-SEQ_LEN:]
            data_source = "open_meteo_api"
        else:
            # Strategy 3: Fallback
            month_key = f"temp_mean_month_{start_dt.month}"
            month_avg = stats.get(month_key, stats.get("temp_mean", 20.0))
            weather_history = [{'temperature_celsius': float(month_avg), **DEFAULT_WEATHER}] * SEQ_LEN
            data_source = "fallback_defaults"
    
    # Build feature sequence
    sequence = []
    for i in range(SEQ_LEN):
        day = start_dt - timedelta(days=SEQ_LEN - i)
        weather = weather_history[i]
        
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
            'temperature_celsius': weather['temperature_celsius'],
            'humidity': weather['humidity'],
            'pressure_mb': weather['pressure_mb'],
            'wind_kph': weather['wind_kph'],
            'precip_mm': weather['precip_mm'],
            'cloud': weather['cloud'],
            'uv_index': weather['uv_index']
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
        hist_data = HISTORICAL_DATA.get((nearest_country, date_str))
        actual = hist_data['temperature_celsius'] if hist_data else None
        
        forecast_item = {"date": date_str, "day": date.strftime("%A"), "temp": round(float(preds[i]), 1)}
        if actual is not None:
            forecast_item["actual"] = actual
        forecasts.append(forecast_item)
    
    return forecasts, climate_zone, nearest_country, data_source


# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Weather Trend Forecasting V2 (STABLE)", version="2.0.0-stable", description="✅ STABLE - Production-ready weather forecasting API. Recommended for production use.")
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
        forecast, climate_zone, nearest_country, data_source = predict_forecast(req.lat, req.lon, req.start_date)
        temps = [f["temp"] for f in forecast]
        trend = "↗ Warming" if temps[-1] > temps[0] else "↘ Cooling" if temps[-1] < temps[0] else "→ Stable"
        
        actuals = [f.get("actual") for f in forecast if f.get("actual") is not None]
        preds_with_actual = [f["temp"] for f in forecast if f.get("actual") is not None]
        
        summary = {
            "min": min(temps), "max": max(temps), "avg": round(np.mean(temps), 1), "trend": trend,
            "has_actual": len(actuals) > 0, "actual_days": len(actuals),
            "data_source": data_source, "model": "advanced-transformer-v4"
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
    return {"status": "healthy", "version": "4.0-advanced", "model": "advanced-transformer-grn",
            "countries": len(COUNTRIES), "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
            "features": len(FEATURE_COLS), "historical_records": len(HISTORICAL_DATA),
            "open_meteo_enabled": True, "mae": "2.00°C"}
