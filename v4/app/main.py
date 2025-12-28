"""
V4 Weather Forecasting API
FastAPI backend for V4 Ensemble Weather Forecaster.
EXPERIMENTAL - Full-featured web application.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v4.src.config import V4Config
from v4.src.inference import V4Forecaster
from v4.src.utils import get_climate_zone, get_hemisphere, validate_coordinates

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="V4 Ensemble Weather Forecaster (EXPERIMENTAL)",
    description="""
    🌍 **V4 Experimental** - Advanced Ensemble Weather Forecasting
    
    Combines XGBoost (from assessment analysis) with Transformer+GRN for superior accuracy.
    
    **Features:**
    - 📊 7-day temperature forecast
    - 🎯 Confidence intervals
    - 📈 Actual vs Predicted comparison
    - 🌍 Climate zone detection
    - 📉 Feature importance API
    """,
    version="4.0.0-experimental",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize forecaster
forecaster: V4Forecaster | None = None


@app.on_event("startup")
async def startup():
    """Initialize forecaster on startup."""
    global forecaster
    try:
        forecaster = V4Forecaster()
    except Exception as e:
        print(f"Warning: Could not load forecaster: {e}")
        forecaster = None


# =============================================================================
# Pydantic Models
# =============================================================================


class ForecastRequest(BaseModel):
    """Forecast request model."""

    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    weather_scenario: dict[str, float] | None = Field(
        None, description="Optional what-if weather scenario"
    )


class HistoricalRequest(BaseModel):
    """Historical comparison request."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve interactive map UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        """
        <html>
        <head><title>V4 Forecaster</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>🌍 V4 Ensemble Weather Forecaster</h1>
            <p>EXPERIMENTAL</p>
            <p>API docs: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "4.0.0-experimental",
        "model_loaded": forecaster is not None and forecaster._loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get model architecture and configuration info."""
    if forecaster is None:
        return {
            "version": "4.0",
            "name": "V4 Ensemble (Not Loaded)",
            "loaded": False,
        }
    return forecaster.get_model_info()


@app.get("/api/countries")
async def get_countries():
    """Get list of supported countries with metadata."""
    # Load location stats from V2 if available
    location_file = Path("v2/models/location_stats.csv")
    if location_file.exists():
        import pandas as pd

        df = pd.read_csv(location_file)
        countries = []
        for _, row in df.iterrows():
            countries.append({
                "name": row.get("country", row.get("location_name", "Unknown")),
                "latitude": row.get("latitude", row.get("lat", 0)),
                "longitude": row.get("longitude", row.get("lon", 0)),
                "climate_zone": get_climate_zone(row.get("latitude", row.get("lat", 0))),
            })
        return {"countries": countries, "count": len(countries)}

    # Fallback: sample countries
    sample_countries = [
        {"name": "Egypt", "latitude": 30.04, "longitude": 31.24, "climate_zone": "Subtropical"},
        {"name": "United States", "latitude": 38.89, "longitude": -77.04, "climate_zone": "Temperate"},
        {"name": "Brazil", "latitude": -15.79, "longitude": -47.88, "climate_zone": "Tropical"},
        {"name": "Australia", "latitude": -33.87, "longitude": 151.21, "climate_zone": "Subtropical"},
        {"name": "Germany", "latitude": 52.52, "longitude": 13.40, "climate_zone": "Temperate"},
        {"name": "Japan", "latitude": 35.68, "longitude": 139.65, "climate_zone": "Temperate"},
        {"name": "South Africa", "latitude": -33.93, "longitude": 18.42, "climate_zone": "Subtropical"},
        {"name": "Canada", "latitude": 45.42, "longitude": -75.70, "climate_zone": "Temperate"},
        {"name": "Norway", "latitude": 59.91, "longitude": 10.75, "climate_zone": "Subarctic"},
        {"name": "Singapore", "latitude": 1.35, "longitude": 103.82, "climate_zone": "Tropical"},
    ]
    return {"countries": sample_countries, "count": len(sample_countries)}


@app.get("/api/nearest")
async def get_nearest_country(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Find nearest country to coordinates."""
    from v4.src.utils import haversine_distance

    countries_response = await get_countries()
    countries = countries_response["countries"]

    if not countries:
        raise HTTPException(status_code=500, detail="No countries available")

    # Find nearest
    min_dist = float("inf")
    nearest = countries[0]
    for country in countries:
        dist = haversine_distance(lat, lon, country["latitude"], country["longitude"])
        if dist < min_dist:
            min_dist = dist
            nearest = country

    return {
        "nearest": nearest,
        "distance_km": round(min_dist, 1),
        "query": {"latitude": lat, "longitude": lon},
    }


@app.post("/api/forecast")
async def forecast_post(request: ForecastRequest):
    """Generate 7-day ensemble forecast (POST)."""
    try:
        validate_coordinates(request.lat, request.lon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not initialized")

    try:
        result = forecaster.predict(
            latitude=request.lat,
            longitude=request.lon,
            start_date=request.start_date,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast")
async def forecast_get(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD, defaults to today)"),
):
    """Generate 7-day ensemble forecast (GET)."""
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not initialized")

    try:
        result = forecaster.predict(
            latitude=lat,
            longitude=lon,
            start_date=start_date,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical")
async def get_historical_comparison(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD, defaults to start_date + 7 days)"),
):
    """
    Compare model predictions against actual historical temperatures.

    Fetches actual data from Open-Meteo Archive API for comparison.
    """
    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = start + timedelta(days=6)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Ensure dates are in the past
    today = datetime.now()
    if end >= today:
        end = today - timedelta(days=1)
    if start >= today:
        raise HTTPException(status_code=400, detail="Start date must be in the past")

    # Fetch actual data from Open-Meteo
    actual_temps = await fetch_open_meteo_historical(lat, lon, start, end)

    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not initialized")

    # Generate forecast for comparison
    result = forecaster.predict_with_comparison(
        latitude=lat,
        longitude=lon,
        start_date=start,
        actual_temps=actual_temps,
    )

    return result


async def fetch_open_meteo_historical(
    lat: float, lon: float, start: datetime, end: datetime
) -> list[float | None]:
    """Fetch historical temperatures from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean",
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "daily" in data and "temperature_2m_mean" in data["daily"]:
                return data["daily"]["temperature_2m_mean"]
            return []
    except Exception as e:
        print(f"Open-Meteo fetch error: {e}")
        return []


@app.get("/api/climate-zones")
async def get_climate_zones():
    """Get climate zone definitions."""
    return {
        "zones": [
            {"name": "Tropical", "latitude_range": "0° - 23.5°", "description": "Hot and humid year-round"},
            {"name": "Subtropical", "latitude_range": "23.5° - 35°", "description": "Hot summers, mild winters"},
            {"name": "Temperate", "latitude_range": "35° - 55°", "description": "Four distinct seasons"},
            {"name": "Subarctic", "latitude_range": "55° - 66.5°", "description": "Cold winters, cool summers"},
            {"name": "Polar", "latitude_range": "66.5° - 90°", "description": "Cold year-round"},
        ],
        "ensemble_weights": V4Config.ENSEMBLE_WEIGHTS,
    }


@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance rankings from assessment analysis."""
    # Top features from assessment SHAP analysis
    importance = {
        "method": "Average Rank (Correlation, RF, XGBoost, Permutation, SHAP)",
        "source": "assessment/outputs/feature_importance_comparison.csv",
        "rankings": [
            {"rank": 1, "feature": "temperature_fahrenheit", "avg_rank": 1.2},
            {"rank": 2, "feature": "feels_like_celsius", "avg_rank": 1.8},
            {"rank": 3, "feature": "feels_like_fahrenheit", "avg_rank": 4.0},
            {"rank": 4, "feature": "humidity", "avg_rank": 6.8},
            {"rank": 5, "feature": "latitude", "avg_rank": 9.4},
            {"rank": 6, "feature": "uv_index", "avg_rank": 9.6},
            {"rank": 7, "feature": "air_quality_Carbon_Monoxide", "avg_rank": 10.4},
            {"rank": 8, "feature": "pressure_mb", "avg_rank": 12.2},
            {"rank": 9, "feature": "longitude", "avg_rank": 12.2},
            {"rank": 10, "feature": "wind_mph", "avg_rank": 13.0},
        ],
    }
    return importance


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
