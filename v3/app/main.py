"""
V3 Weather Forecasting API
FastAPI backend for Climate-Aware Transformer predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from typing import Any

import httpx  # For async HTTP requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# V3 imports
from v3.src import V3Config, V3Forecaster

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="V3 Climate-Aware Forecaster (EXPERIMENTAL)",
    description="⚠️ EXPERIMENTAL - 7-day temperature forecasting with Hybrid Architecture. Use V2 for production.",
    version="3.1.0-experimental",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecaster (lazy loading)
forecaster: V3Forecaster | None = None


def get_forecaster() -> V3Forecaster:
    global forecaster
    if forecaster is None:
        forecaster = V3Forecaster()
    return forecaster


# =============================================================================
# Pydantic Models
# =============================================================================


class WeatherInput(BaseModel):
    """Single day weather observation."""

    humidity: float | None = Field(None, ge=0, le=100)
    pressure_mb: float | None = Field(None, ge=900, le=1100)
    wind_kph: float | None = Field(None, ge=0)
    cloud: float | None = Field(None, ge=0, le=100)
    precip_mm: float | None = Field(None, ge=0)
    uv_index: float | None = Field(None, ge=0, le=15)
    visibility_km: float | None = Field(None, ge=0)
    air_quality_Ozone: float | None = None
    air_quality_PM25: float | None = Field(None, alias="air_quality_PM2.5")


class ForecastRequest(BaseModel):
    """Forecast request with location and optional what-if parameters."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str = "Egypt"  # Default for backward compatibility
    start_date: str | None = None  # YYYY-MM-DD, defaults to today
    weather_scenario: WeatherInput | None = None  # What-if scenario


class ForecastDay(BaseModel):
    """Single day forecast."""

    date: str
    day: str
    temperature: float


class ForecastResponse(BaseModel):
    """Complete forecast response."""

    forecast: list[ForecastDay]
    location: dict[str, Any]
    model_info: dict[str, Any]


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return HTMLResponse(
        "<h1>V3.1 Weather Forecaster API</h1><p>UI not found. Use /docs for API.</p>"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": forecaster is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "3.1.0 (Hybrid)",
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get model configuration and metadata."""
    try:
        fc = get_forecaster()
        return fc.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Generate 7-day temperature forecast using V3.1 Hybrid Model.

    Supports "what-if" scenarios by providing custom weather parameters.
    """
    try:
        fc = get_forecaster()

        # Parse date
        if request.start_date:
            start_date = request.start_date
        else:
            start_date = datetime.now().strftime("%Y-%m-%d")

        # Build weather history from scenario (or use defaults)
        weather_history = None
        if request.weather_scenario:
            scenario = request.weather_scenario.model_dump(exclude_none=True)
            weather_history = [scenario.copy() for _ in range(V3Config.SEQ_LEN)]

        # Get prediction
        result = fc.predict(
            latitude=request.latitude,
            longitude=request.longitude,
            country=request.country,
            start_date=start_date,
            weather_history=weather_history,
        )

        return {
            "forecast": result["forecast"],
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "climate_zone": result["climate_zone"],
                "hemisphere": result["hemisphere"],
            },
            "model_info": {"version": "V3 Climate-Aware Transformer", "device": fc.device},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/forecast")
async def get_forecast(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
):
    """
    Simple GET endpoint for forecast.

    Example: /api/forecast?lat=30.0&lon=31.2&date=2024-01-15
    """
    request = ForecastRequest(latitude=lat, longitude=lon, start_date=date)
    return await create_forecast(request)


@app.get("/api/climate-zones")
async def get_climate_zones():
    """Return climate zone definitions."""
    return {
        "zones": [
            {"name": "Tropical", "latitude_range": "0° - 23.5°"},
            {"name": "Subtropical", "latitude_range": "23.5° - 35°"},
            {"name": "Temperate", "latitude_range": "35° - 55°"},
            {"name": "Subarctic", "latitude_range": "55° - 66.5°"},
            {"name": "Polar", "latitude_range": "66.5° - 90°"},
        ]
    }


# =============================================================================
# Historical Comparison (Predicted vs Actual)
# =============================================================================


@app.get("/api/historical")
async def get_historical_comparison(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    country: str = Query("Egypt", description="Country name"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
):
    """
    Compare model predictions against actual historical temperatures.

    Fetches actual temps from Open-Meteo Archive API, then runs the model
    to produce predictions for the same date range.
    """
    try:
        fc = get_forecaster()

        # 1. Fetch Actual Historical Temperatures from Open-Meteo
        open_meteo_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily=temperature_2m_mean&timezone=auto"
        )

        async with httpx.AsyncClient() as client:
            resp = await client.get(open_meteo_url)
            resp.raise_for_status()
            data = resp.json()

        dates = data.get("daily", {}).get("time", [])
        actuals = data.get("daily", {}).get("temperature_2m_mean", [])

        if not dates or not actuals:
            raise HTTPException(
                status_code=404, detail="No historical data found for this location/range."
            )

        # 2. Run Model Predictions for each date in the range
        # Note: The model predicts 7 days ahead. For comparison, we'll use the Day 1 prediction.
        predictions = []
        for date_str in dates:
            result = fc.predict(latitude=lat, longitude=lon, country=country, start_date=date_str)
            # Take the first forecast day (Day 0 = start_date itself)
            predictions.append(result["forecast"][0]["temperature"])

        return {
            "dates": dates,
            "actual": actuals,
            "predicted": predictions,
            "location": {"latitude": lat, "longitude": lon, "country": country},
        }

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"Open-Meteo API error: {e.response.text}"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Static Files
# =============================================================================

# Mount static files if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
