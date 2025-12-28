"""
V4 Weather Forecasting API (Optimized)
Historical Backtesting with trained XGBoost model.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v4.src.inference import V4Forecaster

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="V4 Weather Backtester",
    description="""
    🌍 **V4 Historical Backtesting** - XGBoost Weather Analysis
    
    Test the model on historical data from May 2024 - December 2025.
    
    **Model Performance:**
    - Training MAE: 0.96°C
    - Test MAE: 1.21°C
    
    **Features:**
    - 📊 Actual vs Predicted comparison
    - 🌍 211 countries
    - 📈 22 input features
    """,
    version="4.0.0",
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
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve interactive UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        """
        <html>
        <head><title>V4 Backtester</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h1>🌍 V4 Weather Backtester</h1>
            <p>Historical Backtesting with XGBoost</p>
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
        "version": "4.0.0",
        "model_loaded": forecaster is not None and forecaster._loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info")
async def get_model_info():
    """Get model information."""
    if forecaster is None:
        return {"error": "Forecaster not initialized"}
    return forecaster.get_model_info()


@app.get("/api/date-range")
async def get_date_range():
    """Get valid date range for backtesting."""
    if forecaster is None:
        return {"min": "2024-05-16", "max": "2025-12-24"}
    return forecaster.get_date_range()


@app.get("/api/countries")
async def get_countries():
    """Get list of available countries."""
    if forecaster is None:
        return {"countries": [], "count": 0}

    countries = forecaster.get_available_countries()
    return {"countries": countries, "count": len(countries)}


@app.get("/api/predict")
async def predict(
    country: str = Query(None, description="Country name"),
    lat: float = Query(None, ge=-90, le=90, description="Latitude (if no country)"),
    lon: float = Query(None, ge=-180, le=180, description="Longitude (if no country)"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD, defaults to start_date)"),
):
    """
    Predict temperatures and compare with actual values.
    
    Returns predicted vs actual for the specified date range.
    """
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not initialized")

    if country is None and (lat is None or lon is None):
        raise HTTPException(status_code=400, detail="Provide either 'country' or 'lat'+'lon'")

    # Validate dates
    date_range = forecaster.get_date_range()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        if date_range["min"]:
            min_date = datetime.strptime(date_range["min"], "%Y-%m-%d")
            if start < min_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Start date must be >= {date_range['min']}"
                )
        if date_range["max"]:
            max_date = datetime.strptime(date_range["max"], "%Y-%m-%d")
            if start > max_date:
                raise HTTPException(
                    status_code=400,
                    detail=f"Start date must be <= {date_range['max']}"
                )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    if end_date is None:
        end_date = start_date

    result = forecaster.predict(
        country=country,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance from trained model."""
    import json
    importance_path = Path(__file__).parent.parent / "models" / "feature_importance.json"
    if importance_path.exists():
        with open(importance_path) as f:
            importance = json.load(f)
        # Sort by importance
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return {
            "rankings": [{"rank": i + 1, "feature": f, "importance": round(v, 4)} 
                        for i, (f, v) in enumerate(sorted_imp[:15])],
            "model": "XGBoost",
            "total_features": len(importance),
        }
    return {"rankings": [], "error": "Feature importance not found"}


@app.get("/api/climate-zones")
async def get_climate_zones():
    """Get climate zone definitions."""
    return {
        "zones": [
            {"name": "Tropical", "latitude_range": "0° - 23.5°"},
            {"name": "Subtropical", "latitude_range": "23.5° - 35°"},
            {"name": "Temperate", "latitude_range": "35° - 55°"},
            {"name": "Subarctic", "latitude_range": "55° - 66.5°"},
            {"name": "Polar", "latitude_range": "66.5° - 90°"},
        ],
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
