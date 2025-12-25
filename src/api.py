from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import sys

# Add src to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import XGBoostForecaster
from data_loader import DataLoader
from features import TimeSeriesFeatures

app = FastAPI(
    title="Weather Trend Forecasting API",
    description="API for serving 24-hour temperature forecasts using Recursive XGBoost.",
    version="1.0.0"
)

# Global objects
model = None
loader = None
fe = None

class PredictionRequest(BaseModel):
    timestamp: str  # ISO 8601 format e.g. "2024-05-16T12:00:00"
    temperature_celsius: float
    # We might need recent lags for accurate prediction
    # For MVP, we'll assume the client sends the current snapshot 
    # and the model handles the lack of deep history via its internal logic (or simple shifting)
    # Ideally, the API should accept a history window.
    # To keep it simple per requirements: "Accepting current_temperature and timestamp"
    
class ForecastResponse(BaseModel):
    forecast: dict # {timestamp: temperature}

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), '../models/forecast_model.pkl')
    try:
        model = XGBoostForecaster.load(model_path)
        print(f"Model loaded from {model_path}")
        
        # Initialize helpers
        global loader, fe
        loader = DataLoader(raw_data_path="dummy_path") # Path not needed for live fetch
        fe = TimeSeriesFeatures()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, we might want to crash if model fails to load
        
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=ForecastResponse)
def predict_forecast(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        # 1. Fetch Live History (Context)
        # CRITICAL FIX: We need enough history to calculate the max lag (168h).
        # If we only fetch 168h, the first valid lag_168 is at index 168, which doesn't exist.
        # We fetch 2 weeks (336h) to ensure we have a sizable valid buffer after dropna().
        print("Fetching live context...")
        history_df = loader.fetch_realtime_history(hours_back=336)
        
        if history_df.empty:
            # Fallback: Create minimal DataFrame from request
            print("Warning: Live fetch failed. Using cold-start.")
            history_df = pd.DataFrame([{
                'temperature_celsius': request.temperature_celsius
            }], index=[pd.to_datetime(request.timestamp)])
            
        # 2. Feature Engineering on Context
        # We need to run transform on the history to get lags/rolling
        # Then pick the LAST row as our 'current_input'
        df_features = fe.transform(history_df)
        
        if df_features.empty:
             raise HTTPException(status_code=500, detail="Feature engineering returned empty set (not enough history?)")
             
        # Select features
        features = model.features # Get feature names from model
        current_input = df_features[features].iloc[-1:].copy()
        
        # 3. Predict
        forecast_df = model.predict_next_24h(current_input)
        
        # Format Response
        # Convert timestamp index to string keys
        result = {str(ts): val for ts, val in zip(forecast_df.index, forecast_df['predicted_temperature'])}
        
        return {"forecast": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
