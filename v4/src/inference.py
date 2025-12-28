"""
V4 Inference Module.
High-level API for V4 Ensemble Weather Forecaster.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from v4.src.config import MODELS_DIR, SCALER_PATH, TRANSFORMER_MODEL_PATH, XGBOOST_MODEL_PATH, V4Config
from v4.src.models import EnsembleForecaster, TransformerForecaster, XGBoostForecaster
from v4.src.preprocessing import V4Preprocessor
from v4.src.utils import format_forecast_response, get_climate_zone, get_hemisphere


class V4Forecaster:
    """
    High-level V4 Ensemble Weather Forecaster API.

    Combines XGBoost and Transformer models with climate-zone-specific weighting.
    """

    def __init__(self, device: str = None):
        """
        Initialize V4 Forecaster.

        Args:
            device: PyTorch device (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = None
        self.ensemble = None
        self.scaler = None
        self._loaded = False

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model artifacts from disk."""
        # Load scaler
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
            self.preprocessor = V4Preprocessor(scaler=self.scaler)
        else:
            self.preprocessor = V4Preprocessor()

        # Load XGBoost model
        xgboost = None
        if XGBOOST_MODEL_PATH.exists():
            xgboost_models = joblib.load(XGBOOST_MODEL_PATH)
            xgboost = XGBoostForecaster(pred_len=V4Config.PRED_LEN)
            xgboost.models = xgboost_models

        # Load Transformer model
        transformer = None
        if TRANSFORMER_MODEL_PATH.exists():
            checkpoint = torch.load(TRANSFORMER_MODEL_PATH, map_location=self.device)

            # Get model config from checkpoint or use defaults
            input_dim = checkpoint.get("input_dim", len(V4Config.get_all_features()))
            d_model = checkpoint.get("d_model", V4Config.TRANSFORMER_PARAMS["d_model"])
            nhead = checkpoint.get("nhead", V4Config.TRANSFORMER_PARAMS["nhead"])
            num_layers = checkpoint.get("num_layers", V4Config.TRANSFORMER_PARAMS["num_layers"])
            dropout = checkpoint.get("dropout", V4Config.TRANSFORMER_PARAMS["dropout"])

            transformer = TransformerForecaster(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                seq_len=V4Config.SEQ_LEN,
                pred_len=V4Config.PRED_LEN,
            )

            if "model_state_dict" in checkpoint:
                transformer.load_state_dict(checkpoint["model_state_dict"])
            else:
                transformer.load_state_dict(checkpoint)

        # Create ensemble
        self.ensemble = EnsembleForecaster(
            xgboost_model=xgboost,
            transformer_model=transformer,
            device=self.device,
        )

        self._loaded = xgboost is not None or transformer is not None

    def predict(
        self,
        latitude: float,
        longitude: float,
        start_date: str | datetime,
        weather_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate 7-day temperature forecast.

        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)
            start_date: Forecast start date (YYYY-MM-DD or datetime)
            weather_history: Optional historical weather data (14 days)

        Returns:
            Forecast response dictionary
        """
        # Parse date
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Get climate zone for ensemble weighting
        climate_zone = get_climate_zone(latitude)

        # Build input sequence
        if weather_history:
            sequence = self.preprocessor.build_sequence(
                weather_history, latitude, longitude, start_date - timedelta(days=V4Config.SEQ_LEN)
            )
        else:
            # Use default values for demo/testing
            sequence = self._create_default_sequence(latitude, longitude, start_date)

        # Make prediction
        if self._loaded:
            predictions, lower, upper = self.ensemble.predict_with_uncertainty(sequence, climate_zone)
        else:
            # Fallback: simple climatology-based prediction
            predictions = self._climatology_prediction(latitude, start_date)
            lower = predictions - 3
            upper = predictions + 3

        # Format response
        return format_forecast_response(
            predictions=predictions,
            start_date=start_date,
            latitude=latitude,
            longitude=longitude,
            confidence_intervals=(lower, upper),
        )

    def predict_with_comparison(
        self,
        latitude: float,
        longitude: float,
        start_date: str | datetime,
        actual_temps: list[float] | None = None,
        weather_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate forecast with actual vs predicted comparison.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Forecast start date
            actual_temps: Optional list of actual temperatures for comparison
            weather_history: Optional historical weather data

        Returns:
            Forecast response with comparison metrics
        """
        # Parse date
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        climate_zone = get_climate_zone(latitude)

        # Build sequence
        if weather_history:
            sequence = self.preprocessor.build_sequence(
                weather_history, latitude, longitude, start_date - timedelta(days=V4Config.SEQ_LEN)
            )
        else:
            sequence = self._create_default_sequence(latitude, longitude, start_date)

        # Make prediction
        if self._loaded:
            predictions, lower, upper = self.ensemble.predict_with_uncertainty(sequence, climate_zone)
        else:
            predictions = self._climatology_prediction(latitude, start_date)
            lower = predictions - 3
            upper = predictions + 3

        # Convert actual_temps to numpy array
        actual_array = None
        if actual_temps:
            actual_array = np.array(actual_temps[:V4Config.PRED_LEN])

        # Format response
        return format_forecast_response(
            predictions=predictions,
            start_date=start_date,
            latitude=latitude,
            longitude=longitude,
            confidence_intervals=(lower, upper),
            actual_temps=actual_array,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get model metadata and configuration."""
        return {
            "version": "4.0",
            "name": "V4 Ensemble Weather Forecaster",
            "architecture": "XGBoost + Transformer with GRN",
            "seq_len": V4Config.SEQ_LEN,
            "pred_len": V4Config.PRED_LEN,
            "n_features": len(V4Config.get_all_features()),
            "device": self.device,
            "loaded": self._loaded,
            "has_xgboost": self.ensemble.xgboost is not None if self.ensemble else False,
            "has_transformer": self.ensemble.transformer is not None if self.ensemble else False,
            "climate_zones": list(V4Config.ENSEMBLE_WEIGHTS.keys()),
            "top_features": V4Config.TOP_FEATURES[:5],
        }

    def _create_default_sequence(
        self, latitude: float, longitude: float, start_date: datetime
    ) -> np.ndarray:
        """Create default sequence for demo/testing."""
        from v4.src.config import DEFAULT_WEATHER_VALUES

        weather_history = [DEFAULT_WEATHER_VALUES.copy() for _ in range(V4Config.SEQ_LEN)]
        return self.preprocessor.build_sequence(
            weather_history, latitude, longitude, start_date - timedelta(days=V4Config.SEQ_LEN)
        )

    def _climatology_prediction(self, latitude: float, start_date: datetime) -> np.ndarray:
        """
        Simple climatology-based prediction as fallback.
        Uses latitude and month to estimate temperature.
        """
        import math

        # Base temperature by latitude
        abs_lat = abs(latitude)
        base_temp = 30 - (abs_lat / 90) * 50  # ~30°C at equator, ~-20°C at poles

        # Seasonal adjustment
        month = start_date.month
        hemisphere = 1 if latitude >= 0 else -1
        seasonal_offset = hemisphere * 10 * math.cos(2 * math.pi * (month - 1) / 12)

        base = base_temp + seasonal_offset

        # Generate 7-day forecast with small daily variation
        predictions = np.array([
            base + (i - 3) * 0.5 + np.random.randn() * 0.5
            for i in range(V4Config.PRED_LEN)
        ])

        return predictions


# Convenience function for quick predictions
def quick_forecast(
    latitude: float, longitude: float, start_date: str = None
) -> dict[str, Any]:
    """
    Quick forecast helper function.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (defaults to today)

    Returns:
        Forecast dictionary
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    forecaster = V4Forecaster()
    return forecaster.predict(latitude, longitude, start_date)
