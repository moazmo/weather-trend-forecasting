"""
V3 Inference Module
High-level API for weather forecasting.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import DEFAULT_WEATHER_VALUES, MODEL_CHECKPOINT, V3Config
from .model import V3ClimateTransformer
from .preprocessing import V3Preprocessor


class V3Forecaster:
    """
    High-level interface for V3 weather forecasting.

    Usage:
        forecaster = V3Forecaster()
        predictions = forecaster.predict(
            latitude=51.5,
            longitude=-0.1,
            start_date="2024-01-15",
            weather_history=[...]
        )
    """

    def __init__(self, model_path: Path | None = None, device: str = "auto"):
        """
        Initialize forecaster.

        Args:
            model_path: Path to model checkpoint. Uses default if None.
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
        """
        self.model_path = model_path or MODEL_CHECKPOINT

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and preprocessor
        self.model = self._load_model()
        self.preprocessor = V3Preprocessor()

    def _load_model(self) -> V3ClimateTransformer:
        """Load model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        model = V3ClimateTransformer.from_checkpoint(str(self.model_path), device=self.device)
        return model.to(self.device)

    def predict(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        weather_history: list[dict[str, float]] | None = None,
        return_dates: bool = True,
    ) -> dict[str, Any]:
        """
        Generate weather forecast.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Forecast start date (YYYY-MM-DD)
            weather_history: List of historical weather dicts. If None, uses defaults.
            return_dates: Whether to include date strings in output

        Returns:
            Dictionary with predictions and metadata
        """
        # Parse date
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = start_date

        # Default weather history if not provided
        if weather_history is None:
            weather_history = [DEFAULT_WEATHER_VALUES.copy() for _ in range(V3Config.SEQ_LEN)]

        # Build input sequence
        sequence = self.preprocessor.build_sequence(
            latitude=latitude,
            longitude=longitude,
            start_date=start_dt,
            weather_history=weather_history,
        )

        # Add batch dimension
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()[0]

        # Build response
        result = {
            "predictions": predictions.tolist(),
            "latitude": latitude,
            "longitude": longitude,
            "climate_zone": V3Config.get_climate_zone(latitude),
            "hemisphere": "Northern" if latitude >= 0 else "Southern",
        }

        if return_dates:
            result["forecast_dates"] = [
                (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(V3Config.PRED_LEN)
            ]
            result["forecast"] = [
                {
                    "date": (start_dt + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "day": (start_dt + timedelta(days=i)).strftime("%A"),
                    "temperature": round(float(predictions[i]), 1),
                }
                for i in range(V3Config.PRED_LEN)
            ]

        return result

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Batch prediction for multiple sequences.

        Args:
            sequences: Array of shape (batch, seq_len, n_features)

        Returns:
            Predictions of shape (batch, pred_len)
        """
        X = torch.FloatTensor(sequences).to(self.device)

        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()

        return predictions

    def get_model_info(self) -> dict[str, Any]:
        """Return model configuration and metadata."""
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        # Convert numpy types to native Python types for JSON serialization
        def to_python(val):
            if val is None:
                return None
            if hasattr(val, "item"):  # numpy scalar
                return val.item()
            return val

        return {
            "input_dim": to_python(checkpoint.get("input_dim")),
            "d_model": to_python(checkpoint.get("d_model")),
            "nhead": to_python(checkpoint.get("nhead")),
            "num_layers": to_python(checkpoint.get("num_layers")),
            "seq_len": to_python(checkpoint.get("seq_len", V3Config.SEQ_LEN)),
            "pred_len": to_python(checkpoint.get("pred_len", V3Config.PRED_LEN)),
            "test_mae": to_python(checkpoint.get("test_mae")),
            "device": self.device,
        }
