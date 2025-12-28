"""
V3 Inference Module (Hybrid Architecture)
High-level API for weather forecasting using V3.1 Hybrid Static-Dynamic Transformer.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

# Use absolute imports for better reliability in this project structure
from v3.src.config import DEFAULT_WEATHER_VALUES, V3Config
from v3.src.model import HybridClimateTransformer
from v3.src.preprocessing import V3Preprocessor

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_CHECKPOINT = BASE_DIR / "v3" / "models" / "v3_hybrid_best.pt"
ARTIFACTS_PATH = BASE_DIR / "v3" / "models" / "v3_1_production_artifacts.joblib"


class V3Forecaster:
    """
    High-level interface for V3.1 Hybrid weather forecasting.
    Separates processing for Dynamic (Weather) and Static (Geo) features.

    Usage:
        forecaster = V3Forecaster()
        predictions = forecaster.predict(
            latitude=30.0,
            longitude=31.2,
            country="Egypt",
            start_date="2024-01-15"
        )
    """

    def __init__(self, model_path: Path | None = None, device: str = "auto"):
        self.model_path = model_path or MODEL_CHECKPOINT

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load Artifacts (Scalers, Encoders)
        if not ARTIFACTS_PATH.exists():
            raise FileNotFoundError(f"Artifacts not found at {ARTIFACTS_PATH}")

        self.artifacts = joblib.load(ARTIFACTS_PATH)
        self.scaler_dyn = self.artifacts["scaler_dyn"]
        self.scaler_stat = self.artifacts["scaler_stat"]
        self.country_encoder = self.artifacts["country_encoder"]
        self.dyn_features = self.artifacts["dyn_features"]
        self.stat_features = self.artifacts["stat_features"]

        # Load Model
        self.model = self._load_model()
        self.preprocessor = V3Preprocessor()

    def _load_model(self) -> HybridClimateTransformer:
        """Load hybrid model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # Initialize model with saved dimensions
        model = HybridClimateTransformer(
            num_countries=self.artifacts["num_countries"],
            dyn_input_dim=len(self.dyn_features),
            stat_input_dim=len(self.stat_features),
            d_model=128,  # Architecture constant
            num_layers=4,  # Architecture constant
        )

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        latitude: float,
        longitude: float,
        country: str = "Egypt",
        start_date: str | datetime = None,
        weather_history: list[dict[str, float]] | None = None,
        return_dates: bool = True,
    ) -> dict[str, Any]:
        """
        Generate 7-day weather forecast.
        """
        # 1. Parse Input
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        elif start_date is None:
            start_dt = datetime.now()
        else:
            start_dt = start_date

        # Default history
        if weather_history is None:
            weather_history = [DEFAULT_WEATHER_VALUES.copy() for _ in range(V3Config.SEQ_LEN)]

        # 2. Preprocess Dynamic Features (Sequence)
        # Note: We reuse V3Preprocessor to get raw values, then filter/scale here
        raw_sequence = self.preprocessor.build_raw_sequence(
            latitude=latitude,
            longitude=longitude,
            start_date=start_dt,
            weather_history=weather_history,
        )

        # Extract only the dynamic columns we need
        # raw_sequence is list of dicts. We convert to DataFrame-like array
        import pandas as pd

        seq_df = pd.DataFrame(raw_sequence)

        # Ensure Month sin/cos
        seq_df = self._add_time_features(seq_df, start_dt)

        # Select Dynamic Features
        # Fill missing with 0 just in case
        for col in self.dyn_features:
            if col not in seq_df.columns:
                seq_df[col] = 0.0

        X_dyn = seq_df[self.dyn_features].values

        # Scale Dynamic
        X_dyn = self.scaler_dyn.transform(X_dyn)

        # 3. Preprocess Static Features
        # ["latitude", "longitude", "abs_latitude", "hemisphere_encoded"]
        abs_lat = abs(latitude)
        hemi = 1 if latitude >= 0 else 0
        X_stat = np.array([[latitude, longitude, abs_lat, hemi]])  # Shape (1, 4)

        # Scale Static
        X_stat = self.scaler_stat.transform(X_stat)

        # 4. Process Country
        try:
            country_id = self.country_encoder.transform([country])[0]
        except ValueError:
            # Fallback to mostly populated country if unknown
            country_id = self.country_encoder.transform(["Egypt"])[0]

        X_country = np.array([country_id])

        # 5. Convert to Tensors
        t_dyn = torch.FloatTensor(X_dyn).unsqueeze(0).to(self.device)  # [1, Seq, Dyn_Dim]
        t_stat = torch.FloatTensor(X_stat).to(self.device)  # [1, Stat_Dim]
        t_country = torch.LongTensor(X_country).to(self.device)  # [1]

        # 6. Predict
        with torch.no_grad():
            preds = self.model(t_dyn, t_stat, t_country).cpu().numpy()[0]

        # 7. Format Output
        result = {
            "predictions": preds.tolist(),
            "latitude": latitude,
            "longitude": longitude,
            "country": country,
            "climate_zone": V3Config.get_climate_zone(latitude),
            "hemisphere": "Northern" if latitude >= 0 else "Southern",
        }

        if return_dates:
            result["forecast"] = [
                {
                    "date": (start_dt + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "day": (start_dt + timedelta(days=i)).strftime("%A"),
                    "temperature": round(float(preds[i]), 1),
                }
                for i in range(len(preds))
            ]

        return result

    def _add_time_features(self, df, start_dt):
        """Add cyclic time features to the sequence DataFrame."""
        dates = [start_dt - timedelta(days=i) for i in range(len(df) - 1, -1, -1)]
        months = np.array([d.month for d in dates])
        df["month_sin"] = np.sin(2 * np.pi * months / 12)
        df["month_cos"] = np.cos(2 * np.pi * months / 12)
        return df

    def get_model_info(self) -> dict[str, Any]:
        return {
            "version": "3.1.0 Hybrid",
            "device": self.device,
            "features_dynamic": self.dyn_features,
            "features_static": self.stat_features,
            "num_countries": len(self.country_encoder.classes_),
        }
