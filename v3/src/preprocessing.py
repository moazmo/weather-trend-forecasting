"""
V3 Preprocessing Module
Feature engineering, scaling, and sequence creation.
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_WEATHER_VALUES, SCALER_PATH, V3Config


class V3Preprocessor:
    """
    Preprocessing pipeline for V3 model inputs.

    Handles:
        - Feature engineering (cyclical, derived)
        - Scaling (StandardScaler)
        - Sequence creation for time-series
        - Missing value imputation
    """

    def __init__(self, scaler_path: Path | None = None, require_scaler: bool = False):
        """
        Initialize preprocessor.

        Args:
            scaler_path: Path to fitted StandardScaler. If None, uses default.
            require_scaler: If True, raises error if scaler not found. If False, warns and continues.
        """
        self.scaler_path = scaler_path or SCALER_PATH
        self.scaler = None
        self._load_scaler(require_scaler)

    def _load_scaler(self, require: bool = False) -> None:
        """Load fitted scaler from disk."""
        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
        elif require:
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        # If not required and not found, scaler remains None (V3.1 uses separate scalers)

    @staticmethod
    def compute_cyclical_features(date: datetime) -> dict[str, float]:
        """
        Compute cyclical temporal features from date.

        Args:
            date: Datetime object

        Returns:
            Dictionary with sin/cos encodings
        """
        month = date.month
        day_of_year = date.timetuple().tm_yday
        hour = date.hour if hasattr(date, "hour") else 12

        return {
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "day_year_sin": np.sin(2 * np.pi * day_of_year / 365),
            "day_year_cos": np.cos(2 * np.pi * day_of_year / 365),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        }

    @staticmethod
    def compute_geographic_features(latitude: float, longitude: float) -> dict[str, float]:
        """
        Compute derived geographic features.

        Args:
            latitude: Latitude in degrees (-90 to 90)
            longitude: Longitude in degrees (-180 to 180)

        Returns:
            Dictionary with derived geographic features
        """
        return {
            "latitude": latitude,
            "longitude": longitude,
            "abs_latitude": abs(latitude),
            "latitude_normalized": abs(latitude) / 90.0,
            "hemisphere_encoded": 1 if latitude >= 0 else 0,
        }

    def build_feature_vector(
        self, date: datetime, latitude: float, longitude: float, weather_data: dict[str, float]
    ) -> np.ndarray:
        """
        Build complete feature vector for a single timestep.

        Args:
            date: Date for this observation
            latitude: Location latitude
            longitude: Location longitude
            weather_data: Dictionary of weather values (can have missing keys)

        Returns:
            1D numpy array of features
        """
        # Geographic features
        geo = self.compute_geographic_features(latitude, longitude)

        # Cyclical temporal features
        temporal = self.compute_cyclical_features(date)

        # Weather features (with defaults for missing)
        weather = {}
        for key in [
            "humidity",
            "pressure_mb",
            "wind_kph",
            "cloud",
            "precip_mm",
            "uv_index",
            "visibility_km",
            "gust_kph",
            "wind_degree",
            "air_quality_Ozone",
            "air_quality_Nitrogen_dioxide",
            "air_quality_PM2.5",
            "air_quality_Carbon_Monoxide",
            "air_quality_Sulphur_dioxide",
        ]:
            weather[key] = weather_data.get(key, DEFAULT_WEATHER_VALUES.get(key, 0.0))

        # Combine in correct order (must match training)
        feature_vector = [
            geo["latitude"],
            geo["longitude"],
            geo["abs_latitude"],
            geo["latitude_normalized"],
            geo["hemisphere_encoded"],
            weather["humidity"],
            weather["pressure_mb"],
            weather["wind_kph"],
            weather["cloud"],
            weather["precip_mm"],
            weather["uv_index"],
            weather["visibility_km"],
            weather["gust_kph"],
            weather["wind_degree"],
            weather["air_quality_Ozone"],
            weather["air_quality_Nitrogen_dioxide"],
            weather["air_quality_PM2.5"],
            weather["air_quality_Carbon_Monoxide"],
            weather["air_quality_Sulphur_dioxide"],
            temporal["month_sin"],
            temporal["month_cos"],
            temporal["day_year_sin"],
            temporal["day_year_cos"],
            temporal["hour_sin"],
            temporal["hour_cos"],
        ]

        return np.array(feature_vector, dtype=np.float32)

    def build_sequence(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        weather_history: list[dict[str, float]],
        seq_len: int = V3Config.SEQ_LEN,
    ) -> np.ndarray:
        """
        Build input sequence for model.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Date of forecast
            weather_history: List of weather dicts (oldest to newest)
            seq_len: Sequence length

        Returns:
            Scaled sequence of shape (seq_len, n_features)
        """
        # Ensure we have enough history
        if len(weather_history) < seq_len:
            # Pad with defaults
            padding = seq_len - len(weather_history)
            weather_history = [{}] * padding + weather_history

        # Build feature vectors for each timestep
        sequence = []
        for i in range(seq_len):
            day_offset = seq_len - i
            date = start_date - timedelta(days=day_offset)
            vector = self.build_feature_vector(date, latitude, longitude, weather_history[i])
            sequence.append(vector)

        sequence = np.array(sequence)

        # Scale
        if self.scaler is not None:
            # Flatten, scale, reshape
            original_shape = sequence.shape
            sequence = self.scaler.transform(sequence.reshape(-1, original_shape[-1]))
            sequence = sequence.reshape(original_shape)

        return sequence.astype(np.float32)

    def build_raw_sequence(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        weather_history: list[dict[str, float]],
        seq_len: int = V3Config.SEQ_LEN,
    ) -> list[dict[str, float]]:
        """
        Build raw weather sequence for V3.1 Hybrid Model.
        Returns unscalled list of dicts for flexible feature extraction.
        """
        # Ensure we have enough history
        if len(weather_history) < seq_len:
            padding = seq_len - len(weather_history)
            weather_history = [DEFAULT_WEATHER_VALUES.copy()] * padding + weather_history
            
        sequence = []
        for i in range(seq_len):
            # We assume history is chronological up to start_date
            # But actually `weather_history` usually represents the *past* leading up to today.
            # So index -1 is yesterday, -2 is day before.
            # We just return the history enriched with defaults.
            
            day_data = weather_history[i].copy()
            
            # Fill defaults
            for k, v in DEFAULT_WEATHER_VALUES.items():
                if k not in day_data:
                    day_data[k] = v
                    
            sequence.append(day_data)
            
        return sequence

    def create_sequences_from_df(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "temperature_celsius",
        seq_len: int = V3Config.SEQ_LEN,
        pred_len: int = V3Config.PRED_LEN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from DataFrame for training/evaluation.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            seq_len: Input sequence length
            pred_len: Output prediction length

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X = df[feature_cols].values
        y = df[target_col].values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create sliding windows
        Xs, ys = [], []
        for i in range(len(X_scaled) - seq_len - pred_len + 1):
            Xs.append(X_scaled[i : (i + seq_len)])
            ys.append(y[(i + seq_len) : (i + seq_len + pred_len)])

        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
