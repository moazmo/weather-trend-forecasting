"""
V4 Preprocessing Module.
Feature engineering and data preparation for V4 Ensemble Forecaster.
Based on assessment feature importance analysis.
"""

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from v4.src.config import DEFAULT_WEATHER_VALUES, V4Config
from v4.src.utils import compute_cyclical_features, compute_geographic_features


class V4Preprocessor:
    """
    V4 Feature Engineering and Preprocessing.
    Uses top features identified in SHAP analysis.
    """

    def __init__(self, scaler=None):
        """
        Initialize preprocessor.

        Args:
            scaler: Optional pre-fitted StandardScaler
        """
        self.scaler = scaler
        self.feature_names = V4Config.get_all_features()

    def build_feature_vector(
        self,
        weather_data: dict[str, Any],
        latitude: float,
        longitude: float,
        date: datetime,
    ) -> np.ndarray:
        """
        Build feature vector from raw weather data.

        Args:
            weather_data: Dictionary with weather measurements
            latitude: Location latitude
            longitude: Location longitude
            date: Observation date

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Top features from assessment
        for feature in V4Config.TOP_FEATURES:
            value = weather_data.get(feature, DEFAULT_WEATHER_VALUES.get(feature, 0))
            features.append(float(value) if value is not None else 0.0)

        # Cyclical features
        cyclical = compute_cyclical_features(date)
        for feature in V4Config.CYCLICAL_FEATURES:
            features.append(cyclical.get(feature, 0.0))

        # Geographic features
        geo = compute_geographic_features(latitude, longitude)
        for feature in V4Config.GEOGRAPHIC_FEATURES:
            features.append(geo.get(feature, 0.0))

        return np.array(features, dtype=np.float32)

    def build_sequence(
        self,
        weather_history: list[dict[str, Any]],
        latitude: float,
        longitude: float,
        start_date: datetime,
    ) -> np.ndarray:
        """
        Build input sequence from weather history.

        Args:
            weather_history: List of daily weather observations
            latitude: Location latitude
            longitude: Location longitude
            start_date: Sequence start date

        Returns:
            Sequence array of shape (seq_len, n_features)
        """
        seq_len = V4Config.SEQ_LEN
        sequence = []

        for i in range(seq_len):
            date = start_date + timedelta(days=i)

            if i < len(weather_history):
                weather = weather_history[i]
            else:
                # Pad with default values
                weather = DEFAULT_WEATHER_VALUES.copy()

            features = self.build_feature_vector(weather, latitude, longitude, date)
            sequence.append(features)

        sequence = np.array(sequence, dtype=np.float32)

        # Apply scaling if scaler available
        if self.scaler is not None:
            original_shape = sequence.shape
            sequence = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
            sequence = sequence.reshape(original_shape)

        return sequence

    def create_sequences_from_df(
        self, df: pd.DataFrame, seq_len: int = None, pred_len: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences from DataFrame.

        Args:
            df: DataFrame with weather data (must include 'temperature_celsius')
            seq_len: Input sequence length (default: V4Config.SEQ_LEN)
            pred_len: Prediction length (default: V4Config.PRED_LEN)

        Returns:
            Tuple of (X, y) arrays
        """
        seq_len = seq_len or V4Config.SEQ_LEN
        pred_len = pred_len or V4Config.PRED_LEN

        # Determine available features
        available_features = [f for f in V4Config.TOP_FEATURES if f in df.columns]

        # Add cyclical features if not present
        if "month_sin" not in df.columns and "last_updated" in df.columns:
            df["last_updated"] = pd.to_datetime(df["last_updated"])
            df["month"] = df["last_updated"].dt.month
            df["day_of_year"] = df["last_updated"].dt.dayofyear
            df["hour"] = df["last_updated"].dt.hour.fillna(12)

            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
            df["day_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
            df["day_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Add geographic features if not present
        if "abs_latitude" not in df.columns and "latitude" in df.columns:
            df["abs_latitude"] = df["latitude"].abs()
            df["latitude_normalized"] = df["abs_latitude"] / 90.0
            df["hemisphere_encoded"] = (df["latitude"] >= 0).astype(int)

        # Collect all available features
        all_features = []
        for f in V4Config.get_all_features():
            if f in df.columns:
                all_features.append(f)

        if len(all_features) == 0:
            raise ValueError("No valid features found in DataFrame")

        # Create sequences
        X, y = [], []
        target = "temperature_celsius"

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        data = df[all_features + [target]].dropna().values

        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i : i + seq_len, :-1])  # Features only
            y.append(data[i + seq_len : i + seq_len + pred_len, -1])  # Target

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def filter_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter anomalies using IQR bounds from assessment.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered = df.copy()

        for col, bounds in V4Config.ANOMALY_BOUNDS.items():
            if col in filtered.columns:
                mask = (filtered[col] >= bounds["lower"]) & (filtered[col] <= bounds["upper"])
                filtered = filtered[mask]

        return filtered

    def prepare_xgboost_features(
        self, sequence: np.ndarray
    ) -> np.ndarray:
        """
        Flatten sequence for XGBoost input.

        Args:
            sequence: Shape (seq_len, n_features)

        Returns:
            Flattened array of shape (seq_len * n_features,)
        """
        return sequence.flatten()

    def prepare_batch_for_xgboost(
        self, sequences: np.ndarray
    ) -> np.ndarray:
        """
        Prepare batch of sequences for XGBoost.

        Args:
            sequences: Shape (batch_size, seq_len, n_features)

        Returns:
            Flattened array of shape (batch_size, seq_len * n_features)
        """
        batch_size = sequences.shape[0]
        return sequences.reshape(batch_size, -1)
