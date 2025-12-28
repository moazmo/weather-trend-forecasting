"""
V4 Inference Module (Optimized).
Uses trained XGBoost model for historical backtesting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = Path("data/raw/GlobalWeatherRepository.csv")


class V4Forecaster:
    """
    V4 Weather Forecaster using trained XGBoost model.
    Performs historical backtesting on the training dataset.
    """

    def __init__(self):
        """Initialize forecaster with trained model."""
        self.model = None
        self.scaler = None
        self.features = None
        self.df = None
        self.metadata = None
        self._loaded = False

        self._load_artifacts()
        self._load_dataset()

    def _load_artifacts(self):
        """Load model artifacts."""
        model_path = MODELS_DIR / "xgboost_model.joblib"
        scaler_path = MODELS_DIR / "scaler.joblib"
        features_path = MODELS_DIR / "features.json"
        metadata_path = MODELS_DIR / "metadata.json"

        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"Loaded XGBoost model from {model_path}")

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")

        if features_path.exists():
            with open(features_path) as f:
                self.features = json.load(f)["features"]
            print(f"Loaded {len(self.features)} features")

        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        self._loaded = self.model is not None and self.scaler is not None

    def _load_dataset(self):
        """Load the historical dataset for backtesting."""
        if DATA_PATH.exists():
            print(f"Loading dataset from {DATA_PATH}...")
            self.df = pd.read_csv(DATA_PATH, low_memory=False)
            self.df["last_updated"] = pd.to_datetime(self.df["last_updated"])
            self.df["date"] = self.df["last_updated"].dt.date

            # Add derived features
            self._add_derived_features()

            print(f"Loaded {len(self.df):,} records from {self.df['country'].nunique()} countries")
            print(f"Date range: {self.df['last_updated'].min()} to {self.df['last_updated'].max()}")
        else:
            print(f"Warning: Dataset not found at {DATA_PATH}")

    def _add_derived_features(self):
        """Add cyclical and geographic features."""
        if self.df is None:
            return

        # Cyclical time features
        self.df["month"] = self.df["last_updated"].dt.month
        self.df["day_of_year"] = self.df["last_updated"].dt.dayofyear

        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        self.df["day_year_sin"] = np.sin(2 * np.pi * self.df["day_of_year"] / 365)
        self.df["day_year_cos"] = np.cos(2 * np.pi * self.df["day_of_year"] / 365)

        # Geographic features
        self.df["abs_latitude"] = self.df["latitude"].abs()
        self.df["latitude_normalized"] = self.df["abs_latitude"] / 90.0
        self.df["hemisphere"] = (self.df["latitude"] >= 0).astype(int)

    def get_available_countries(self) -> list[dict]:
        """Get list of countries in the dataset."""
        if self.df is None:
            return []

        countries = []
        for country in self.df["country"].unique():
            country_df = self.df[self.df["country"] == country]
            sample = country_df.iloc[0]
            countries.append({
                "name": country,
                "latitude": float(sample["latitude"]),
                "longitude": float(sample["longitude"]),
                "records": len(country_df),
            })

        return sorted(countries, key=lambda x: x["name"])

    def get_date_range(self) -> dict:
        """Get the valid date range for backtesting."""
        if self.df is None:
            return {"min": None, "max": None}

        return {
            "min": self.df["last_updated"].min().strftime("%Y-%m-%d"),
            "max": self.df["last_updated"].max().strftime("%Y-%m-%d"),
        }

    def get_records_for_date_range(
        self,
        country: str = None,
        lat: float = None,
        lon: float = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Get records for a date range, optionally filtered by country/location.

        Args:
            country: Country name filter
            lat, lon: Approximate location (finds nearest)
            start_date, end_date: Date range (YYYY-MM-DD)

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            return pd.DataFrame()

        result = self.df.copy()

        # Filter by country
        if country:
            result = result[result["country"].str.lower() == country.lower()]

        # Filter by location (find nearest if country not found)
        elif lat is not None and lon is not None:
            # Find records near this location
            result["dist"] = np.sqrt(
                (result["latitude"] - lat) ** 2 + (result["longitude"] - lon) ** 2
            )
            # Get records from the nearest location
            nearest_lat = result.loc[result["dist"].idxmin(), "latitude"]
            nearest_lon = result.loc[result["dist"].idxmin(), "longitude"]
            result = result[
                (result["latitude"] == nearest_lat) & (result["longitude"] == nearest_lon)
            ]

        # Filter by date range
        if start_date:
            start = pd.to_datetime(start_date)
            result = result[result["last_updated"] >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            result = result[result["last_updated"] <= end]

        return result.sort_values("last_updated")

    def predict(
        self,
        country: str = None,
        lat: float = None,
        lon: float = None,
        start_date: str = None,
        end_date: str = None,
    ) -> dict[str, Any]:
        """
        Predict temperatures for historical data and compare with actual.

        Args:
            country: Country name
            lat, lon: Location coordinates
            start_date, end_date: Date range for backtesting

        Returns:
            Prediction results with actual comparison
        """
        if not self._loaded:
            return {"error": "Model not loaded", "predictions": []}

        # Get historical records
        records = self.get_records_for_date_range(
            country=country, lat=lat, lon=lon, start_date=start_date, end_date=end_date
        )

        if records.empty:
            return {"error": "No data found for this location/date range", "predictions": []}

        # Prepare features
        available_features = [f for f in self.features if f in records.columns]
        X = records[available_features].fillna(0).values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        # Get actual values
        actual = records["temperature_celsius"].values

        # Calculate errors
        errors = np.abs(predictions - actual)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Build response
        location_info = {
            "country": records.iloc[0]["country"],
            "latitude": float(records.iloc[0]["latitude"]),
            "longitude": float(records.iloc[0]["longitude"]),
        }

        daily_results = []
        for i, (_, row) in enumerate(records.iterrows()):
            daily_results.append({
                "date": row["last_updated"].strftime("%Y-%m-%d"),
                "datetime": row["last_updated"].strftime("%Y-%m-%d %H:%M"),
                "predicted": round(float(predictions[i]), 1),
                "actual": round(float(actual[i]), 1),
                "error": round(float(errors[i]), 2),
            })

        # Climate zone
        lat_val = location_info["latitude"]
        abs_lat = abs(lat_val)
        if abs_lat < 23.5:
            climate_zone = "Tropical"
        elif abs_lat < 35:
            climate_zone = "Subtropical"
        elif abs_lat < 55:
            climate_zone = "Temperate"
        elif abs_lat < 66.5:
            climate_zone = "Subarctic"
        else:
            climate_zone = "Polar"

        return {
            "location": location_info,
            "climate_zone": climate_zone,
            "hemisphere": "Northern" if lat_val >= 0 else "Southern",
            "predictions": daily_results,
            "summary": {
                "count": len(daily_results),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "min_predicted": round(float(predictions.min()), 1),
                "max_predicted": round(float(predictions.max()), 1),
                "min_actual": round(float(actual.min()), 1),
                "max_actual": round(float(actual.max()), 1),
            },
            "model": {
                "name": "V4 XGBoost",
                "test_mae": self.metadata.get("test_mae", 1.21) if self.metadata else 1.21,
            },
        }

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "version": "4.0",
            "name": "V4 XGBoost Weather Forecaster",
            "type": "Historical Backtesting",
            "loaded": self._loaded,
            "n_features": len(self.features) if self.features else 0,
            "features": self.features[:10] if self.features else [],
            "date_range": self.get_date_range(),
            "metadata": self.metadata,
        }
