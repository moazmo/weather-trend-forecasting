"""
V4 Configuration Module.
Central configuration for the V4 Ensemble Weather Forecaster.
Based on assessment analysis findings.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = Path("data")

# Model artifact paths
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_model.pt"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

# =============================================================================
# Model Configuration
# =============================================================================


class V4Config:
    """V4 Ensemble Model Configuration."""

    # Sequence parameters
    SEQ_LEN = 14  # 14-day input sequence
    PRED_LEN = 7  # 7-day forecast

    # XGBoost hyperparameters (from assessment grid search)
    XGBOOST_PARAMS = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
    }

    # Transformer hyperparameters (from V2)
    TRANSFORMER_PARAMS = {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.2,
    }

    # Ensemble weights by climate zone (XGBoost, Transformer)
    ENSEMBLE_WEIGHTS = {
        "Tropical": (0.6, 0.4),
        "Subtropical": (0.5, 0.5),
        "Temperate": (0.4, 0.6),
        "Subarctic": (0.5, 0.5),
        "Polar": (0.6, 0.4),
    }

    # =============================================================================
    # Features (from assessment feature_importance_comparison.csv)
    # =============================================================================

    # Top 15 features by average importance rank
    TOP_FEATURES = [
        "humidity",
        "latitude",
        "uv_index",
        "pressure_mb",
        "longitude",
        "wind_mph",
        "gust_mph",
        "air_quality_Ozone",
        "air_quality_PM10",
        "gust_kph",
        "cloud",
        "air_quality_Carbon_Monoxide",
        "air_quality_PM2.5",
        "wind_degree",
        "visibility_km",
    ]

    # Cyclical features (auto-derived)
    CYCLICAL_FEATURES = [
        "month_sin",
        "month_cos",
        "day_year_sin",
        "day_year_cos",
        "hour_sin",
        "hour_cos",
    ]

    # Geographic features
    GEOGRAPHIC_FEATURES = [
        "abs_latitude",
        "latitude_normalized",
        "hemisphere_encoded",
        "climate_zone_encoded",
    ]

    # Air quality features
    AIR_QUALITY_FEATURES = [
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_PM2.5",
        "air_quality_PM10",
    ]

    # =============================================================================
    # Anomaly Detection Thresholds (from assessment anomaly_summary.json)
    # =============================================================================

    # IQR bounds for anomaly filtering
    ANOMALY_BOUNDS = {
        "temperature_celsius": {"lower": 0.7, "upper": 44.7},
        "humidity": {"lower": 0, "upper": 100},
        "pressure_mb": {"lower": 998, "upper": 1030},
        "wind_kph": {"lower": 0, "upper": 35.25},
        "precip_mm": {"lower": 0, "upper": 50},
        "uv_index": {"lower": 0, "upper": 14.7},
    }

    # Isolation Forest contamination (from assessment: 5711/114289 = ~5%)
    ISOLATION_FOREST_CONTAMINATION = 0.05

    # =============================================================================
    # Climate Zones
    # =============================================================================

    CLIMATE_ZONES = {
        "Tropical": {"lat_range": (0, 23.5)},
        "Subtropical": {"lat_range": (23.5, 35)},
        "Temperate": {"lat_range": (35, 55)},
        "Subarctic": {"lat_range": (55, 66.5)},
        "Polar": {"lat_range": (66.5, 90)},
    }

    @classmethod
    def get_climate_zone(cls, latitude: float) -> str:
        """Determine climate zone from latitude."""
        abs_lat = abs(latitude)
        for zone, config in cls.CLIMATE_ZONES.items():
            if config["lat_range"][0] <= abs_lat < config["lat_range"][1]:
                return zone
        return "Polar"

    @classmethod
    def get_hemisphere(cls, latitude: float) -> str:
        """Determine hemisphere from latitude."""
        return "Northern" if latitude >= 0 else "Southern"

    @classmethod
    def get_all_features(cls) -> list:
        """Get all features for model input."""
        return cls.TOP_FEATURES + cls.CYCLICAL_FEATURES + cls.GEOGRAPHIC_FEATURES


# Default weather values for missing data imputation
DEFAULT_WEATHER_VALUES = {
    "humidity": 60.0,
    "pressure_mb": 1013.25,
    "wind_kph": 10.0,
    "wind_mph": 6.2,
    "wind_degree": 180,
    "cloud": 50,
    "precip_mm": 0.0,
    "uv_index": 5.0,
    "visibility_km": 10.0,
    "gust_kph": 15.0,
    "gust_mph": 9.3,
    "air_quality_Carbon_Monoxide": 200.0,
    "air_quality_Ozone": 50.0,
    "air_quality_PM2.5": 15.0,
    "air_quality_PM10": 25.0,
}
