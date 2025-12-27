"""
V3 Configuration Module
Central configuration for paths, feature columns, and hyperparameters.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Base paths (relative to v3/ directory)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Model artifacts
MODEL_CHECKPOINT = MODELS_DIR / "v3_climate_transformer.pt"
SCALER_PATH = MODELS_DIR / "v3_scaler.joblib"

# =============================================================================
# Feature Configuration
# =============================================================================


class V3Config:
    """Central configuration for V3 model."""

    # Sequence parameters
    SEQ_LEN: int = 14  # 14 days of history
    PRED_LEN: int = 7  # 7 days forecast

    # Model architecture
    D_MODEL: int = 128
    NHEAD: int = 8
    NUM_LAYERS: int = 4
    DROPOUT: float = 0.2

    # Training
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 50
    PATIENCE: int = 5

    # Feature Tiers (for UI input handling)
    TIER_1_FEATURES: list[str] = [
        # Essential - Always required
        "latitude",
        "longitude",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "cloud",
    ]

    TIER_2_FEATURES: list[str] = [
        # Important - Expandable section
        "precip_mm",
        "uv_index",
        "visibility_km",
        "gust_kph",
        "wind_degree",
    ]

    TIER_3_FEATURES: list[str] = [
        # Air Quality - Optional
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_PM2.5",
        "air_quality_Carbon_Monoxide",
        "air_quality_Sulphur_dioxide",
    ]

    AUTO_DERIVED_FEATURES: list[str] = [
        # Computed from date/location
        "month_sin",
        "month_cos",
        "day_year_sin",
        "day_year_cos",
        "hour_sin",
        "hour_cos",
        "hemisphere_encoded",
        "abs_latitude",
        "latitude_normalized",
    ]

    @classmethod
    def get_all_features(cls) -> list[str]:
        """Return all feature columns in order."""
        return (
            ["latitude", "longitude", "abs_latitude", "latitude_normalized", "hemisphere_encoded"]
            + [
                "humidity",
                "pressure_mb",
                "wind_kph",
                "cloud",
                "precip_mm",
                "uv_index",
                "visibility_km",
                "gust_kph",
                "wind_degree",
            ]
            + cls.TIER_3_FEATURES
            + ["month_sin", "month_cos", "day_year_sin", "day_year_cos", "hour_sin", "hour_cos"]
        )

    @classmethod
    def get_climate_zone(cls, latitude: float) -> str:
        """Determine climate zone from latitude."""
        lat = abs(latitude)
        if lat < 23.5:
            return "Tropical"
        elif lat < 35:
            return "Subtropical"
        elif lat < 55:
            return "Temperate"
        elif lat < 66.5:
            return "Subarctic"
        else:
            return "Polar"

    @classmethod
    def get_hemisphere(cls, latitude: float) -> int:
        """Return 1 for Northern, 0 for Southern hemisphere."""
        return 1 if latitude >= 0 else 0


# =============================================================================
# Default Values for Missing Inputs
# =============================================================================

DEFAULT_WEATHER_VALUES: dict[str, float] = {
    "humidity": 50.0,
    "pressure_mb": 1013.0,
    "wind_kph": 10.0,
    "cloud": 50.0,
    "precip_mm": 0.0,
    "uv_index": 5.0,
    "visibility_km": 10.0,
    "gust_kph": 15.0,
    "wind_degree": 180.0,
    "air_quality_Ozone": 50.0,
    "air_quality_Nitrogen_dioxide": 20.0,
    "air_quality_PM2.5": 15.0,
    "air_quality_Carbon_Monoxide": 500.0,
    "air_quality_Sulphur_dioxide": 10.0,
}
