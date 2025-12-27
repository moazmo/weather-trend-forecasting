"""
Pytest Fixtures for V3 Tests.
Shared fixtures for model, data, and API testing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v3.src.config import MODELS_DIR, SCALER_PATH, V3Config
from v3.src.model import GatedResidualNetwork, V3ClimateTransformer


@pytest.fixture
def sample_config():
    """Return V3Config class for testing."""
    return V3Config


@pytest.fixture
def sample_input():
    """Create sample input tensor for model testing."""
    batch_size = 4
    seq_len = V3Config.SEQ_LEN
    input_dim = 25  # Approximate number of features
    return torch.randn(batch_size, seq_len, input_dim)


@pytest.fixture
def sample_model():
    """Create a small model instance for testing."""
    return V3ClimateTransformer(
        input_dim=25,
        d_model=32,  # Smaller for faster tests
        nhead=4,
        num_layers=2,
        dropout=0.1,
        seq_len=V3Config.SEQ_LEN,
        pred_len=V3Config.PRED_LEN,
    )


@pytest.fixture
def sample_grn():
    """Create a GRN instance for testing."""
    return GatedResidualNetwork(input_dim=25, hidden_dim=64, output_dim=32, dropout=0.1)


@pytest.fixture
def sample_weather_data():
    """Create sample weather data dictionary."""
    return {
        "humidity": 65.0,
        "pressure_mb": 1015.0,
        "wind_kph": 12.0,
        "cloud": 40.0,
        "precip_mm": 0.5,
        "uv_index": 6.0,
        "visibility_km": 10.0,
        "gust_kph": 18.0,
        "wind_degree": 180.0,
        "air_quality_Ozone": 45.0,
        "air_quality_Nitrogen_dioxide": 15.0,
        "air_quality_PM2.5": 12.0,
        "air_quality_Carbon_Monoxide": 400.0,
        "air_quality_Sulphur_dioxide": 8.0,
    }


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for preprocessing tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "latitude": np.random.uniform(-90, 90, n),
            "longitude": np.random.uniform(-180, 180, n),
            "temperature_celsius": np.random.uniform(-10, 40, n),
            "humidity": np.random.uniform(0, 100, n),
            "pressure_mb": np.random.uniform(980, 1040, n),
            "wind_kph": np.random.uniform(0, 50, n),
            "cloud": np.random.uniform(0, 100, n),
            "month_sin": np.sin(2 * np.pi * np.random.randint(1, 13, n) / 12),
            "month_cos": np.cos(2 * np.pi * np.random.randint(1, 13, n) / 12),
        }
    )


@pytest.fixture
def models_exist():
    """Check if trained model artifacts exist."""
    return MODELS_DIR.exists() and (MODELS_DIR / "v3_climate_transformer.pt").exists()


@pytest.fixture
def scaler_exists():
    """Check if scaler artifact exists."""
    return SCALER_PATH.exists()
