"""
V4 Test Fixtures.
Shared pytest fixtures for V4 testing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v4.src.config import V4Config


@pytest.fixture
def sample_config():
    """Return V4Config class for testing."""
    return V4Config


@pytest.fixture
def sample_weather_data():
    """Sample weather data dictionary."""
    return {
        "humidity": 65.0,
        "pressure_mb": 1013.0,
        "wind_kph": 15.0,
        "wind_mph": 9.3,
        "cloud": 40,
        "precip_mm": 0.0,
        "uv_index": 6.0,
        "visibility_km": 10.0,
        "gust_kph": 20.0,
        "gust_mph": 12.4,
        "wind_degree": 180,
        "air_quality_Carbon_Monoxide": 200.0,
        "air_quality_Ozone": 50.0,
        "air_quality_PM2.5": 15.0,
        "air_quality_PM10": 25.0,
        "latitude": 30.0,
        "longitude": 31.0,
    }


@pytest.fixture
def sample_sequence():
    """Sample input sequence for model testing."""
    seq_len = V4Config.SEQ_LEN
    n_features = len(V4Config.get_all_features())
    return np.random.randn(seq_len, n_features).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Sample batch for model testing."""
    batch_size = 4
    seq_len = V4Config.SEQ_LEN
    n_features = len(V4Config.get_all_features())
    return np.random.randn(batch_size, seq_len, n_features).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample target values."""
    return np.random.randn(100, V4Config.PRED_LEN).astype(np.float32) * 10 + 20
