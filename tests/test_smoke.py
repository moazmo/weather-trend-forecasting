import pytest
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import DataLoader
from features import TimeSeriesFeatures
from models import XGBoostForecaster
from api import app

def test_imports():
    """Simple smoke test to ensure modules import correctly."""
    assert DataLoader is not None
    assert TimeSeriesFeatures is not None
    assert XGBoostForecaster is not None
    assert app is not None

def test_feature_engineering_structure():
    """Test that feature generator produces expected columns."""
    fe = TimeSeriesFeatures()
    
    # Dummy data
    df = pd.DataFrame({
        'temperature_celsius': [10, 11, 12, 13, 14, 15]
    }, index=pd.date_range('2025-01-01', periods=6, freq='H'))
    
    # Transform
    # Note: Lags (up to 168) will produce NaNs and be dropped.
    # We just want to check logic doesn't crash.
    try:
        df_out = fe.transform(df)
        assert isinstance(df_out, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"Feature engineering failed: {e}")
