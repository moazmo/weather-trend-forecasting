"""
Tests for V3 Inference Module.
Tests high-level forecasting API.
"""

import pytest


class TestV3Forecaster:
    """Test suite for V3Forecaster inference API."""

    def test_forecaster_initialization(self, models_exist, scaler_exists):
        """Forecaster should initialize without errors."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        assert forecaster is not None
        assert forecaster.model is not None
        assert forecaster.preprocessor is not None

    def test_predict_returns_dict(self, models_exist, scaler_exists):
        """Predict should return a dictionary."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        result = forecaster.predict(latitude=30.0, longitude=31.2, start_date="2024-01-15")

        assert isinstance(result, dict)
        assert "predictions" in result
        assert "forecast" in result
        assert "climate_zone" in result

    def test_predict_forecast_length(self, models_exist, scaler_exists):
        """Should return 7-day forecast."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        result = forecaster.predict(latitude=30.0, longitude=31.2, start_date="2024-01-15")

        assert len(result["predictions"]) == 7
        assert len(result["forecast"]) == 7

    def test_predict_forecast_structure(self, models_exist, scaler_exists):
        """Forecast items should have correct structure."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        result = forecaster.predict(latitude=30.0, longitude=31.2, start_date="2024-01-15")

        for day in result["forecast"]:
            assert "date" in day
            assert "day" in day
            assert "temperature" in day
            assert isinstance(day["temperature"], (int, float))

    def test_predict_climate_zone(self, models_exist, scaler_exists):
        """Should correctly identify climate zone."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()

        # Egypt (30°N) = Subtropical
        result = forecaster.predict(latitude=30.0, longitude=31.2, start_date="2024-01-15")
        assert result["climate_zone"] == "Subtropical"

        # Equator = Tropical
        result = forecaster.predict(latitude=0.0, longitude=0.0, start_date="2024-01-15")
        assert result["climate_zone"] == "Tropical"

    def test_predict_hemisphere(self, models_exist, scaler_exists):
        """Should correctly identify hemisphere."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()

        result_north = forecaster.predict(latitude=45.0, longitude=0.0, start_date="2024-01-15")
        assert result_north["hemisphere"] == "Northern"

        result_south = forecaster.predict(latitude=-33.0, longitude=18.0, start_date="2024-01-15")
        assert result_south["hemisphere"] == "Southern"

    def test_predict_with_weather_history(self, models_exist, scaler_exists, sample_weather_data):
        """Should accept custom weather history."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        weather_history = [sample_weather_data.copy() for _ in range(14)]

        result = forecaster.predict(
            latitude=30.0, longitude=31.2, start_date="2024-01-15", weather_history=weather_history
        )

        assert len(result["predictions"]) == 7

    def test_get_model_info(self, models_exist, scaler_exists):
        """Should return model metadata."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()
        info = forecaster.get_model_info()

        assert "input_dim" in info
        assert "d_model" in info
        assert "device" in info


class TestBatchPrediction:
    """Test batch prediction functionality."""

    def test_batch_predict(self, models_exist, scaler_exists):
        """Should handle batch predictions."""
        if not models_exist or not scaler_exists:
            pytest.skip("Model artifacts not found")

        import numpy as np

        from v3.src.config import V3Config
        from v3.src.inference import V3Forecaster

        forecaster = V3Forecaster()

        # Create dummy batch
        batch_size = 8
        seq_len = V3Config.SEQ_LEN
        n_features = 25
        sequences = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)

        predictions = forecaster.predict_batch(sequences)

        assert predictions.shape == (batch_size, V3Config.PRED_LEN)
