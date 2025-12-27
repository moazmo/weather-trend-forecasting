"""
Tests for V3 Preprocessing Module.
Tests feature engineering, scaling, and sequence creation.
"""

from datetime import datetime

import pytest


class TestV3Config:
    """Test suite for configuration."""

    def test_config_constants(self, sample_config):
        """Config should have required constants."""
        assert sample_config.SEQ_LEN == 14
        assert sample_config.PRED_LEN == 7
        assert sample_config.D_MODEL == 128
        assert sample_config.NHEAD == 8

    def test_get_all_features(self, sample_config):
        """Should return list of all features."""
        features = sample_config.get_all_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert "latitude" in features
        assert "humidity" in features

    def test_climate_zone_tropical(self, sample_config):
        """Latitude 0-23.5 should be Tropical."""
        assert sample_config.get_climate_zone(0) == "Tropical"
        assert sample_config.get_climate_zone(20) == "Tropical"
        assert sample_config.get_climate_zone(-15) == "Tropical"

    def test_climate_zone_subtropical(self, sample_config):
        """Latitude 23.5-35 should be Subtropical."""
        assert sample_config.get_climate_zone(30) == "Subtropical"
        assert sample_config.get_climate_zone(-30) == "Subtropical"

    def test_climate_zone_temperate(self, sample_config):
        """Latitude 35-55 should be Temperate."""
        assert sample_config.get_climate_zone(45) == "Temperate"
        assert sample_config.get_climate_zone(-50) == "Temperate"

    def test_climate_zone_polar(self, sample_config):
        """Latitude 66.5-90 should be Polar."""
        assert sample_config.get_climate_zone(70) == "Polar"
        assert sample_config.get_climate_zone(-80) == "Polar"

    def test_hemisphere(self, sample_config):
        """Should correctly identify hemisphere."""
        assert sample_config.get_hemisphere(30) == 1  # Northern
        assert sample_config.get_hemisphere(-30) == 0  # Southern
        assert sample_config.get_hemisphere(0) == 1  # Equator = Northern


class TestCyclicalFeatures:
    """Test cyclical feature computation."""

    def test_cyclical_features_exist(self, scaler_exists):
        """Should compute cyclical features."""
        if not scaler_exists:
            pytest.skip("Scaler not found - preprocessor not available")

        from v3.src.preprocessing import V3Preprocessor

        date = datetime(2024, 6, 15)
        features = V3Preprocessor.compute_cyclical_features(date)

        assert "month_sin" in features
        assert "month_cos" in features
        assert "day_year_sin" in features
        assert "day_year_cos" in features

    def test_cyclical_values_range(self, scaler_exists):
        """Cyclical values should be in [-1, 1]."""
        if not scaler_exists:
            pytest.skip("Scaler not found")

        from v3.src.preprocessing import V3Preprocessor

        date = datetime(2024, 1, 1)
        features = V3Preprocessor.compute_cyclical_features(date)

        for key, value in features.items():
            assert -1 <= value <= 1, f"{key} = {value} not in [-1, 1]"


class TestGeographicFeatures:
    """Test geographic feature computation."""

    def test_geographic_features(self, scaler_exists):
        """Should compute geographic features correctly."""
        if not scaler_exists:
            pytest.skip("Scaler not found")

        from v3.src.preprocessing import V3Preprocessor

        features = V3Preprocessor.compute_geographic_features(30.0, 31.2)

        assert features["latitude"] == 30.0
        assert features["longitude"] == 31.2
        assert features["abs_latitude"] == 30.0
        assert features["hemisphere_encoded"] == 1

    def test_southern_hemisphere(self, scaler_exists):
        """Should handle southern hemisphere."""
        if not scaler_exists:
            pytest.skip("Scaler not found")

        from v3.src.preprocessing import V3Preprocessor

        features = V3Preprocessor.compute_geographic_features(-33.9, 18.4)

        assert features["abs_latitude"] == 33.9
        assert features["hemisphere_encoded"] == 0


class TestSequenceCreation:
    """Test sequence creation for time-series."""

    def test_sequence_shape(self, scaler_exists, sample_weather_data):
        """Sequence should have correct shape."""
        if not scaler_exists:
            pytest.skip("Scaler not found")

        from v3.src.config import V3Config
        from v3.src.preprocessing import V3Preprocessor

        preprocessor = V3Preprocessor()
        weather_history = [sample_weather_data.copy() for _ in range(V3Config.SEQ_LEN)]

        sequence = preprocessor.build_sequence(
            latitude=30.0,
            longitude=31.2,
            start_date=datetime(2024, 1, 15),
            weather_history=weather_history,
        )

        assert sequence.shape[0] == V3Config.SEQ_LEN
        assert len(sequence.shape) == 2  # (seq_len, n_features)

    def test_sequence_padding(self, scaler_exists, sample_weather_data):
        """Should pad short history with defaults."""
        if not scaler_exists:
            pytest.skip("Scaler not found")

        from v3.src.config import V3Config
        from v3.src.preprocessing import V3Preprocessor

        preprocessor = V3Preprocessor()
        # Only 5 days of history (less than SEQ_LEN=14)
        short_history = [sample_weather_data.copy() for _ in range(5)]

        sequence = preprocessor.build_sequence(
            latitude=30.0,
            longitude=31.2,
            start_date=datetime(2024, 1, 15),
            weather_history=short_history,
        )

        # Should still produce full sequence
        assert sequence.shape[0] == V3Config.SEQ_LEN
