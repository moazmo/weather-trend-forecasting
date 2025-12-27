"""
Tests for V3 Utility Functions.
Tests validation, error metrics, and helpers.
"""

from datetime import datetime

import numpy as np
import pytest


class TestValidation:
    """Test coordinate validation."""

    def test_valid_coordinates(self):
        """Should accept valid coordinates."""
        from v3.src.utils import validate_coordinates

        assert validate_coordinates(0, 0) is True
        assert validate_coordinates(90, 180) is True
        assert validate_coordinates(-90, -180) is True
        assert validate_coordinates(45.5, -73.6) is True

    def test_invalid_latitude(self):
        """Should reject invalid latitude."""
        from v3.src.utils import validate_coordinates

        with pytest.raises(ValueError):
            validate_coordinates(91, 0)

        with pytest.raises(ValueError):
            validate_coordinates(-91, 0)

    def test_invalid_longitude(self):
        """Should reject invalid longitude."""
        from v3.src.utils import validate_coordinates

        with pytest.raises(ValueError):
            validate_coordinates(0, 181)

        with pytest.raises(ValueError):
            validate_coordinates(0, -181)


class TestDateParsing:
    """Test date string parsing."""

    def test_valid_date(self):
        """Should parse valid date strings."""
        from v3.src.utils import parse_date

        result = parse_date("2024-01-15")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_date_format(self):
        """Should reject invalid date formats."""
        from v3.src.utils import parse_date

        with pytest.raises(ValueError):
            parse_date("01-15-2024")  # Wrong format

        with pytest.raises(ValueError):
            parse_date("2024/01/15")  # Wrong separator


class TestErrorMetrics:
    """Test error metric calculations."""

    def test_compute_error_metrics(self):
        """Should compute MAE, RMSE, MAPE, R²."""
        from v3.src.utils import compute_error_metrics

        y_true = np.array([20.0, 22.0, 25.0, 23.0, 21.0])
        y_pred = np.array([19.0, 23.0, 24.0, 24.0, 20.0])

        metrics = compute_error_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics

    def test_perfect_prediction(self):
        """Perfect prediction should have MAE=0, R²=1."""
        from v3.src.utils import compute_error_metrics

        y = np.array([20.0, 22.0, 25.0])
        metrics = compute_error_metrics(y, y)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_metrics_reasonable_range(self):
        """Metrics should be in reasonable ranges."""
        from v3.src.utils import compute_error_metrics

        y_true = np.array([20.0, 22.0, 25.0, 23.0, 21.0])
        y_pred = np.array([21.0, 21.0, 26.0, 22.0, 22.0])

        metrics = compute_error_metrics(y_true, y_pred)

        assert 0 <= metrics["mae"] <= 100
        assert 0 <= metrics["rmse"] <= 100
        assert 0 <= metrics["mape"] <= 100
        assert -10 <= metrics["r2"] <= 1


class TestClimateZone:
    """Test climate zone helper."""

    def test_climate_zones(self):
        """Should classify all latitude ranges correctly."""
        from v3.src.utils import get_climate_zone

        assert get_climate_zone(10) == "Tropical"
        assert get_climate_zone(30) == "Subtropical"
        assert get_climate_zone(45) == "Temperate"
        assert get_climate_zone(60) == "Subarctic"
        assert get_climate_zone(75) == "Polar"

    def test_climate_zone_southern(self):
        """Should work for southern hemisphere."""
        from v3.src.utils import get_climate_zone

        assert get_climate_zone(-10) == "Tropical"
        assert get_climate_zone(-45) == "Temperate"
        assert get_climate_zone(-75) == "Polar"


class TestFormatForecastResponse:
    """Test response formatting."""

    def test_format_forecast_response(self):
        """Should format predictions into structured response."""
        from v3.src.utils import format_forecast_response

        predictions = np.array([20.0, 21.0, 22.0, 23.0, 22.0, 21.0, 20.0])
        start_date = datetime(2024, 1, 15)

        response = format_forecast_response(predictions, start_date, 30.0, 31.2)

        assert "location" in response
        assert "forecast" in response
        assert len(response["forecast"]) == 7

        assert response["location"]["climate_zone"] == "Subtropical"
        assert response["forecast"][0]["date"] == "2024-01-15"
        assert response["forecast"][0]["day"] == "Monday"
