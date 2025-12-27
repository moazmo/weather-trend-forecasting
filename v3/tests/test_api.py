"""
Tests for V3 FastAPI Application.
Tests API endpoints and responses.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(models_exist, scaler_exists):
    """Create test client for FastAPI app."""
    if not models_exist or not scaler_exists:
        pytest.skip("Model artifacts not found")

    from v3.app.main import app

    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestForecastEndpoints:
    """Test forecast API endpoints."""

    def test_forecast_post_success(self, client):
        """POST /api/forecast should return forecast."""
        response = client.post(
            "/api/forecast", json={"latitude": 30.0, "longitude": 31.2, "start_date": "2024-01-15"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 7

    def test_forecast_post_with_scenario(self, client):
        """POST /api/forecast should accept weather scenarios."""
        response = client.post(
            "/api/forecast",
            json={
                "latitude": 30.0,
                "longitude": 31.2,
                "start_date": "2024-01-15",
                "weather_scenario": {"humidity": 80, "pressure_mb": 1000, "wind_kph": 20},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["forecast"]) == 7

    def test_forecast_get_success(self, client):
        """GET /api/forecast should return forecast."""
        response = client.get("/api/forecast?lat=30.0&lon=31.2")
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data

    def test_forecast_response_structure(self, client):
        """Forecast response should have correct structure."""
        response = client.post(
            "/api/forecast", json={"latitude": 45.0, "longitude": -0.1, "start_date": "2024-06-01"}
        )
        data = response.json()

        assert "location" in data
        assert "latitude" in data["location"]
        assert "longitude" in data["location"]
        assert "climate_zone" in data["location"]
        assert "hemisphere" in data["location"]

        assert "model_info" in data
        assert "forecast" in data

        for day in data["forecast"]:
            assert "date" in day
            assert "day" in day
            assert "temperature" in day

    def test_forecast_invalid_latitude(self, client):
        """Should reject invalid latitude."""
        response = client.post(
            "/api/forecast",
            json={"latitude": 100.0, "longitude": 31.2, "start_date": "2024-01-15"},  # Invalid
        )
        assert response.status_code == 422  # Validation error

    def test_forecast_invalid_longitude(self, client):
        """Should reject invalid longitude."""
        response = client.post(
            "/api/forecast",
            json={"latitude": 30.0, "longitude": 200.0, "start_date": "2024-01-15"},  # Invalid
        )
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    def test_model_info(self, client):
        """GET /api/model-info should return model metadata."""
        response = client.get("/api/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "input_dim" in data or "d_model" in data


class TestClimateZonesEndpoint:
    """Test climate zones endpoint."""

    def test_climate_zones(self, client):
        """GET /api/climate-zones should return zone definitions."""
        response = client.get("/api/climate-zones")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert len(data["zones"]) == 5  # 5 climate zones


class TestRootEndpoint:
    """Test root UI endpoint."""

    def test_root_returns_html(self, client):
        """GET / should return HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
