"""
V4 API Tests.
Tests for FastAPI endpoints.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v4.app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self):
        """Health response should contain status."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_version(self):
        """Health response should contain version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert "4.0" in data["version"]


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    def test_model_info_returns_200(self):
        """Model info should return 200."""
        response = client.get("/api/model-info")
        assert response.status_code == 200

    def test_model_info_contains_architecture(self):
        """Model info should contain architecture details."""
        response = client.get("/api/model-info")
        data = response.json()
        assert "version" in data
        assert "name" in data


class TestForecastEndpoint:
    """Test forecast endpoints."""

    def test_forecast_post_valid(self):
        """POST forecast with valid data should work."""
        response = client.post(
            "/api/forecast",
            json={"lat": 30.0, "lon": 31.0, "start_date": "2024-06-15"},
        )
        # May return 503 if forecaster not loaded, or 200 if successful
        assert response.status_code in [200, 503]

    def test_forecast_post_invalid_lat(self):
        """POST forecast with invalid latitude should fail."""
        response = client.post(
            "/api/forecast",
            json={"lat": 100.0, "lon": 31.0, "start_date": "2024-06-15"},
        )
        assert response.status_code == 422

    def test_forecast_get_valid(self):
        """GET forecast with valid params should work."""
        response = client.get("/api/forecast?lat=30&lon=31&start_date=2024-06-15")
        assert response.status_code in [200, 503]


class TestCountriesEndpoint:
    """Test countries endpoint."""

    def test_countries_returns_200(self):
        """Countries endpoint should return 200."""
        response = client.get("/api/countries")
        assert response.status_code == 200

    def test_countries_contains_list(self):
        """Countries response should contain list."""
        response = client.get("/api/countries")
        data = response.json()
        assert "countries" in data
        assert isinstance(data["countries"], list)
        assert len(data["countries"]) > 0


class TestNearestEndpoint:
    """Test nearest country endpoint."""

    def test_nearest_returns_200(self):
        """Nearest endpoint should return 200."""
        response = client.get("/api/nearest?lat=30&lon=31")
        assert response.status_code == 200

    def test_nearest_contains_country(self):
        """Nearest response should contain country."""
        response = client.get("/api/nearest?lat=30&lon=31")
        data = response.json()
        assert "nearest" in data
        assert "name" in data["nearest"]


class TestClimateZonesEndpoint:
    """Test climate zones endpoint."""

    def test_climate_zones_returns_200(self):
        """Climate zones should return 200."""
        response = client.get("/api/climate-zones")
        assert response.status_code == 200

    def test_climate_zones_contains_zones(self):
        """Response should contain zone list."""
        response = client.get("/api/climate-zones")
        data = response.json()
        assert "zones" in data
        assert len(data["zones"]) == 5


class TestFeatureImportanceEndpoint:
    """Test feature importance endpoint."""

    def test_feature_importance_returns_200(self):
        """Feature importance should return 200."""
        response = client.get("/api/feature-importance")
        assert response.status_code == 200

    def test_feature_importance_contains_rankings(self):
        """Response should contain rankings."""
        response = client.get("/api/feature-importance")
        data = response.json()
        assert "rankings" in data
        assert len(data["rankings"]) > 0


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_html(self):
        """Root should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
