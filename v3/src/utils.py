"""
V3 Utility Functions
Helper functions for common operations.
"""

from datetime import datetime
from typing import Any

import numpy as np


def get_climate_zone(latitude: float) -> str:
    """
    Determine climate zone from latitude.

    Args:
        latitude: Latitude in degrees (-90 to 90)

    Returns:
        Climate zone name
    """
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


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate geographic coordinates.

    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
    return True


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Datetime object
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD") from e


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression error metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, MAPE, R²
    """
    residuals = y_pred - y_true

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mape = np.mean(np.abs(residuals / (y_true + 1e-8))) * 100

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
    }


def format_forecast_response(
    predictions: np.ndarray, start_date: datetime, latitude: float, longitude: float
) -> dict[str, Any]:
    """
    Format model predictions into API response.

    Args:
        predictions: Array of predicted temperatures
        start_date: First prediction date
        latitude: Location latitude
        longitude: Location longitude

    Returns:
        Formatted response dictionary
    """
    from datetime import timedelta

    forecast = []
    for i, temp in enumerate(predictions):
        date = start_date + timedelta(days=i)
        forecast.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "day": date.strftime("%A"),
                "temperature": round(float(temp), 1),
            }
        )

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "climate_zone": get_climate_zone(latitude),
            "hemisphere": "Northern" if latitude >= 0 else "Southern",
        },
        "forecast": forecast,
    }
