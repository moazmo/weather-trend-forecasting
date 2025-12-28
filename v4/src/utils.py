"""
V4 Utility Functions.
Helper functions for V4 Weather Forecaster.
"""

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np


def get_climate_zone(latitude: float) -> str:
    """
    Determine climate zone from latitude.

    Args:
        latitude: Geographic latitude (-90 to 90)

    Returns:
        Climate zone name (Tropical, Subtropical, Temperate, Subarctic, Polar)
    """
    abs_lat = abs(latitude)
    if abs_lat < 23.5:
        return "Tropical"
    elif abs_lat < 35:
        return "Subtropical"
    elif abs_lat < 55:
        return "Temperate"
    elif abs_lat < 66.5:
        return "Subarctic"
    else:
        return "Polar"


def get_hemisphere(latitude: float) -> str:
    """Determine hemisphere from latitude."""
    return "Northern" if latitude >= 0 else "Southern"


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate geographic coordinates.

    Args:
        latitude: Must be between -90 and 90
        longitude: Must be between -180 and 180

    Returns:
        True if valid

    Raises:
        ValueError: If coordinates are out of range
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
        date_str: Date in YYYY-MM-DD format

    Returns:
        datetime object

    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression error metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with mae, rmse, mape, r2
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "r2": round(float(r2), 4),
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def format_forecast_response(
    predictions: np.ndarray,
    start_date: datetime,
    latitude: float,
    longitude: float,
    confidence_intervals: np.ndarray | None = None,
    actual_temps: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Format predictions into structured API response.

    Args:
        predictions: Array of predicted temperatures
        start_date: Forecast start date
        latitude: Location latitude
        longitude: Location longitude
        confidence_intervals: Optional (lower, upper) bounds
        actual_temps: Optional actual temperatures for comparison

    Returns:
        Formatted response dictionary
    """
    climate_zone = get_climate_zone(latitude)
    hemisphere = get_hemisphere(latitude)

    forecast = []
    for i, temp in enumerate(predictions):
        day_date = start_date + timedelta(days=i)
        day_data = {
            "date": day_date.strftime("%Y-%m-%d"),
            "day": day_date.strftime("%A"),
            "day_index": i + 1,
            "temperature": round(float(temp), 1),
        }

        # Add confidence intervals if available
        if confidence_intervals is not None:
            day_data["confidence_lower"] = round(float(confidence_intervals[0][i]), 1)
            day_data["confidence_upper"] = round(float(confidence_intervals[1][i]), 1)

        # Add actual temperature if available
        if actual_temps is not None and i < len(actual_temps):
            if actual_temps[i] is not None:
                day_data["actual"] = round(float(actual_temps[i]), 1)
                day_data["error"] = round(abs(float(temp) - float(actual_temps[i])), 1)

        forecast.append(day_data)

    # Calculate summary statistics
    temps = predictions
    summary = {
        "min": round(float(np.min(temps)), 1),
        "max": round(float(np.max(temps)), 1),
        "avg": round(float(np.mean(temps)), 1),
        "trend": "warming" if temps[-1] > temps[0] else "cooling" if temps[-1] < temps[0] else "stable",
    }

    # Add MAE if actual data available
    if actual_temps is not None:
        valid_actuals = [a for a in actual_temps if a is not None]
        if valid_actuals:
            valid_preds = predictions[: len(valid_actuals)]
            metrics = compute_error_metrics(np.array(valid_actuals), valid_preds)
            summary["mae"] = metrics["mae"]
            summary["has_actual"] = True
            summary["actual_days"] = len(valid_actuals)
    else:
        summary["has_actual"] = False

    return {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "climate_zone": climate_zone,
            "hemisphere": hemisphere,
        },
        "forecast": forecast,
        "summary": summary,
        "model": "V4 Ensemble (XGBoost + Transformer)",
    }


def compute_cyclical_features(date: datetime) -> dict[str, float]:
    """
    Compute cyclical time features.

    Args:
        date: Input datetime

    Returns:
        Dictionary with cyclical features
    """
    month = date.month
    day_of_year = date.timetuple().tm_yday
    hour = date.hour

    return {
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "day_year_sin": math.sin(2 * math.pi * day_of_year / 365),
        "day_year_cos": math.cos(2 * math.pi * day_of_year / 365),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
    }


def compute_geographic_features(latitude: float, longitude: float) -> dict[str, float]:
    """
    Compute geographic features.

    Args:
        latitude: Location latitude
        longitude: Location longitude

    Returns:
        Dictionary with geographic features
    """
    climate_zone = get_climate_zone(latitude)
    zone_encoding = {
        "Tropical": 0,
        "Subtropical": 1,
        "Temperate": 2,
        "Subarctic": 3,
        "Polar": 4,
    }

    return {
        "abs_latitude": abs(latitude),
        "latitude_normalized": abs(latitude) / 90.0,
        "hemisphere_encoded": 1 if latitude >= 0 else 0,
        "climate_zone_encoded": zone_encoding.get(climate_zone, 2),
    }
