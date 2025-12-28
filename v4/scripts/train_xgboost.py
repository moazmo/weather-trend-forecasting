"""
V4 Train XGBoost Model.
Train XGBoost on historical weather data for temperature prediction.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# XGBoost import
try:
    from xgboost import XGBRegressor
except ImportError:
    print("Installing xgboost...")
    os.system("pip install xgboost")
    from xgboost import XGBRegressor


def load_and_prepare_data():
    """Load and prepare the GlobalWeatherRepository dataset."""
    print("Loading dataset...")
    df = pd.read_csv("data/raw/GlobalWeatherRepository.csv", low_memory=False)
    print(f"Loaded {len(df):,} rows, {df['country'].nunique()} countries")

    # Parse dates
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["date"] = df["last_updated"].dt.date

    # Add cyclical time features
    df["month"] = df["last_updated"].dt.month
    df["day_of_year"] = df["last_updated"].dt.dayofyear
    df["hour"] = df["last_updated"].dt.hour

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Add geographic features
    df["abs_latitude"] = df["latitude"].abs()
    df["latitude_normalized"] = df["abs_latitude"] / 90.0
    df["hemisphere"] = (df["latitude"] >= 0).astype(int)

    return df


def get_feature_columns():
    """Get feature columns for model training."""
    # From assessment feature importance analysis
    return [
        # Primary weather features
        "humidity",
        "pressure_mb",
        "wind_kph",
        "wind_mph",
        "cloud",
        "uv_index",
        "visibility_km",
        "gust_kph",
        "precip_mm",
        # Air quality
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_PM2.5",
        "air_quality_PM10",
        # Geographic
        "latitude",
        "longitude",
        "abs_latitude",
        "latitude_normalized",
        "hemisphere",
        # Temporal (cyclical)
        "month_sin",
        "month_cos",
        "day_year_sin",
        "day_year_cos",
    ]


def train_model():
    """Train XGBoost model."""
    df = load_and_prepare_data()

    # Feature columns
    feature_cols = get_feature_columns()

    # Ensure all features exist
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Using {len(available_features)} features")

    # Prepare data
    X = df[available_features].copy()
    y = df["temperature_celsius"].copy()

    # Handle any remaining NaN
    X = X.fillna(X.median())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    print("Training XGBoost...")
    model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_mae = np.mean(np.abs(y_train - train_pred))
    test_mae = np.mean(np.abs(y_test - test_pred))
    train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))

    print(f"\n=== Results ===")
    print(f"Train MAE: {train_mae:.2f}°C, RMSE: {train_rmse:.2f}°C")
    print(f"Test MAE:  {test_mae:.2f}°C, RMSE: {test_rmse:.2f}°C")

    # Feature importance
    importance = dict(zip(available_features, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features:")
    for i, (feat, imp) in enumerate(importance_sorted[:10], 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    # Save artifacts
    models_dir = Path("v4/models")
    models_dir.mkdir(exist_ok=True)

    # Save model
    joblib.dump(model, models_dir / "xgboost_model.joblib")
    print(f"\nSaved model to {models_dir / 'xgboost_model.joblib'}")

    # Save scaler
    joblib.dump(scaler, models_dir / "scaler.joblib")
    print(f"Saved scaler to {models_dir / 'scaler.joblib'}")

    # Save feature list
    with open(models_dir / "features.json", "w") as f:
        json.dump({"features": available_features}, f, indent=2)
    print(f"Saved features to {models_dir / 'features.json'}")

    # Save feature importance
    with open(models_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    print(f"Saved importance to {models_dir / 'feature_importance.json'}")

    # Save metadata
    metadata = {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "n_features": len(available_features),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "date_range": {
            "min": str(df["last_updated"].min()),
            "max": str(df["last_updated"].max()),
        },
        "countries": int(df["country"].nunique()),
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {models_dir / 'metadata.json'}")

    return model, scaler, test_mae


if __name__ == "__main__":
    train_model()
