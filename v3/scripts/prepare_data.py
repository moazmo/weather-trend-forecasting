"""
Data Preparation Script for DVC Pipeline
Cleans raw data and outputs processed dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_data():
    """Load, clean, and prepare raw weather data."""

    # Paths
    raw_path = Path("data/raw/GlobalWeatherRepository.csv")
    out_path = Path("data/processed/weather_v3_ready.csv")
    stats_path = Path("data/processed/data_stats.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    initial_rows = len(df)
    print(f"📊 Initial rows: {initial_rows:,}")

    # ==========================================================================
    # Data Cleaning
    # ==========================================================================

    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 3. Clean country names
    if "country" in df.columns:
        df["country"] = df["country"].str.strip()

    # 4. Parse datetime
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        df["month"] = df["last_updated"].dt.month
        df["day_of_year"] = df["last_updated"].dt.dayofyear
        df["hour"] = df["last_updated"].dt.hour.fillna(12)

    # ==========================================================================
    # Feature Engineering
    # ==========================================================================

    # Geographic features
    if "latitude" in df.columns:
        df["abs_latitude"] = df["latitude"].abs()
        df["latitude_normalized"] = df["abs_latitude"] / 90.0
        df["hemisphere_encoded"] = (df["latitude"] >= 0).astype(int)

    # Cyclical features
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "day_of_year" in df.columns:
        df["day_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ==========================================================================
    # Save
    # ==========================================================================

    final_rows = len(df)
    print(f"✅ Final rows: {final_rows:,} (dropped {initial_rows - final_rows:,})")

    # Save processed data
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")

    # Save statistics
    stats = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "columns": len(df.columns),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "missing_values": int(df.isnull().sum().sum()),
        "countries": df["country"].nunique() if "country" in df.columns else 0,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📈 Stats saved to {stats_path}")

    return df


if __name__ == "__main__":
    prepare_data()
