# Production-Grade Weather Analysis & Forecasting System

## Overview
This project is a modular, production-ready system for analyzing weather trends, detecting anomalies, and forecasting future conditions using an ensemble of Statistical and Machine Learning models.

## Architecture
```text
.
├── data/
│   ├── raw/            # Original immutable data
│   └── processed/      # Cleaned data ready for modeling
├── notebooks/          # HTML Interactives and SHAP plots
├── src/
│   ├── data_loader.py  # Data ingestion
│   ├── preprocessing.py# Feature engineering
│   ├── models.py       # Forecasting ensembles
│   ├── anomalies.py    # Outlier detection
│   └── visualization.py# Dashboard generators
├── tests/              # Pytest suite
├── requirements.txt
└── README.md           # Project documentation
```

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python main.py --location "London" --data_path "data/GlobalWeatherRepository.csv"
   ```

## Key Features
*   **Ultrathink Preprocessing**: Handles missing values via Time-Series Interpolation per location group. Prevents data leakage between regions.
*   **Ensemble Forecasting**: Combines SARIMA (Seasonality), XGBoost (Non-linear patterns), and Prophet (Holidays/Trends) using validation-RMSE weighted averaging.
*   **Anomaly Detection**: Multivariate Isolation Forest to detect outlier weather events.
*   **Explainable AI**: SHAP integration to explain *why* the model predicts specific temperatures.
