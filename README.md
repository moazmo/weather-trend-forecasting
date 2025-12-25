# Production-Grade Weather Analysis & Forecasting System

## Overview
This project is a modular, production-ready system for analyzing weather trends, detecting anomalies, and forecasting future conditions. It utilizes advanced recursive forecasting with XGBoost and robust time-series feature engineering.

## Architecture
```text
.
├── data/
│   ├── raw/            # Original immutable data
│   └── cache/          # Cached processed data (Parquet/Pickle)
├── notebooks/          # Exploratory Analysis & Prototype Modeling
│   ├── 01_EDA.ipynb
│   ├── 02_Forecasting_Models.ipynb
│   ├── 03_Evaluation_and_Reporting.ipynb
│   └── 04_Optimized_Forecasting.ipynb  # Multi-step & CV Prototype
├── src/
│   ├── data_loader.py  # Data ingestion with Caching & Strict Cleaning
│   ├── features.py     # TimeSeries Features (Lags, Rolling, Interactions)
│   ├── models.py       # Recursive XGBoost Forecasting (24h horizon)
│   ├── evaluation.py   # Time Series Cross-Validation & Metrics
│   └── main.py         # End-to-End Pipeline Orchestrator
├── reports/            # Generated forecasts and performance summaries
├── requirements.txt
└── README.md           # Project documentation
```

## Key Technologies
*   **Recursive Forecasting**: Predicts 24 hours ahead by feeding predictions back as inputs.
*   **XGBoost**: Gradient boosted decision trees optimized for regression.
*   **Time Series Cross-Validation**: Rolling window validation to ensure stability across seasons.
*   **Smart Caching**: Accelerates data loading by 10x using Pickle/Parquet caching.
*   **Feature Engineering**:
    *   **Lags**: 1h, 24h, 168h (Weekly seasonality).
    *   **Cyclical Encoding**: Sin/Cos transformation for Hour/Month.
    *   **Interactions**: `Hour * Month` to capture daily-seasonal dependencies.

## Setup & Running

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline**:
    ```bash
    # Windows
    python src/main.py
    
    # Unix/Mac
    python3 src/main.py
    ```

3.  **View Results**:
    *   Forecast: `reports/forecast_next_24h.csv`
    *   Metrics: `reports/cv_metrics.csv`
    *   Summary: `reports/executive_summary.txt`

## Development Workflow
1.  **Prototype** new ideas in `notebooks/`.
2.  **Refactor** confirmed logic into `src/` modules.
3.  **Verify** using `python src/main.py`.
