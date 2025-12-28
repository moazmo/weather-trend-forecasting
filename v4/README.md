# V4 Ensemble Weather Forecaster

> **EXPERIMENTAL** - Advanced Ensemble Weather Forecasting

## Overview

V4 combines insights from the assessment analysis to create an ensemble forecaster:

- **XGBoost** (2.94°C MAE from assessment) - Best tree-based model
- **Transformer with GRN** (from V2) - Best sequence model
- **Climate-zone-specific weighting** - Adapts ensemble to location

## Features

- 🗺️ **Interactive Map** - Click anywhere for predictions
- 📊 **Actual vs Predicted** - Historical comparison charts
- 🎯 **Confidence Intervals** - Uncertainty estimation
- 📈 **Feature Importance** - SHAP-based rankings API
- 🌍 **Climate Zone Detection** - Automatic zone classification

## Quick Start

```bash
# Run locally
uvicorn v4.app.main:app --port 8002

# Open browser
# http://localhost:8002
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive web UI |
| GET | `/health` | Health check |
| GET | `/api/model-info` | Ensemble architecture info |
| GET | `/api/countries` | Available countries |
| GET | `/api/nearest` | Find nearest country |
| POST | `/api/forecast` | 7-day ensemble forecast |
| GET | `/api/historical` | Actual vs Predicted comparison |
| GET | `/api/climate-zones` | Zone definitions |
| GET | `/api/feature-importance` | SHAP rankings |

## Assessment Insights Used

### Top Features (SHAP Analysis)
1. humidity
2. latitude
3. uv_index
4. pressure_mb
5. longitude

### Anomaly Bounds (IQR)
- Temperature: 0.7°C - 44.7°C
- Precipitation: 18.8% outliers filtered

### Ensemble Weights by Zone
| Zone | XGBoost | Transformer |
|------|---------|-------------|
| Tropical | 60% | 40% |
| Subtropical | 50% | 50% |
| Temperate | 40% | 60% |
| Subarctic | 50% | 50% |
| Polar | 60% | 40% |

## Directory Structure

```
v4/
├── src/
│   ├── config.py        # Configuration
│   ├── models.py        # GRN, Transformer, XGBoost, Ensemble
│   ├── preprocessing.py # Feature engineering
│   ├── inference.py     # V4Forecaster API
│   └── utils.py         # Helpers
├── app/
│   ├── main.py          # FastAPI application
│   └── static/
│       └── index.html   # Interactive UI
├── models/              # Model artifacts
├── tests/               # pytest suite
└── Dockerfile           # Container config
```

---

*Built for PM Accelerator Assessment*
