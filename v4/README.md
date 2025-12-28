# V4 Weather Backtester

XGBoost-based weather prediction with historical backtesting.

## Model Performance

| Metric | Value |
|--------|-------|
| **Test MAE** | **1.21°C** |
| Test RMSE | 1.68°C |
| Features | 22 |
| Countries | 211 |

## Quick Start

```bash
uvicorn v4.app.main:app --port 8002
```

Open http://localhost:8002

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interactive UI |
| `GET /api/predict` | Run backtest |
| `GET /api/countries` | List countries |
| `GET /api/date-range` | Valid dates |
| `GET /api/feature-importance` | SHAP rankings |

## Structure

```
v4/
├── app/
│   ├── main.py              # FastAPI
│   └── static/index.html    # UI
├── models/
│   ├── xgboost_model.joblib
│   └── scaler.joblib
├── scripts/
│   └── train_xgboost.py
└── src/
    ├── config.py
    ├── inference.py
    └── utils.py
```

---

*PM Accelerator Assessment*
