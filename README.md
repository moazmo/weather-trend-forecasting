# 🌍 Weather Trend Forecasting System

> **PM Accelerator Mission**: "By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders."

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.21°C_MAE-green.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive weather analysis and forecasting project for the **PM Accelerator AI Product Management Assessment**.

<table>
<tr>
<td align="center" style="padding: 20px;">

### 📊 View Complete Assessment Presentation

**[📥 Download PMA_presentation.pptx](assessment/PMA_presentation.pptx)**

*EDA, model comparisons, climate analysis, and key insights*

</td>
</tr>
</table>

---

## 🎯 Project Overview

This repository contains two main components:

| Component | Description |
|-----------|-------------|
| **📊 Assessment** | Comprehensive data analysis with 7 Jupyter notebooks |
| **🔮 V4 Backtester** | Interactive web app demonstrating model accuracy |

---

## 📊 Assessment Analysis

Comprehensive analysis of global weather data (114K records, 204 countries).
*Note: Data quality analysis identified 211 raw country entries, which were consolidated to 204 unique countries after resolving typos and standardization.*

### 📓 Notebooks

| Notebook | Description | Key Techniques |
|----------|-------------|----------------|
| [`00_data_quality.ipynb`](assessment/notebooks/00_data_quality.ipynb) | Data validation & cleaning | Missing values, outlier detection |
| [`01_advanced_eda.ipynb`](assessment/notebooks/01_advanced_eda.ipynb) | Exploratory Analysis | Z-score, IQR, Isolation Forest |
| [`02_forecasting_models.ipynb`](assessment/notebooks/02_forecasting_models.ipynb) | Model comparisons | XGBoost, Random Forest, Gradient Boosting |
| [`03_climate_analysis.ipynb`](assessment/notebooks/03_climate_analysis.ipynb) | Climate patterns | Zones, seasonal trends |
| [`04_environmental_impact.ipynb`](assessment/notebooks/04_environmental_impact.ipynb) | Air quality correlation | PM2.5, Ozone, NO₂ |
| [`05_feature_importance.ipynb`](assessment/notebooks/05_feature_importance.ipynb) | Feature analysis | SHAP, Permutation, Correlation |
| [`06_spatial_analysis.ipynb`](assessment/notebooks/06_spatial_analysis.ipynb) | Geographic patterns | Choropleth maps |

### 📈 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **Test MAE** | **1.21°C** ✅ |
| **Countries** | 204 |
| **Features** | 22 (from SHAP analysis) |
| **Records** | 114,203 |
| **Date Range** | May 2024 - Dec 2025 |

---

## 🔮 V4 Weather Backtester

Interactive web application for historical model evaluation.

### Features

- 🗺️ **Interactive Map** - Click to select from 204 countries
- 📊 **Actual vs Predicted** - Visual comparison charts
- 📅 **Date Selection** - Choose dates within dataset (May 2024 - Dec 2025)
- 📈 **MAE Calculation** - Per-prediction accuracy metrics

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run V4 backtester
uvicorn v4.app.main:app --port 8002

# Open browser
# http://localhost:8002
```

### How It Works

1. **Select a country** from dropdown (204 available)
2. **Pick a date range** (within May 2024 - Dec 2025)
3. **Click "Run Backtest"**
4. **View results**: Predicted vs Actual chart + MAE

---

## 📁 Project Structure

```
weather-trend-forecasting/
├── assessment/
│   ├── notebooks/           # 7 analysis notebooks
│   ├── outputs/             # CSV results, figures
│   └── PMA_presentation.pptx
├── v4/
│   ├── app/                 # FastAPI + interactive UI
│   ├── models/              # Trained XGBoost model
│   ├── scripts/             # Training scripts
│   └── src/                 # Core modules
├── data/
│   └── raw/                 # GlobalWeatherRepository.csv
└── README.md
```

---

## 🔧 Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
fastapi>=0.100
plotly>=5.15
```

See full requirements in `requirements.txt`.

---

## 📊 Model Details

### XGBoost Configuration

```python
XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

### Top Features (SHAP Analysis)

1. humidity
2. latitude
3. uv_index
4. pressure_mb
5. longitude
6. air_quality_Carbon_Monoxide
7. month_sin / month_cos (cyclical)

---

## 👤 Author

**Moaz Muhammad**

*Built for the PM Accelerator*
