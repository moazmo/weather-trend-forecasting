# 🌍 Weather Trend Forecasting - Advanced Assessment

> **PM Accelerator Mission**: "By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills."

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Data Cleaning Methodology](#-data-cleaning-methodology)
4. [Advanced EDA & Anomaly Detection](#-advanced-eda--anomaly-detection)
5. [Forecasting Models](#-forecasting-models)
6. [Advanced Analyses](#-advanced-analyses)
7. [Key Insights](#-key-insights)
8. [How to Run](#-how-to-run)
9. [Dependencies](#-dependencies)

---

## 📊 Project Overview

This assessment provides a comprehensive analysis of global weather data, implementing advanced machine learning techniques for temperature forecasting and in-depth analyses of climate, environmental, and spatial patterns.

### Key Deliverables

| Notebook | Description |
|----------|-------------|
| `01_advanced_eda.ipynb` | Anomaly detection using Z-score, IQR, Isolation Forest, LOF |
| `02_forecasting_models.ipynb` | SARIMA, Prophet, XGBoost, RF, LSTM, Transformer + Ensemble |
| `03_climate_analysis.ipynb` | Climate zones, seasonal patterns, regional trends |
| `04_environmental_impact.ipynb` | Air quality correlation with weather parameters |
| `05_feature_importance.ipynb` | Correlation, RF, XGBoost, Permutation, SHAP analysis |
| `06_spatial_analysis.ipynb` | Geographical patterns, choropleth maps, continental analysis |

---

## 📁 Dataset Description

### GlobalWeatherRepository.csv

- **Size**: ~30 MB
- **Records**: 200,000+ observations
- **Countries**: 140+ countries worldwide
- **Time Range**: Multiple years of daily weather data

### Key Features

| Category | Features |
|----------|----------|
| **Temperature** | `temperature_celsius` (target variable) |
| **Atmospheric** | `humidity`, `pressure_mb`, `cloud`, `visibility_km` |
| **Wind** | `wind_kph`, `wind_degree`, `gust_kph` |
| **Precipitation** | `precip_mm`, `uv_index` |
| **Air Quality** | `PM2.5`, `PM10`, `Ozone`, `NO2`, `CO`, `SO2` |
| **Geographic** | `latitude`, `longitude`, `country` |

---

## 🧹 Data Cleaning Methodology

### 1. Missing Value Handling
- **Numeric columns**: Time-series interpolation for temporal continuity
- **Categorical columns**: Forward/backward fill for consistency
- **Air quality metrics**: Mean imputation where sparse

### 2. Data Quality Checks
- Removed duplicate records based on country + timestamp
- Standardized country names for consistency
- Validated temperature ranges (-60°C to 60°C)
- Checked coordinate validity (lat: -90 to 90, lon: -180 to 180)

### 3. Feature Engineering
- **Lag Features**: `temp_lag_1`, `temp_lag_7`, `temp_lag_14`, etc.
- **Rolling Statistics**: 7-day and 14-day moving averages and std
- **Cyclical Encoding**: `month_sin`, `month_cos`, `day_sin`, `day_cos`
- **Derived Features**: `climate_zone`, `hemisphere`, `abs_latitude`

---

## 🔍 Advanced EDA & Anomaly Detection

### Methods Implemented

| Method | Description | Typical Outlier % |
|--------|-------------|-------------------|
| **Z-Score** | Statistical outliers > 3σ | 1-2% |
| **IQR** | 1.5 × Interquartile Range | 3-5% |
| **Isolation Forest** | ML ensemble isolation | 5% (configurable) |
| **Local Outlier Factor** | Density-based detection | 5% (configurable) |

### Key Findings
- Temperature anomalies concentrated in continental regions
- Air quality outliers correlate with industrial areas
- Seasonal patterns affect anomaly distribution

---

## 📈 Forecasting Models

### Models Compared

| Model | Type | Strengths |
|-------|------|-----------|
| **SARIMA** | Statistical | Interpretable, handles seasonality |
| **Prophet** | Additive | Robust to missing data, trend detection |
| **XGBoost** | Gradient Boosting | Handles non-linear relationships |
| **Random Forest** | Ensemble Trees | Feature importance, low overfitting |
| **LSTM** | Deep Learning | Sequence patterns, long-term dependencies |
| **Transformer** | Deep Learning | Attention mechanism, parallel processing |

### Ensemble Approach
- **Simple Average**: Equal weight combination
- **Weighted Average**: Inverse MAE weighting
- **Improvement**: Typically 10-15% MAE reduction vs best single model

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Primary metric
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R² Score**: Variance explained
- **MAPE** (Mean Absolute Percentage Error): Relative accuracy

---

## 🔬 Advanced Analyses

### 1. Climate Analysis
- **Climate Zones**: Tropical, Subtropical, Temperate, Subarctic, Polar
- **Seasonal Patterns**: Hemisphere-specific temperature cycles
- **Trend Analysis**: Long-term climate variations
- **Amplitude Study**: Countries with highest seasonal variation

### 2. Environmental Impact
- **Air Quality Metrics**: PM2.5, Ozone, NO2, CO, SO2 distributions
- **Weather Correlation**: Wind speed reduces pollution levels
- **Regional Patterns**: Urban vs rural pollution differences
- **Seasonal Effects**: Higher pollution in winter months

### 3. Feature Importance
- **Correlation**: Basic linear relationship measurement
- **Model-Based**: Random Forest and XGBoost importance
- **Permutation**: Model-agnostic importance
- **SHAP Values**: Game-theoretic feature attribution

### 4. Spatial Analysis
- **Global Maps**: Temperature distribution visualization
- **Choropleth Maps**: Country-level aggregations
- **Latitude Analysis**: Temperature-latitude relationship
- **Continental Patterns**: Regional climate comparisons

---

## 💡 Key Insights

1. **Model Performance**
   - Ensemble methods consistently outperform individual models
   - XGBoost and Random Forest show best single-model accuracy
   - Deep learning requires more data for optimal performance

2. **Feature Importance**
   - Geographic features (latitude, longitude) are most predictive
   - Temporal features capture seasonal patterns effectively
   - Air quality has moderate predictive value

3. **Climate Patterns**
   - Tropical regions show most stable temperatures
   - Temperate zones have highest seasonal variability
   - Coastal regions have lower temperature extremes

4. **Environmental Insights**
   - Strong negative correlation: wind speed ↔ pollution
   - Seasonal pollution peaks in winter months
   - Temperature inversely affects some pollutants

---

## 🚀 How to Run

### Prerequisites
```bash
# Navigate to assessment folder
cd f:\WeatherTrendForecasting\assessment

# Install additional dependencies
pip install -r requirements.txt
```

### Running Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or run individual notebooks
jupyter nbconvert --execute notebooks/01_advanced_eda.ipynb --to html
```

### Generate Presentation
```bash
python scripts/create_presentation.py
# Creates outputs/presentation.pptx
```

---

## 📦 Dependencies

### Core Dependencies (from main project)
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
prophet>=1.1.0
statsmodels>=0.14.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Additional for Assessment
```
python-pptx>=0.6.21  # Presentation generation
shap>=0.42.0         # SHAP analysis
kaleido>=0.2.1       # Static image export
```

---

## 📂 Folder Structure

```
assessment/
├── notebooks/
│   ├── 01_advanced_eda.ipynb
│   ├── 02_forecasting_models.ipynb
│   ├── 03_climate_analysis.ipynb
│   ├── 04_environmental_impact.ipynb
│   ├── 05_feature_importance.ipynb
│   └── 06_spatial_analysis.ipynb
├── scripts/
│   └── create_presentation.py
├── outputs/
│   ├── figures/
│   └── presentation.pptx
├── requirements.txt
└── README.md
```

---

## 📝 License

This project is part of the PM Accelerator assessment program.

---

*Created with ❤️ for the PM Accelerator Advanced Assessment*
