# 🌍 Weather Trend Forecasting System

> **PM Accelerator Mission**: "By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills."

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A production-grade machine learning system for precision global weather forecasting. Powered by an **Advanced Transformer with Gated Residual Networks (GRN)** achieving **2.00°C MAE** across 186 countries.

<table>
<tr>
<td align="center" style="padding: 20px;">

### 📊 View Complete Assessment Presentation

**[📥 Download PMA_presentation.pptx](assessment/PMA_presentation.pptx)**

*Comprehensive PowerPoint covering EDA, model evolution, climate analysis, and key insights*

</td>
</tr>
</table>

---

## 📊 Assessment Overview

This project provides a comprehensive analysis and forecasting solution for global weather data, implementing advanced machine learning techniques from exploratory data analysis through production deployment.

### 📓 Assessment Notebooks

| Notebook | Description | Key Techniques |
|----------|-------------|----------------|
| [`00_data_quality.ipynb`](assessment/notebooks/00_data_quality.ipynb) | Data validation & cleaning | Missing value analysis, outlier detection |
| [`01_advanced_eda.ipynb`](assessment/notebooks/01_advanced_eda.ipynb) | Exploratory Data Analysis | Z-score, IQR, Isolation Forest, LOF anomaly detection |
| [`02_forecasting_models.ipynb`](assessment/notebooks/02_forecasting_models.ipynb) | Model comparisons | SARIMA, Prophet, XGBoost, Random Forest, LSTM, Transformer |
| [`03_climate_analysis.ipynb`](assessment/notebooks/03_climate_analysis.ipynb) | Climate patterns | Zones, seasonal trends, regional analysis |
| [`04_environmental_impact.ipynb`](assessment/notebooks/04_environmental_impact.ipynb) | Air quality correlation | PM2.5, Ozone, NO₂ vs weather parameters |
| [`05_feature_importance.ipynb`](assessment/notebooks/05_feature_importance.ipynb) | Feature analysis | Correlation, RF, XGBoost, Permutation, SHAP |
| [`06_spatial_analysis.ipynb`](assessment/notebooks/06_spatial_analysis.ipynb) | Geographic patterns | Choropleth maps, latitude analysis |

### 📈 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Advanced Transformer + GRN |
| **MAE** | **2.00°C** ✅ |
| **Countries** | 186 |
| **R² Score** | 0.95 |

---

## ✨ Key Features

### 🧠 V2 Advanced Transformer (Production)
- **Production-ready** and fully tested
- **2.00°C MAE** - Best accuracy in the system
- **929K parameters** with Gated Residual Networks
- Real-time weather data integration via **Open-Meteo API**
- Interactive map-based location selection
- 7-day temperature forecasting

### 🌍 Interactive Web Application
- Click-anywhere interactive **Leaflet map**
- Automatic **nearest country detection**
- Real-time **Plotly** temperature visualizations
- Historical forecast comparison (Predicted vs Actual)
- Dark-themed professional UI

### 🔧 Full MLOps Pipeline
- **MLflow** for experiment tracking
- **DVC** for data versioning
- **GitHub Actions** for CI/CD
- **Docker** containerization
- **55+ unit tests** with pytest

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up weather-api
# Open http://localhost:8001
```

### Option 2: Local Installation

```bash
# Clone & Install
git clone https://github.com/moazmo/weather-trend-forecasting.git
cd weather-trend-forecasting
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run V2 Application (Production)
uvicorn v2.app.main:app --port 8001
# Open http://localhost:8001
```

### Run Assessment Notebooks

```bash
cd assessment
pip install -r requirements.txt
jupyter notebook
```

---

## 🧠 Model Architecture

### V2 Advanced Transformer + GRN

```
Input Sequence (14 days × 23 features)
        ↓
┌─────────────────────────────────┐
│  Gated Residual Network (GRN)  │  ← Variable Selection
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Positional Embedding          │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  Transformer Encoder (×4)       │  ← 8-Head Self-Attention
│  + GRN Layers                   │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│   Output Projection             │
└─────────────────────────────────┘
        ↓
7-Day Temperature Forecast
```

### Model Evolution

| Model | Architecture | MAE | Status |
|-------|-------------|-----|--------|
| **V2** | Advanced Transformer + GRN | **2.00°C** | 🟢 **Production** |
| V3 | Hybrid Static-Dynamic Transformer | ~6.8°C | 🟡 Experimental |
| V1 | Basic MLP | ~4.50°C | ⚪ Retired |

---

## 📁 Project Structure

```
WeatherTrendForecasting/
├── assessment/              # 📊 PM Accelerator Assessment
│   ├── notebooks/           # Analysis notebooks (7 notebooks)
│   ├── outputs/             # Results & visualizations
│   ├── scripts/             # Presentation generation
│   └── README.md            # Assessment documentation
├── v2/                      # 🌟 V2 Production Application
│   ├── app/                 # FastAPI + Interactive UI
│   │   ├── main.py          # API endpoints & model
│   │   └── templates/       # HTML frontend
│   ├── models/              # Trained model checkpoints
│   └── notebooks/           # Model development
├── v3/                      # 🧪 V3 Experimental
│   ├── app/                 # Next-gen features
│   ├── src/                 # Production modules
│   ├── tests/               # pytest suite (55+ tests)
│   └── scripts/             # DVC pipeline scripts
├── notebooks/               # Main analysis notebooks
├── data/                    # Weather datasets
├── models/                  # V1 model artifacts
├── reports/                 # Analysis results
├── .github/workflows/       # CI/CD Pipelines
├── dvc.yaml                 # DVC Pipeline Definition
└── requirements.txt         # Dependencies
```

---

## 🌐 API Reference (V2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive web UI with map |
| `GET` | `/api/countries` | List available countries with coordinates |
| `GET` | `/api/nearest?lat=X&lon=Y` | Find nearest country to coordinates |
| `POST` | `/api/forecast` | Generate 7-day forecast |
| `GET` | `/api/health` | Model status & metrics |

**Example:**
```bash
curl -X POST "http://localhost:8001/api/forecast" \
     -H "Content-Type: application/json" \
     -d '{"lat": 30.04, "lon": 31.23, "start_date": "2024-06-15"}'
```

---

## 🔬 Assessment Deliverables

### Analysis Techniques
- **Anomaly Detection**: Z-Score, IQR, Isolation Forest, Local Outlier Factor
- **Forecasting Models**: SARIMA, Prophet, XGBoost, Random Forest, LSTM, Transformer
- **Feature Importance**: Correlation, Model-based, Permutation, SHAP values
- **Spatial Analysis**: Choropleth maps, latitude-temperature relationships

### Outputs
- `assessment/outputs/model_comparison.csv` - Model performance metrics
- `assessment/outputs/spatial_summary.csv` - Geographic analysis results
- `assessment/outputs/figures/` - Visualization exports
- `assessment/PMA_presentation.pptx` - PowerPoint presentation

---

## 🧪 V3 Experimental

> ⚠️ **Note**: V3 is under active development and not recommended for production use.

V3 features a hybrid static-dynamic architecture with country embeddings:

```bash
uvicorn v3.app.main:app --port 8000
# Open http://localhost:8000
```

---

## 🔄 MLOps Pipeline

### DVC Pipeline
```bash
dvc repro  # Run full pipeline
dvc dag    # View pipeline DAG
```

### GitHub Actions
| Workflow | Description |
|----------|-------------|
| `v3-test.yml` | Lint + pytest on push |
| `v3-train.yml` | Scheduled model training |
| `v3-pipeline.yml` | Full DVC pipeline |

### Run Tests
```bash
pytest v3/tests/ -v --cov=v3/src
```

---

## 📦 Dependencies

### Core
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
```

### Assessment-Specific
```
python-pptx>=0.6.21  # Presentation generation
shap>=0.42.0         # SHAP analysis
kaleido>=0.2.1       # Static image export
```

Explore all the requirements in requirements.txt files.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Moaz Muhammad**

*Built for the PM Accelerator*
