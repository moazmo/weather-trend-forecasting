# 🌍 Weather Trend Forecasting System

> **PM Accelerator Mission**: "By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills."

> 📊 **[View Interactive Presentation →](https://moazmo.github.io/weather-trend-forecasting/)** | Full interactive charts on GitHub Pages

<table>
<tr>
<td align="center" style="padding: 20px; background-color: #1a73e8;">

### 🎯 Want to see the full presentation with interactive charts?

**[👉 Open on GitHub Pages 👈](https://moazmo.github.io/weather-trend-forecasting/)**

*Explore EDA visualizations, model comparisons, and climate analysis with full interactivity!*

</td>
</tr>
</table>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-enabled-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-purple.svg)](https://dvc.org/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

A production-grade machine learning system for precision global weather forecasting. Powered by an **Advanced Transformer with Gated Residual Networks (GRN)**, featuring full MLOps pipeline with **MLflow**, **DVC**, and **GitHub Actions CI/CD**.

---

## ✨ Key Features

### 🧠 V2 Advanced Transformer (Stable - Recommended)
- **Production-ready** with stable performance
- **2.00°C MAE** - Best tested accuracy
- Clean FastAPI interface

### 🧪 V3 Climate-Aware Transformer (Experimental)
- **929K parameters** with Hybrid Static-Dynamic Architecture + Country Embeddings
- **25 input features** including air quality, geography, and cyclical time encoding
- **14-day input → 7-day forecast** sequence-to-sequence architecture
- ⚠️ **Under active development** - Use V2 for production

### 🔧 Full MLOps Pipeline
- **MLflow** for experiment tracking and model registry
- **DVC** for data versioning and pipeline reproducibility
- **GitHub Actions** for automated CI/CD
- **Docker** containerization with multi-stage builds
- **pytest** with 55+ automated tests

### 🌍 Interactive Web Application
- Click-anywhere interactive **Leaflet map**
- Automatic **climate zone detection**
- Real-time **Plotly** temperature visualizations
- Dark-themed professional UI

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# V2 Stable Application (Recommended)
cd v2
docker-compose up v2-api
# Open http://localhost:8000

# V3 Experimental (Development Only)
cd v3
docker-compose up v3-api
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

# Run V2 Stable Application (Recommended)
uvicorn v2.app.main:app --port 8000
# Open http://localhost:8000

# Run V3 Experimental (Development Only)
uvicorn v3.app.main:app --port 8001
# Open http://localhost:8001
```

---

## 🧠 Model Architecture

### V3 Climate-Aware Transformer

```
Input Sequence (14 days × 25 features)
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

### Performance

| Model | Architecture | MAE | Status |
|-------|-------------|-----|--------|
| **V2** | Advanced Transformer | **2.00°C** | 🟢 **Stable (Recommended)** |
| V3 | Hybrid Climate-Aware Transformer | ~6.8°C | 🟡 Experimental |
| V1 | Basic Transformer | ~4.50°C | ⚪ Retired |

---

## 🔄 CI/CD Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GitHub    │───→│   GitHub    │───→│   MLflow    │───→│   Docker    │
│    Push     │    │   Actions   │    │  Tracking   │    │   Deploy    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │                  │
     ▼                   ▼                  ▼                  ▼
  [Code]            [55 Tests]         [Metrics]          [Container]
```

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `v3-test.yml` | Push/PR | Lint + pytest |
| `v3-train.yml` | Weekly/Manual | MLflow training |
| `v3-pipeline.yml` | Weekly/Manual | Full DVC pipeline |

---

## 📁 Project Structure

```
WeatherTrendForecasting/
├── v3/                      # 🌟 V3 Climate-Aware Transformer
│   ├── app/                 # FastAPI + Interactive UI
│   ├── src/                 # Production modules
│   ├── mlflow/              # MLflow training scripts
│   ├── tests/               # pytest suite (55 tests)
│   ├── scripts/             # DVC pipeline scripts
│   └── notebooks/           # Analysis notebooks
├── v2/                      # V2 Legacy Application
├── .github/workflows/       # CI/CD Pipelines
├── dvc.yaml                 # DVC Pipeline Definition
├── Dockerfile               # Production Container
└── requirements.txt         # Dependencies
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status |
| `GET` | `/api/model-info` | Model metadata |
| `POST` | `/api/forecast` | Generate 7-day forecast |
| `GET` | `/api/climate-zones` | Climate zone definitions |

**Example:**
```bash
curl -X POST "http://localhost:8000/api/forecast" \
     -H "Content-Type: application/json" \
     -d '{"latitude": 30.04, "longitude": 31.23, "start_date": "2024-06-15"}'
```

---

## 🧪 Development

```bash
# Run tests
pytest v3/tests/ -v

# Train with MLflow
python -m v3.mlflow.train --experiment local --epochs 50

# View MLflow UI
mlflow ui --port 5000

# Run DVC pipeline
dvc repro
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
