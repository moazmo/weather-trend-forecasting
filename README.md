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

A production-grade machine learning system for precision global weather forecasting. Powered by an **Advanced Transformer with Gated Residual Networks (GRN)** achieving **2.00°C MAE**.

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

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up v2-api
# Open http://localhost:8000
```

### Option 2: Local Installation

```bash
# Clone & Install
git clone https://github.com/moazmo/weather-trend-forecasting.git
cd weather-trend-forecasting
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run V2 Application
uvicorn v2.app.main:app --port 8000
# Open http://localhost:8000
```

---

## 🧠 Model Architecture

### V2 Advanced Transformer + GRN

```
Input Sequence (14 days × 11 features)
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

### Model Versions

| Model | Architecture | MAE | Status |
|-------|-------------|-----|--------|
| **V2** | Advanced Transformer + GRN | **2.00°C** | 🟢 **Production** |
| V3 | Hybrid Static-Dynamic Transformer | ~6.8°C | 🟡 Experimental |
| V1 | Basic Transformer | ~4.50°C | ⚪ Retired |

---

## 📁 Project Structure

```
WeatherTrendForecasting/
├── v2/                      # 🌟 V2 Production Application
│   ├── app/                 # FastAPI + Interactive UI
│   │   ├── main.py          # API endpoints & model
│   │   └── templates/       # HTML frontend
│   ├── models/              # Trained model checkpoints
│   └── notebooks/           # Analysis notebooks
├── v3/                      # 🧪 V3 Experimental (Development)
│   ├── app/                 # Next-gen features
│   ├── src/                 # Hybrid architecture
│   └── notebooks/           # Research experiments
├── data/                    # Weather datasets
├── .github/workflows/       # CI/CD Pipelines
└── requirements.txt         # Dependencies
```

---

## 🌐 API Reference (V2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive web UI |
| `GET` | `/api/countries` | List available countries |
| `GET` | `/api/nearest?lat=X&lon=Y` | Find nearest country |
| `POST` | `/api/forecast` | Generate 7-day forecast |

**Example:**
```bash
curl -X POST "http://localhost:8000/api/forecast" \
     -H "Content-Type: application/json" \
     -d '{"lat": 30.04, "lon": 31.23, "start_date": "2024-06-15"}'
```

---

## 🧪 V3 Experimental

> ⚠️ **Note**: V3 is under active development and not recommended for production use.

V3 features a hybrid static-dynamic architecture with country embeddings, but is still being tested. To try it:

```bash
uvicorn v3.app.main:app --port 8001
# Open http://localhost:8001
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
