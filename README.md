# 🌍 Weather Trend Forecasting

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning system for global weather temperature forecasting. Built with PyTorch and FastAPI, this project provides 7-day temperature trend predictions for 180+ countries using a unified neural network model.

![Weather Forecast Demo](docs/demo.png)

---

## ✨ Features

- 🌡️ **7-Day Temperature Forecasting** - Predict temperature trends for any country
- 🧠 **Unified MLP Model** - Single neural network trained on 180+ countries
- 🔧 **Optuna Optimization** - Hyperparameter tuning for optimal performance
- 📊 **Interactive Visualizations** - Plotly-powered charts and analysis
- 🚀 **GPU Accelerated** - CUDA support for fast training
- 🌐 **REST API** - FastAPI backend for easy integration
- 📈 **Anomaly Detection** - Multiple methods (Z-Score, IQR, Isolation Forest, LOF)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 11.8+ (optional, for faster training)

### Installation

```bash
# Clone the repository
git clone https://github.com/moazmo/weather-trend-forecasting.git
cd weather-trend-forecasting

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter notebook notebooks/
```

### Run Web Applications

#### V1 - Country Dropdown (Port 8000)
```bash
uvicorn app.main:app --reload --port 8000
```
Open: http://localhost:8000

#### V2 - Interactive Map (Port 8001)
```bash
uvicorn v2.app.main:app --reload --port 8001
```
Open: http://localhost:8001

| Version | Port | Features |
|---------|------|----------|
| **V1** | 8000 | Country dropdown, unified MLP |
| **V2** | 8001 | Interactive map, location-based model, climate zones |

---

## 📁 Project Structure

```
WeatherTrendForecasting/
├── 📂 notebooks/               # Jupyter notebooks
│   ├── 01_eda_anomaly_detection.ipynb
│   ├── 02_forecasting_models.ipynb
│   ├── 03_advanced_forecasting.ipynb
│   ├── 04_data_quality_analysis.ipynb
│   └── 05_unified_global_model.ipynb
├── 📂 data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data
├── 📂 models/                  # Saved model artifacts
│   ├── global_weather_mlp.pt
│   ├── feature_scaler.joblib
│   ├── country_encoder.joblib
│   └── model_config.json
├── 📂 app/                     # FastAPI web application
├── 📂 reports/                 # Analysis reports
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda_anomaly_detection` | Exploratory data analysis, statistical analysis, anomaly detection |
| `02_forecast_v1_baseline` | Baseline models: ARIMA, XGBoost, Prophet, Ensemble |
| `03_advanced_forecasting` | Time series CV, Optuna tuning, multi-country models |
| `04_data_quality_analysis` | Data quality checks, country name fixes, missing value analysis |
| `05_unified_global_model` | V1 Model: Unified MLP for all countries |
| `v2/notebooks/03_lstm_model` | V2 Model: LSTM with sequence modeling |
| `v2/notebooks/04_transformer_model` | **V2.3 Model**: Transformer with attention mechanism |

---

## 🧠 Model Architecture

### Transformer (V2.3) - Current Production Model
State-of-the-art sequence modeling architecture.

```
Input Sequence (30 Days)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 Layers)
    ├── Multi-Head Self-Attention (8 heads)
    └── Feed Forward Network (256 units)
    ↓
Output Head -> 7-Day Forecast
```

### Previous Models
- **V2.2 LSTM**: 2-layer LSTM with 128 hidden units.
- **V1 MLP**: Simple feed-forward network with 3 dense layers.

### Performance Evolution

| Model | Architecture | MAE (Mean Absolute Error) |
|-------|--------------|---------------------------|
| V1 | MLP | ~4-5°C |
| V2.2 | LSTM | 2.05°C |
| **V2.3** | **Transformer** | **2.05°C (Faster Training)** |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/api/countries` | List available countries |
| GET | `/api/nearest` | Find nearest country to coordinates |
| POST | `/api/forecast` | Get 7-day forecast (Transformer powered) |
| GET | `/api/health` | Health check |

### Example Request (V2)

```bash
curl -X POST "http://localhost:8001/api/forecast" \
  -H "Content-Type: application/json" \
  -d '{"lat": 30.04, "lon": 31.23, "start_date": "2025-01-15"}'
```

---

## 📊 Dataset
- **Source**: Global Weather Repository
- **Records**: 100,000+ observations
- **Countries**: 186 countries
- **Features**: Temperature, Lat/Lon, Climate Zones, Temporal Embeddings
- **Time Range**: ~2 years daily data

---

## 🛠️ Tech Stack
| Category | Technologies |
|----------|-------------|
| **Deep Learning** | **PyTorch** (Transformer, LSTM, MLP) |
| **Backend** | **FastAPI**, Uvicorn |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly, Leaflet.js (Frontend) |
| **Notebooks** | Jupyter |

---

## 🗺️ Roadmap

- [x] EDA & Anomaly Detection
- [x] Baseline Forecasting Models
- [x] Multi-Country Models
- [x] **V1**: Unified Global MLP Model
- [x] **V2**: Location-Based Model with Interactive Map
- [x] **V2.2**: LSTM Sequence Model
- [x] **V2.3**: Transformer Attention Model
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] Cloud Deployment

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohamed Ashraf (MoAzMo)**

- GitHub: [@moazmo](https://github.com/moazmo)

---

## 🙏 Acknowledgments

- Global Weather Repository for the dataset
- PyTorch team for the excellent deep learning framework
- Optuna team for the hyperparameter optimization library
