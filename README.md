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

### Run Web App (Coming Soon)

```bash
uvicorn app.main:app --reload
```

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
| `02_forecasting_models` | Baseline models: ARIMA, XGBoost, Prophet, Ensemble |
| `03_advanced_forecasting` | Time series CV, Optuna tuning, multi-country models |
| `04_data_quality_analysis` | Data quality checks, country name fixes, missing value analysis |
| `05_unified_global_model` | **Main model**: Unified MLP for all countries |

---

## 🧠 Model Architecture

```
Input (20 features)
    ├── Country encoding
    ├── Geographic (lat, lon)
    ├── Temporal (month, day, week...)
    ├── Cyclical (sin/cos encodings)
    └── Lag features (1, 2, 3, 7, 14, 30 days)
         ↓
MLP Neural Network
    ├── Dense + BatchNorm + ReLU + Dropout
    ├── Dense + BatchNorm + ReLU + Dropout
    └── Dense + BatchNorm + ReLU + Dropout
         ↓
Output: Temperature (°C)
```

### Performance

| Metric | Value |
|--------|-------|
| MAE | ~2-3°C |
| RMSE | ~3-4°C |
| R² | ~0.85+ |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/api/countries` | List available countries |
| POST | `/api/forecast` | Get 7-day forecast |
| GET | `/api/health` | Health check |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/forecast" \
  -H "Content-Type: application/json" \
  -d '{"country": "Egypt", "start_date": "2025-01-15"}'
```

### Example Response

```json
{
  "country": "Egypt",
  "forecast": [
    {"date": "2025-01-15", "temperature": 18.2, "day_name": "Wednesday"},
    {"date": "2025-01-16", "temperature": 17.8, "day_name": "Thursday"},
    ...
  ],
  "summary": {
    "min_temp": 15.9,
    "max_temp": 20.3,
    "trend": "Cooling then Warming"
  }
}
```

---

## 📊 Dataset

- **Source**: Global Weather Repository
- **Records**: 114,000+ observations
- **Countries**: 180+ countries
- **Features**: Temperature, humidity, pressure, wind, precipitation, cloud cover, UV index
- **Time Range**: ~2 years of daily data

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, scikit-learn, XGBoost |
| **Optimization** | Optuna |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **API** | FastAPI, Uvicorn |
| **Deployment** | Docker |

---

## 🗺️ Roadmap

- [x] EDA & Anomaly Detection
- [x] Baseline Forecasting Models
- [x] Multi-Country Models
- [x] Unified Global MLP Model
- [ ] FastAPI Web Application
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] Cloud Deployment
- [ ] Model Monitoring

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
