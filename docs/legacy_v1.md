# Legacy V1 Architecture - Weather Trend Forecasting

> **Note**: This documentation is for the legacy V1 system. For the current production system (V4 Advanced Transformer), please refer to the main [README.md](../README.md).

## Overview
The V1 system was the initial proof-of-concept for global weather forecasting. It utilized a unified Multi-Layer Perceptron (MLP) neural network trained on data from 186 countries.

## Architecture
- **Model**: Simple Feed-Forward Neural Network (MLP)
- **Input**: 20 features (cyclical time features, lag features, geographic encoding)
- **Output**: 7-day temperature forecast
- **Framework**: PyTorch

## Performance
- **MAE**: ~4.0 - 5.0°C
- **Limitation**: While it could capture broad global trends, it lacked the sequence modeling capabilities needed for high-precision local forecasting.

## Web Application (V1)
- **Port**: 8000
- **Interface**: Simple dropdown menu to select a country.
- **Backend**: FastAPI

### Running V1
```bash
# Local
uvicorn app.main:app --reload --port 8000

# Docker
docker compose --profile v1 up
```
