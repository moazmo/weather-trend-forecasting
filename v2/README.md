# Weather Trend Forecasting V2

## Interactive Map-Based Weather Forecasting

V2 introduces an interactive world map for location selection instead of a country dropdown.

### Features

- 🗺️ **Interactive Map** - Click anywhere to select location
- 📍 **Precise Coordinates** - Uses exact lat/lon (not country average)
- 🔍 **Nearest Country** - Finds closest country for prediction
- 📊 **7-Day Forecast** - Temperature trend with visualization

### Run

```bash
cd WeatherTrendForecasting
uvicorn v2.app.main:app --reload --port 8001
```

Open: http://localhost:8001

### How It Works

1. Click on map → get lat/lon
2. Find nearest country (Haversine distance)
3. Use V1 model with exact coordinates
4. Display forecast

### Future Enhancements

- Elevation-based adjustments
- Climate zone classification
- Location-specific model training
