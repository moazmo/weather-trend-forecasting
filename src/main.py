import pandas as pd
import sys
import os

# Add current directory to path so we can import modules if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from features import TimeSeriesFeatures
from models import XGBoostForecaster
from evaluation import Evaluator

def main():
    print("=== Weather Forecasting Pipeline Started ===")
    
    # 1. Load Data
    print("\n[Step 1] Loading Data...")
    loader = DataLoader(raw_data_path='../data/raw/GlobalWeatherRepository.csv')
    try:
        df = loader.load_data()
    except FileNotFoundError:
        # Fallback for dev environment path
        loader = DataLoader(raw_data_path='data/raw/GlobalWeatherRepository.csv', cache_dir='data/cache')
        df = loader.load_data()
        
    print(f"Loaded {len(df)} hourly records.")
    
    # 2. Feature Engineering
    print("\n[Step 2] Feature Engineering...")
    fe = TimeSeriesFeatures()
    df_features = fe.transform(df)
    print(f"Feature set shape: {df_features.shape}")
    print(f"Features: {df_features.columns.tolist()}")
    
    # Prepare X and y
    target_col = 'temperature_celsius'
    features = [c for c in df_features.columns if c != target_col]
    X = df_features[features]
    y = df_features[target_col]
    
    # 3. Model Evaluation (Cross-Validation)
    print("\n[Step 3] Cross-Validation...")
    evaluator = Evaluator(output_dir='reports')
    
    # We pass the class itself, not an instance, for fresh initialization per fold
    cv_results = evaluator.cross_validate(
        XGBoostForecaster, 
        X, 
        y, 
        param_dict={'n_estimators': 500, 'learning_rate': 0.05},
        n_splits=5
    )
    
    evaluator.save_report(cv_results, 'cv_metrics.csv')
    summary = evaluator.generate_summary(cv_results)
    print(summary)
    
    # 4. Final Training & Forecasting
    print("\n[Step 4] Final Training & Multi-Step Forecasting...")
    # Train on EVERYTHING to predict the future
    final_model = XGBoostForecaster(n_estimators=1000, learning_rate=0.01)
    final_model.fit(X, y)
    
    # Recursive Forecast for next 24 Hours
    # taking the last available data point as input
    last_known_data = X.iloc[-1:] 
    print(f"Forecasting from: {last_known_data.index[0]}")
    
    # Note: predict_next_24h expects full df feature-set logic to grab last row, 
    # but based on my implementation it takes 'current_input' which is FEATURES.
    # Logic in models.py: curr_row = current_input.iloc[-1:].copy()
    # So passing X (which has features) is correct.
    
    forecast_df = final_model.predict_next_24h(X)
    
    print("\n=== Forecast for Next 24 Hours ===")
    print(forecast_df)
    
    # Save Forecast
    forecast_df.to_csv('reports/forecast_next_24h.csv')
    print("\nPipeline Complete. Reports saved to reports/ directory.")

if __name__ == "__main__":
    main()
