import optuna
import pandas as pd
import numpy as np
import sys
import os
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from features import TimeSeriesFeatures
from models import XGBoostForecaster
from evaluation import Evaluator

def objective(trial):
    """
    Optuna Objective Function.
    Returns: Average MAE from Cross-Validation.
    """
    # 1. Suggest Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'early_stopping_rounds': 50,
        # 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    # 2. Prepare Data (Only once ideally, but fast enough here)
    # We rely on cached data
    loader = DataLoader('../data/raw/GlobalWeatherRepository.csv')
    try:
        df = loader.load_data()
    except:
        loader = DataLoader('data/raw/GlobalWeatherRepository.csv', cache_dir='data/cache')
        df = loader.load_data()
        
    fe = TimeSeriesFeatures()
    df_features = fe.transform(df)
    
    target_col = 'temperature_celsius'
    features = [c for c in df_features.columns if c != target_col]
    X = df_features[features]
    y = df_features[target_col]
    
    # 3. Cross-Validation
    # Use fewer splits for speed during tuning (e.g. 3)
    evaluator = Evaluator(output_dir='reports')
    cv_results = evaluator.cross_validate(
        XGBoostForecaster, 
        X, y, 
        param_dict=params, 
        n_splits=3
    )
    
    # Minimize MAE
    avg_mae = cv_results[cv_results['Fold'] == 'Average']['MAE'].values[0]
    return avg_mae

def run_tuning():
    print("=== Starting Hyperparameter Tuning with Optuna ===")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 10 trials for demo speed
    
    print("\nBest Trial:")
    print(study.best_trial.params)
    
    # Save Best Params
    best_params_path = 'models/best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    run_tuning()
