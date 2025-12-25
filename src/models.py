import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoostForecaster(BaseEstimator, RegressorMixin):
    """
    Wrapper for XGBoost Regressor optimized for Time Series Forecasting.
    Implements a recursive multi-step forecasting strategy.
    """
    
    def __init__(self, n_estimators=1000, learning_rate=0.01, max_depth=6, early_stopping_rounds=50):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.features = None
        self.target = None
        
    def fit(self, X, y, eval_set=None):
        """
        Trains the XGBoost model.
        X: Feature DataFrame
        y: Target Series
        """
        self.features = X.columns.tolist()
        self.target = y.name
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=42
        )
        
        if eval_set:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
            
        return self
        
    def predict(self, X):
        """Standard single-step prediction."""
        return self.model.predict(X)
        
    def predict_next_24h(self, current_input: pd.DataFrame) -> pd.DataFrame:
        """
        Performs recursive forecasting for the next 24 hours.
        Predicts T+1 -> Feeds prediction back into Lag 1 -> Predicts T+2 ...
        
        current_input: DataFrame containing the FEATURES for the *last known timestamp*.
        """
        predictions = []
        timestamps = []
        
        # We need a mutable copy of the input row to update lags
        # Ensure we only take the last row if multiple are passed
        curr_row = current_input.iloc[-1:].copy()
        
        start_time = curr_row.index[0]
        
        for i in range(1, 25): # Predict 24 steps
            # 1. Component Prediction
            pred_val = self.model.predict(curr_row[self.features])[0]
            predictions.append(pred_val)
            
            # Future Timestamp
            future_time = start_time + pd.Timedelta(hours=i)
            timestamps.append(future_time)
            
            # 2. Recursive Feature Update
            next_row = curr_row.copy()
            next_row.index = [future_time]
            
            # Update Lags (Shift logic)
            # lag_1 becomes the value we just predicted
            # lag_2 becomes the old lag_1, etc.
            # Note: This is a simplified shift. For rigorous correctness with many lags, 
            # one should maintain a history buffer. Here we assume standard lags [1,2,3,24...]
            
            # Shift standard lags (Hardcoded for the specific feature set in features.py)
            # CRITICAL FIX: Use 'curr_row' (snapshot of t) to update 'next_row' (t+1)
            # This prevents cascading overwrites where lag_3 becomes lag_1 because lag_2 was just updated.
            if 'lag_1' in curr_row: next_row['lag_2'] = curr_row['lag_1']
            if 'lag_2' in curr_row: next_row['lag_3'] = curr_row['lag_2']
            if 'lag_24' in curr_row: next_row['lag_24'] = curr_row['lag_24'] # 24h lag requires buffer, keeping static for MVP/short-term
            
            # Note: For production 24h lag updates, we need a history buffer of 24 steps window.
            # Current Simplification: We assume deep history (24, 48, 168) changes slowly or we just keep it from last known.
            # For 24h horizon, 'lag_24' at step T+1 should be the value at T-23. 
            # Since we don't have a rolling buffer here, keeping it static is a limitation but prevents the oscillation bug.
            
            next_row['lag_1'] = pred_val
            
            # Update Time Features
            next_row['hour'] = future_time.hour
            next_row['month'] = future_time.month
            next_row['dayofweek'] = future_time.dayofweek
            
            # Update Cyclical Features
            next_row['hour_sin'] = np.sin(2 * np.pi * next_row['hour'] / 24)
            next_row['hour_cos'] = np.cos(2 * np.pi * next_row['hour'] / 24)
            
            # Update Interactions
            if 'hour_x_month' in next_row:
                next_row['hour_x_month'] = next_row['hour'] * next_row['month']
            
            # Move forward
            curr_row = next_row
            
        return pd.DataFrame({'predicted_temperature': predictions}, index=timestamps)

    def get_feature_importance(self):
        """Returns feature importance DataFrame."""
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
