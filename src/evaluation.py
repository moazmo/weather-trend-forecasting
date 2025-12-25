import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import os

class Evaluator:
    """
    Handles model evaluation, metrics calculation, and reporting.
    Implements Time Series Cross-Validation to ensure robust performance estimates.
    """
    
    def __init__(self, output_dir='../reports'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_metrics(self, y_true, y_pred):
        """Returns dictionary of standard regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = np.nan
            
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
    def cross_validate(self, model_class, X, y, param_dict=None, n_splits=5):
        """
        Performs Time Series Split Cross-Validation.
        
        Args:
            model_class: The class of the model (e.g. XGBoostForecaster) - NOT an instance.
                         We re-instantiate it for every fold to avoid leakage.
            X: Feature DataFrame
            y: Target Series
            param_dict: Arguments to pass to model constructor (e.g. n_estimators)
            n_splits: Number of CV folds
            
        Returns:
            DataFrame of metrics for each fold + Average.
        """
        if param_dict is None:
            param_dict = {}
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []
        
        print(f"Starting Time Series CV with {n_splits} folds...")
        
        fold = 1
        for train_index, test_index in tscv.split(X):
            # Split Data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Initialize & Train FRESH model
            model = model_class(**param_dict)
            model.fit(X_train, y_train)
            
            # Predict
            preds = model.predict(X_test)
            
            # Calculate Metrics
            metrics = self.calculate_metrics(y_test, preds)
            metrics['Fold'] = fold
            fold_results.append(metrics)
            
            print(f"Fold {fold}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
            fold += 1
            
        results_df = pd.DataFrame(fold_results)
        
        # Add Average Row
        avg_metrics = results_df[['MAE', 'RMSE', 'MAPE']].mean().to_dict()
        avg_metrics['Fold'] = 'Average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        return results_df

    def save_report(self, metrics_df, filename='cv_results.csv'):
        """Saves evaluation results to CSV."""
        path = os.path.join(self.output_dir, filename)
        metrics_df.to_csv(path, index=False)
        print(f"Saved evaluation report to {path}")
        
    def generate_summary(self, metrics_df, model_name="XGBoost"):
        """Generates a text summary for business stakeholders."""
        avg_row = metrics_df[metrics_df['Fold'] == 'Average'].iloc[0]
        
        summary = f"""
        === Model Performance Summary: {model_name} ===
        Cross-Validation Strategy: Time Series Split (Robust to Seasonality)
        
        Key Metrics (Average across {len(metrics_df)-1} folds):
        - Mean Absolute Error (MAE): {avg_row['MAE']:.4f} °C
        - Root Mean Squared Error (RMSE): {avg_row['RMSE']:.4f} °C
        
        Interpretation:
        On average, the model's predictions are off by about {avg_row['MAE']:.2f} degrees Celsius.
        """
        
        with open(os.path.join(self.output_dir, 'executive_summary.txt'), 'w') as f:
            f.write(summary)
            
        return summary
