import pandas as pd
import numpy as np

class TimeSeriesFeatures:
    """
    Handles all feature engineering for the Weather Forecasting pipeline.
    Transforms raw time-series data into a supervised learning dataset for XGBoost.
    """
    
    def __init__(self, target_col='temperature_celsius'):
        self.target_col = target_col
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all feature engineering steps to the dataframe.
        """
        df = df.copy()
        
        # 1. Basic Time Features
        df = self._add_time_identifiers(df)
        
        # 2. Cyclical Encoding (The "Clock" logic)
        df = self._add_cyclical_features(df)
        
        # 3. Lag Features (Autocorrelation)
        df = self._add_lags(df)
        
        # 4. Rolling Window Statistics (Trend/Volatility)
        df = self._add_rolling_stats(df)
        
        # 5. Interaction Features (Seasonality x Daily Cycle)
        df = self._add_interactions(df)
        
        # Drop rows with NaNs created by shifting/rolling
        initial_len = len(df)
        df = df.dropna()
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            print(f"Feature Engineering: Dropped {dropped_rows} rows due to lags/rolling windows.")
            
        return df

    def _add_time_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['dayofweek'] = df.index.dayofweek
        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes cyclical time features (hour, month) into Sin/Cos pairs.
        This allows the model to understand that 23:00 is close to 00:00.
        """
        # Hour (24h cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Month (12m cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lagged values of the target variable.
        Lags chosen: 
        - 1, 2, 3 hours (Immediate past)
        - 24, 48 hours (Daily seasonality)
        - 168 hours (Weekly seasonality)
        """
        lags = [1, 2, 3, 24, 48, 168]
        for lag in lags:
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds rolling mean and standard deviation to capture recent trends and volatility.
        Window: 24 hours
        """
        window = 24
        df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[self.target_col].rolling(window=window).std()
        return df
        
    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds interaction terms to capture complex relationships.
        Example: Hour * Month helps differentiate "Winter Mornings" from "Summer Mornings".
        """
        df['hour_x_month'] = df['hour'] * df['month']
        return df

if __name__ == "__main__":
    # Test Block
    # Create dummy data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    dummy_df = pd.DataFrame({'temperature_celsius': np.random.randn(200).cumsum()}, index=dates)
    
    fe = TimeSeriesFeatures()
    df_transformed = fe.transform(dummy_df)
    
    print("Transformed Data Shape:", df_transformed.shape)
    print("Columns:", df_transformed.columns.tolist())
