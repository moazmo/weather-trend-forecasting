import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    """
    Handles loading, cleaning, and caching of weather data.
    Implements the 'Optimized Data Pipeline' by using caching to speed up
    iterations and strict forward-fill limits to ensure data integrity.
    """
    
    def __init__(self, raw_data_path: str, cache_dir: str = '../data/cache'):
        self.raw_data_path = raw_data_path
        self.cache_dir = cache_dir
        self.location = 'London'
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Loads data from cache if available, otherwise processes raw CSV.
        """
        cache_file = os.path.join(self.cache_dir, f'processed_{self.location}.pkl')
        
        if not force_reload and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            return pd.read_pickle(cache_file)
            
        print("Processing raw data...")
        df = self._process_raw_data()
        
        # Save to cache
        print(f"Caching data to {cache_file}...")
        df.to_pickle(cache_file)
        
        return df

    def _process_raw_data(self) -> pd.DataFrame:
        """
        Reads CSV, filters by location, handles missing dates, and strictly cleans anomalies.
        """
        # 1. Load Raw
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
            
        df = pd.read_csv(self.raw_data_path)
        
        # 2. Basic Cleaning
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        
        # Filter for Target Location
        df_city = df[df['location_name'] == self.location].copy()
        
        # Sort and Set Index
        df_city = df_city.sort_values('last_updated').set_index('last_updated')
        
        # 3. Resample to Hourly Frequency
        # This inserts missing rows for gaps in time
        df_hourly = df_city[['temperature_celsius']].resample('H').mean()
        
        # 4. Handle Missing Values (Robust Strategy)
        # Instead of infinite interpolate(), we limit Forward Fill to 3 hours.
        # This prevents the model from learning "flat lines" during long outages.
        df_clean = df_hourly.ffill(limit=3)
        
        # If there are still NaNs after ffill(limit=3), drop them.
        # We prefer shorter, correct data over long, fake data.
        initial_len = len(df_hourly)
        df_clean = df_clean.dropna()
        final_len = len(df_clean)
        
        if initial_len != final_len:
            print(f"Dropped {initial_len - final_len} rows due to extensive missing data gaps (>3 hours).")
            
        return df_clean

if __name__ == "__main__":
    # Quick Test
    loader = DataLoader('../data/raw/GlobalWeatherRepository.csv')
    df = loader.load_data(force_reload=True)
    print(df.head())
    print(df.info())
