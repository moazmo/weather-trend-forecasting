import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta

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

    def fetch_realtime_history(self, hours_back=168):
        """
        Fetches the last N hours of temperature data from Open-Meteo API for London.
        Used for real-time inference context.
        """
        # London coordinates (approx)
        lat = 51.5074
        lon = -0.1278
        
        # Calculate start/end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back + 24) # Buffer
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m"
        }
        
        try:
            print(f"Fetching live data from Open-Meteo: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse
            hourly = data['hourly']
            df = pd.DataFrame({
                'last_updated': pd.to_datetime(hourly['time']),
                'temperature_celsius': hourly['temperature_2m']
            })
            
            # Set Index
            df = df.set_index('last_updated').sort_index()
            
            # Filter to required window
            return df.tail(hours_back)
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame() # Return empty on failure

if __name__ == "__main__":
    # Quick Test
    loader = DataLoader('../data/raw/GlobalWeatherRepository.csv')
    df = loader.load_data(force_reload=True)
    print(df.head())
    print(df.info())
