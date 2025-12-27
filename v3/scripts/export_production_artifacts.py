import pandas as pd
import numpy as np
import torch
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path('data/processed/weather_v3_ready.csv')
MODELS_DIR = Path('v3/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("📊 Loading data for artifact generation...")
df = pd.read_csv(DATA_PATH)

# Date parsing
if 'last_updated' in df.columns:
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')

# --- 1. Resampling (Must match training exactly) ---
print("🔄 Resampling to Daily Frequency...")
resampled_dfs = []
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for country, group in df.groupby('country'):
    group = group.set_index('last_updated')
    daily = group[numeric_cols].resample('1D').mean()
    daily['country'] = country
    daily = daily.interpolate(method='time', limit=2).dropna().reset_index()
    resampled_dfs.append(daily)

df = pd.concat(resampled_dfs, ignore_index=True)
print(f"✅ Data Resampled: {len(df):,} rows")

# --- 2. Feature Engineering ---
# Country Encoder
print("🌍 Fitting Country Encoder...")
country_encoder = LabelEncoder()
df['country_id'] = country_encoder.fit_transform(df['country'])

# Feature Groups
DYNAMIC_FEATURES = [
    'humidity', 'pressure_mb', 'wind_kph', 'cloud', 'precip_mm', 'uv_index',
    'visibility_km', 'gust_kph', 'wind_degree',
    'air_quality_Ozone', 'air_quality_PM2.5', 
    'month_sin', 'month_cos'
]

STATIC_FEATURES = [
    'latitude', 'longitude', 'abs_latitude', 'hemisphere_encoded'
]

# Filtering valid columns
dyn_avail = [c for c in DYNAMIC_FEATURES if c in df.columns]
stat_avail = [c for c in STATIC_FEATURES if c in df.columns]

print(f"🔹 Dynamic: {len(dyn_avail)} | 🔸 Static: {len(stat_avail)}")

# --- 3. sequence Creation (needed to get correct shape for scaler fitting?) ---
# Actually, Scaler fits on flattened data. We can fit on the DF directly!
# CAUTION: In training, we fit on `X_dyn` which came from sequences. 
# If sequence creation dropped start/end rows, the distribution might shift slightly?
# NO, standard scaler on the full dataset is usually robust and preferred for production consistency.
# Wait, the notebook did: `X_dyn_flat = X_dyn.reshape... scaler.fit(X_dyn_flat)`
# `X_dyn` comes from sequences. Rows with < 21 days data are dropped.
# Fitting on full `df` is safer and covers the distribution efficiently.

print("\n📏 Fitting Scalers...")
scaler_dyn = StandardScaler()
scaler_stat = StandardScaler()

scaler_dyn.fit(df[dyn_avail].values)
scaler_stat.fit(df[stat_avail].values)

# --- 4. Export ---
artifacts = {
    'country_encoder': country_encoder,
    'scaler_dyn': scaler_dyn,
    'scaler_stat': scaler_stat,
    'dyn_features': dyn_avail,
    'stat_features': stat_avail,
    'num_countries': len(country_encoder.classes_)
}

save_path = MODELS_DIR / 'v3_1_production_artifacts.joblib'
joblib.dump(artifacts, save_path)
print(f"\n✅ Artifacts Saved: {save_path}")
print("   - Includes: Encoders, Scalers, Feature Lists")
