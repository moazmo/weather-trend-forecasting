from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
SEQ_LEN = 14
PRED_LEN = 7
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path("data/processed/weather_v3_ready.csv")
MODELS_DIR = Path("v3/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Device: {DEVICE}")
print(f"✅ Data: {DATA_PATH}")


# --- MODEL ARCHITECTURE ---
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        residual = self.skip(x) if self.skip else x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.layer_norm(out + residual)


class V3ClimateTransformer(nn.Module):
    def __init__(
        self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.2, seq_len=14, pred_len=7
    ):
        super().__init__()
        self.input_grn = GatedResidualNetwork(input_dim, d_model * 2, d_model, dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )

    def forward(self, x):
        x = self.input_grn(x) + self.pos_encoder
        x = self.transformer(x)
        return self.output_head(self.output_grn(x[:, -1, :]))


# --- DATA PREPARATION ---
def load_and_resample_data():
    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Parse datetime
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    elif "last_updated_epoch" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated_epoch"], unit="s")

    # Sort
    df = df.sort_values(["country", "last_updated"]).reset_index(drop=True)

    print(f"📊 Original Rows: {len(df):,}")
    print("🔄 Resampling to Daily Frequency (Mean)...")

    resampled_dfs = []
    # Identify numeric columns for resampling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for country, group in df.groupby("country"):
        group = group.set_index("last_updated")
        # Resample numeric cols
        daily = group[numeric_cols].resample("1D").mean()

        # Restore country col
        daily["country"] = country

        # Interpolate small gaps (up to 2 days)
        daily = daily.interpolate(method="time", limit=2)

        # Drop remaining NaNs
        daily = daily.dropna()

        daily = daily.reset_index()
        if len(daily) > SEQ_LEN + PRED_LEN:
            resampled_dfs.append(daily)

    df_resampled = pd.concat(resampled_dfs, ignore_index=True)
    print(f"✅ Resampled Rows: {len(df_resampled):,} (strictly daily)")
    return df_resampled


def create_sequences(df, feature_cols, target_col):
    all_X, all_y = [], []

    print("🔄 Creating sequences...")
    for country, country_df in df.groupby("country"):
        country_df = country_df.sort_values("last_updated")

        if len(country_df) < SEQ_LEN + PRED_LEN:
            continue

        X = country_df[feature_cols].values.astype(np.float32)
        y = country_df[target_col].values.astype(np.float32)

        for i in range(len(X) - SEQ_LEN - PRED_LEN + 1):
            all_X.append(X[i : i + SEQ_LEN])
            all_y.append(y[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])

    return np.array(all_X), np.array(all_y)


# --- MAIN ---
def main():
    # 1. Prepare Data
    df = load_and_resample_data()

    FEATURE_COLS = [
        "latitude",
        "longitude",
        "abs_latitude",
        "latitude_normalized",
        "hemisphere_encoded",
        "humidity",
        "pressure_mb",
        "wind_kph",
        "cloud",
        "precip_mm",
        "uv_index",
        "visibility_km",
        "gust_kph",
        "wind_degree",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_PM2.5",
        "air_quality_Carbon_Monoxide",
        "air_quality_Sulphur_dioxide",
        "month_sin",
        "month_cos",
        "day_year_sin",
        "day_year_cos",
        "hour_sin",
        "hour_cos",
    ]

    # Validation: Ensure columns exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"✅ Using {len(available_features)} features")

    X_all, y_all = create_sequences(df, available_features, "temperature_celsius")
    print(f"📊 Total Sequences: {len(X_all):,}")

    # Split
    train_size = int(len(X_all) * 0.70)
    val_size = int(len(X_all) * 0.15)

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_val = X_all[train_size : train_size + val_size]
    y_val = y_all[train_size : train_size + val_size]
    X_test = X_all[train_size + val_size :]
    y_test = y_all[train_size + val_size :]

    # Scale
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, len(available_features))
    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train.reshape(-1, len(available_features))).reshape(
        X_train.shape
    )
    X_val_scaled = scaler.transform(X_val.reshape(-1, len(available_features))).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, len(available_features))).reshape(
        X_test.shape
    )

    # Loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 2. Train
    model = V3ClimateTransformer(input_dim=len(available_features)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    print("\n🚀 Starting Training...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item()
                val_mae += torch.mean(torch.abs(pred - y_b)).item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping")
                break

    # 3. Evaluate & Save
    model.load_state_dict(best_state)
    torch.save(
        {"model_state_dict": best_state, "input_dim": len(available_features), "scaler": scaler},
        MODELS_DIR / "v3_climate_transformer_resampled.pt",
    )

    joblib.dump(scaler, MODELS_DIR / "v3_scaler_resampled.joblib")

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(DEVICE)
            pred = model(X_b)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(y_b.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_mae = np.mean(np.abs(test_preds - test_targets))

    print(f"\n✅ FINAL TEST MAE: {test_mae:.4f} °C")

    # Save improvement report
    with open("improvement_report.txt", "w") as f:
        f.write(f"Resampled Method MAE: {test_mae:.4f}\n")


if __name__ == "__main__":
    main()
