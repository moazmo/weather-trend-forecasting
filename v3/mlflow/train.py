"""
V3.1 MLflow Training Script (Hybrid Architecture)
Production training with experiment tracking, model registry, and artifacts.
Logic synchronized with `07_advanced_model.ipynb`.

Usage:
    python -m v3.mlflow.train --experiment "v3-hybrid-transformer" --epochs 40
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v3.src.config import MODELS_DIR, PROCESSED_DATA_DIR, V3Config
from v3.src.model import HybridClimateTransformer


# --- Data Logic (Matches Notebook) ---
def load_and_resample(path):
    print("📊 Loading raw data...")
    df = pd.read_csv(path)
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    resampled_dfs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("🔄 Resampling to Daily Frequency...")
    for country, group in df.groupby("country"):
        group = group.set_index("last_updated")
        daily = group[numeric_cols].resample("1D").mean()
        daily = daily.interpolate(method="time", limit=2).dropna()
        daily["country"] = country
        daily = daily.reset_index()
        resampled_dfs.append(daily)

    return pd.concat(resampled_dfs, ignore_index=True)


def create_hybrid_sequences(df, dyn_cols, stat_cols, country_col, target_col, seq_len, pred_len):
    X_dyn, X_stat, X_country, y = [], [], [], []

    print("🔄 Generating hybrid sequences...")
    for _, group in df.groupby("country"):
        group = group.sort_values("last_updated")
        if len(group) < seq_len + pred_len:
            continue

        d_data = group[dyn_cols].values.astype(np.float32)
        s_data = group[stat_cols].values.astype(np.float32)
        c_data = group[country_col].values.astype(np.int64)
        t_data = group[target_col].values.astype(np.float32)

        for i in range(len(d_data) - seq_len - pred_len + 1):
            X_dyn.append(d_data[i : i + seq_len])
            X_stat.append(s_data[i])
            X_country.append(c_data[i])
            y.append(t_data[i + seq_len : i + seq_len + pred_len])

    return np.array(X_dyn), np.array(X_stat), np.array(X_country), np.array(y)


def run_training(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        log_params = vars(args)
        mlflow.log_params(log_params)

        # 1. Load Data
        df = load_and_resample(args.data_path)

        # 2. Features
        country_encoder = LabelEncoder()
        df["country_id"] = country_encoder.fit_transform(df["country"])

        DYNAMIC_FEATURES = [
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
            "air_quality_PM2.5",
            "month_sin",
            "month_cos",
        ]
        STATIC_FEATURES = ["latitude", "longitude", "abs_latitude", "hemisphere_encoded"]

        dyn_avail = [c for c in DYNAMIC_FEATURES if c in df.columns]
        stat_avail = [c for c in STATIC_FEATURES if c in df.columns]

        # 3. Sequences
        X_dyn, X_stat, X_country, y = create_hybrid_sequences(
            df,
            dyn_avail,
            stat_avail,
            "country_id",
            "temperature_celsius",
            V3Config.SEQ_LEN,
            V3Config.PRED_LEN,
        )

        # 4. Scaling
        scaler_dyn = StandardScaler()
        # Scale flattened dynamic
        orig_shape = X_dyn.shape
        X_dyn_flat = X_dyn.reshape(-1, orig_shape[-1])
        X_dyn_scaled = scaler_dyn.fit_transform(X_dyn_flat).reshape(orig_shape)

        scaler_stat = StandardScaler()
        X_stat_scaled = scaler_stat.fit_transform(X_stat)

        # 5. Split
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, shuffle=True
        )
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

        def get_tensors(idxs):
            return (
                torch.FloatTensor(X_dyn_scaled[idxs]),
                torch.FloatTensor(X_stat_scaled[idxs]),
                torch.LongTensor(X_country[idxs]),
                torch.FloatTensor(y[idxs]),
            )

        train_data = TensorDataset(*get_tensors(train_idx))
        val_data = TensorDataset(*get_tensors(val_idx))
        test_data = TensorDataset(*get_tensors(test_idx))

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size)
        _test_loader = DataLoader(test_data, batch_size=args.batch_size)  # noqa: F841

        # 6. Model
        model = HybridClimateTransformer(
            num_countries=len(country_encoder.classes_),
            dyn_input_dim=len(dyn_avail),
            stat_input_dim=len(stat_avail),
            d_model=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=args.patience // 2, factor=0.5
        )

        # 7. Train
        best_loss = float("inf")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for xd, xs, xc, y_true in train_loader:
                xd, xs, xc, y_true = xd.to(device), xs.to(device), xc.to(device), y_true.to(device)
                optimizer.zero_grad()
                pred = model(xd, xs, xc)
                loss = criterion(pred, y_true)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Val
            model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for xd, xs, xc, y_true in val_loader:
                    xd, xs, xc, y_true = (
                        xd.to(device),
                        xs.to(device),
                        xc.to(device),
                        y_true.to(device),
                    )
                    pred = model(xd, xs, xc)
                    val_loss += criterion(pred, y_true).item()
                    val_mae += torch.mean(torch.abs(pred - y_true)).item()
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae}, step=epoch
            )

            print(f"Epoch {epoch + 1}/{args.epochs} | MAE: {val_mae:.2f}")
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss

                # Save Model
                model_path = MODELS_DIR / "v3_hybrid_best.pt"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path))

                # Save Artifacts
                artifacts = {
                    "country_encoder": country_encoder,
                    "scaler_dyn": scaler_dyn,
                    "scaler_stat": scaler_stat,
                    "dyn_features": dyn_avail,
                    "stat_features": stat_avail,
                    "num_countries": len(country_encoder.classes_),
                }
                art_path = MODELS_DIR / "v3_1_production_artifacts.joblib"
                joblib.dump(artifacts, art_path)
                mlflow.log_artifact(str(art_path))

        print("✅ Training Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="v3-hybrid", help="Experiment name")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--data-path", type=str, default=str(PROCESSED_DATA_DIR / "weather_v3_ready.csv")
    )

    args = parser.parse_args()
    run_training(args)
