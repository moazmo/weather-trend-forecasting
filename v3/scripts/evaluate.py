"""
Model Evaluation Script for V3 Hybrid Model
Evaluates trained HybridClimateTransformer and outputs metrics.
Synchronized with v3/mlflow/train.py logic.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v3.src.config import MODELS_DIR, PROCESSED_DATA_DIR
from v3.src.model import HybridClimateTransformer


# --- Helper Functions (Mirrored from train.py) ---
def load_and_resample(path):
    print("📊 Loading raw data...", flush=True)
    df = pd.read_csv(path)
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    resampled_dfs = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("🔄 Resampling to Daily Frequency...", flush=True)
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

    print("🔄 Generating hybrid sequences...", flush=True)
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


def evaluate_model():
    """Load model and evaluate on test set."""

    # Paths
    model_path = MODELS_DIR / "v3_climate_transformer.pt"
    scaler_path = MODELS_DIR / "v3_scaler.joblib"
    data_path = PROCESSED_DATA_DIR / "weather_v3_ready.csv"
    metrics_path = MODELS_DIR / "evaluation_results.json"
    preds_path = MODELS_DIR / "predictions_vs_actual.csv"

    print("📂 Loading model and data...", flush=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Load Scalers and Config
    artifacts = joblib.load(scaler_path)
    current_country_encoder = artifacts["country_encoder"]
    scaler_dyn = artifacts["scaler_dyn"]
    scaler_stat = artifacts["scaler_stat"]
    dyn_features = artifacts["dyn_features"]
    stat_features = artifacts["stat_features"]

    # 1. Load Data
    df = load_and_resample(data_path)

    # 2. Features
    # Filter for known countries
    df = df[df["country"].isin(current_country_encoder.classes_)]
    df["country_id"] = current_country_encoder.transform(df["country"])

    # 3. Sequences
    X_dyn, X_stat, X_country, y = create_hybrid_sequences(
        df,
        dyn_features,
        stat_features,
        "country_id",
        "temperature_celsius",
        checkpoint.get("seq_len", 14),
        checkpoint.get("pred_len", 7),
    )

    # 4. Scaling
    orig_shape = X_dyn.shape
    X_dyn_flat = X_dyn.reshape(-1, orig_shape[-1])
    X_dyn_scaled = scaler_dyn.transform(X_dyn_flat).reshape(orig_shape)
    X_stat_scaled = scaler_stat.transform(X_stat)

    # 5. Split (Match train.py split logic exactly)
    indices = np.arange(len(y))
    # train.py: train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    # Select Test Data
    X_dyn_test = X_dyn_scaled[test_idx]
    X_stat_test = X_stat_scaled[test_idx]
    X_country_test = X_country[test_idx]
    y_test_true = y[test_idx]

    print(f"📊 Evaluating on {len(test_idx)} test samples...", flush=True)

    # 6. Model
    model = HybridClimateTransformer(
        num_countries=checkpoint["num_countries"],
        dyn_input_dim=checkpoint["dyn_input_dim"],
        stat_input_dim=checkpoint["stat_input_dim"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        seq_len=checkpoint["seq_len"],
        pred_len=checkpoint["pred_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 7. Evaluate
    batch_size = 128
    test_data = TensorDataset(
        torch.FloatTensor(X_dyn_test),
        torch.FloatTensor(X_stat_test),
        torch.LongTensor(X_country_test),
        torch.FloatTensor(y_test_true),
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for xd, xs, xc, y_true in test_loader:
            pred = model(xd, xs, xc)
            all_preds.append(pred.numpy())
            all_actuals.append(y_true.numpy())

    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    # 8. Metrics
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mape = np.mean(np.abs((preds - actuals) / (actuals + 1e-8))) * 100

    # Per-horizon metrics
    pred_len = checkpoint["pred_len"]
    horizon_mae = [np.mean(np.abs(preds[:, i] - actuals[:, i])) for i in range(pred_len)]

    print("\n🎯 Test Results:", flush=True)
    print(f"   MAE:  {mae:.2f}°C")
    print(f"   RMSE: {rmse:.2f}°C")

    metrics = {
        "test_mae": round(float(mae), 2),
        "test_rmse": round(float(rmse), 2),
        "test_mape": round(float(mape), 2),
        "test_samples": len(test_idx),
        "horizon_mae": [round(float(m), 2) for m in horizon_mae],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}", flush=True)

    # Save predictions sample
    preds_flat = preds.flatten()
    actual_flat = actuals.flatten()

    if len(preds_flat) > 10000:
        idx = np.random.choice(len(preds_flat), 10000, replace=False)
        preds_flat = preds_flat[idx]
        actual_flat = actual_flat[idx]

    pd.DataFrame({"actual": actual_flat, "predicted": preds_flat}).to_csv(preds_path, index=False)
    print(f"📈 Predictions saved to {preds_path}", flush=True)


if __name__ == "__main__":
    evaluate_model()
