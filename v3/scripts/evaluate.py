"""
Model Evaluation Script for DVC Pipeline
Evaluates trained model and outputs metrics.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v3.src.config import V3Config
from v3.src.model import V3ClimateTransformer


def evaluate_model():
    """Load model and evaluate on test set."""

    # Paths
    model_path = Path("v3/models/v3_climate_transformer.pt")
    scaler_path = Path("v3/models/v3_scaler.joblib")
    data_path = Path("data/processed/weather_v3_ready.csv")
    metrics_path = Path("v3/models/evaluation_results.json")
    preds_path = Path("v3/models/predictions_vs_actual.csv")

    print("📂 Loading model and data...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Load model
    model = V3ClimateTransformer(
        input_dim=checkpoint["input_dim"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        seq_len=checkpoint.get("seq_len", V3Config.SEQ_LEN),
        pred_len=checkpoint.get("pred_len", V3Config.PRED_LEN),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load data
    df = pd.read_csv(data_path)

    # Feature columns
    feature_cols = V3Config.get_all_features()
    available_features = [c for c in feature_cols if c in df.columns]
    target_col = "temperature_celsius"

    X = df[available_features].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Scale
    X_scaled = scaler.transform(X)

    # Create test sequences (last 15% of data)
    seq_len = V3Config.SEQ_LEN
    pred_len = V3Config.PRED_LEN

    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len - pred_len + 1):
        Xs.append(X_scaled[i : (i + seq_len)])
        ys.append(y[(i + seq_len) : (i + seq_len + pred_len)])

    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    # Use last 15% as test
    test_start = int(len(X_seq) * 0.85)
    X_test = X_seq[test_start:]
    y_test = y_seq[test_start:]

    print(f"📊 Evaluating on {len(X_test)} test samples...")

    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        preds = model(X_tensor).numpy()

    # Calculate metrics
    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    mape = np.mean(np.abs((preds - y_test) / (y_test + 1e-8))) * 100

    # Per-horizon metrics
    horizon_mae = [np.mean(np.abs(preds[:, i] - y_test[:, i])) for i in range(pred_len)]

    print("\n🎯 Test Results:")
    print(f"   MAE:  {mae:.2f}°C")
    print(f"   RMSE: {rmse:.2f}°C")
    print(f"   MAPE: {mape:.2f}%")

    # Save metrics
    metrics = {
        "test_mae": round(float(mae), 2),
        "test_rmse": round(float(rmse), 2),
        "test_mape": round(float(mape), 2),
        "test_samples": len(X_test),
        "horizon_mae": [round(float(m), 2) for m in horizon_mae],
        "model_params": {
            "input_dim": checkpoint["input_dim"],
            "d_model": checkpoint["d_model"],
            "nhead": checkpoint["nhead"],
            "num_layers": checkpoint["num_layers"],
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved to {metrics_path}")

    # Save predictions for plotting
    # Flatten for scatter plot
    preds_flat = preds.flatten()
    actual_flat = y_test.flatten()

    # Sample if too many points
    if len(preds_flat) > 10000:
        idx = np.random.choice(len(preds_flat), 10000, replace=False)
        preds_flat = preds_flat[idx]
        actual_flat = actual_flat[idx]

    pd.DataFrame({"actual": actual_flat, "predicted": preds_flat}).to_csv(preds_path, index=False)
    print(f"📈 Predictions saved to {preds_path}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
