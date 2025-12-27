"""
V3 MLflow Training Script
Production training with experiment tracking, model registry, and artifacts.

Usage:
    python -m v3.mlflow.train --experiment "v3-climate-transformer" --epochs 50

    # With MLflow UI:
    mlflow ui --port 5000
    # Then open http://localhost:5000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v3.src.config import MODELS_DIR, PROCESSED_DATA_DIR, V3Config
from v3.src.model import V3ClimateTransformer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train V3 Climate Transformer with MLflow tracking"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment", type=str, default="v3-climate-transformer", help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)",
    )

    # Model hyperparameters
    parser.add_argument(
        "--d-model", type=int, default=V3Config.D_MODEL, help="Transformer hidden dimension"
    )
    parser.add_argument(
        "--nhead", type=int, default=V3Config.NHEAD, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=V3Config.NUM_LAYERS, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=V3Config.DROPOUT, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=V3Config.EPOCHS, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=V3Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=V3Config.LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=V3Config.WEIGHT_DECAY, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--patience", type=int, default=V3Config.PATIENCE, help="Early stopping patience"
    )

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(PROCESSED_DATA_DIR / "weather_v3_ready.csv"),
        help="Path to processed data CSV",
    )

    return parser.parse_args()


def load_and_prepare_data(data_path: str, seq_len: int, pred_len: int):
    """Load data and create sequences."""
    print(f"📂 Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Feature columns (must match training notebook)
    feature_cols = V3Config.get_all_features()
    target_col = "temperature_celsius"

    # Filter to columns that exist
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"📊 Using {len(available_features)} features")

    X = df[available_features].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len - pred_len + 1):
        Xs.append(X_scaled[i : (i + seq_len)])
        ys.append(y[(i + seq_len) : (i + seq_len + pred_len)])

    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    # Time-series split (no shuffling)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size : train_size + val_size]
    y_val = y_seq[train_size : train_size + val_size]
    X_test = X_seq[train_size + val_size :]
    y_test = y_seq[train_size + val_size :]

    print(f"📈 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, len(available_features))


def train_model(args):
    """Main training function with MLflow tracking."""

    # Set up MLflow
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "patience": args.patience,
                "seq_len": V3Config.SEQ_LEN,
                "pred_len": V3Config.PRED_LEN,
            }
        )

        # Load data
        (X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim) = load_and_prepare_data(
            args.data_path, V3Config.SEQ_LEN, V3Config.PRED_LEN
        )

        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=args.batch_size,
            shuffle=False,  # No shuffle for time-series
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=args.batch_size,
            shuffle=False,
        )

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mlflow.log_param("device", device)
        print(f"🖥️ Training on {device}")

        # Create model
        model = V3ClimateTransformer(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seq_len=V3Config.SEQ_LEN,
            pred_len=V3Config.PRED_LEN,
        ).to(device)

        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parameters", total_params)
        print(f"🧠 Model parameters: {total_params:,}")

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
                    val_preds.append(y_pred.cpu().numpy())
                    val_targets.append(y_batch.cpu().numpy())

            val_loss /= len(val_loader)
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_mae = np.mean(np.abs(val_preds - val_targets))

            # Log metrics
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}°C"
            )

            # Scheduler step
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "seq_len": V3Config.SEQ_LEN,
                    "pred_len": V3Config.PRED_LEN,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                }
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

        # Test evaluation
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.eval()

        test_preds, test_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                test_preds.append(y_pred.cpu().numpy())
                test_targets.append(y_batch.numpy())

        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_mae = np.mean(np.abs(test_preds - test_targets))
        test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))

        mlflow.log_metrics(
            {"test_mae": test_mae, "test_rmse": test_rmse, "best_epoch": best_checkpoint["epoch"]}
        )

        print("\n🎯 Test Results:")
        print(f"   MAE:  {test_mae:.2f}°C")
        print(f"   RMSE: {test_rmse:.2f}°C")

        # Save artifacts
        MODELS_DIR.mkdir(exist_ok=True)

        # Save checkpoint
        best_checkpoint["test_mae"] = test_mae
        best_checkpoint["test_rmse"] = test_rmse
        checkpoint_path = MODELS_DIR / "v3_climate_transformer.pt"
        torch.save(best_checkpoint, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))

        # Save scaler
        scaler_path = MODELS_DIR / "v3_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        # Log model to MLflow registry
        mlflow.pytorch.log_model(model, "model", registered_model_name="V3ClimateTransformer")

        print("\n✅ Training complete!")
        print(f"   Artifacts saved to: {MODELS_DIR}")
        print(f"   MLflow run ID: {mlflow.active_run().info.run_id}")

        return test_mae


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
