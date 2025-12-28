"""
V4 Models Module.
Ensemble model architecture combining XGBoost and Transformer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v4.src.config import V4Config


# =============================================================================
# Gated Residual Network (from V2)
# =============================================================================


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for variable selection and feature gating."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h) * torch.sigmoid(self.gate(h))
        return self.layer_norm(out + residual)


# =============================================================================
# Transformer Model
# =============================================================================


class TransformerForecaster(nn.Module):
    """
    Transformer-based temperature forecaster with GRN.
    Adapted from V2 Advanced Transformer.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        seq_len: int = 14,
        pred_len: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input GRN for variable selection
        self.input_grn = GatedResidualNetwork(input_dim, d_model * 2, d_model, dropout)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output GRN
        self.output_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # Prediction head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Predictions of shape (batch_size, pred_len)
        """
        # Input processing
        x = self.input_grn(x) + self.pos_encoder

        # Transformer encoding
        x = self.transformer(x)

        # Output processing (use last timestep)
        x = self.output_grn(x[:, -1, :])

        # Prediction
        return self.output_head(x)


# =============================================================================
# XGBoost Wrapper
# =============================================================================


class XGBoostForecaster:
    """
    XGBoost-based temperature forecaster wrapper.
    Best performer from assessment (2.94°C MAE).
    """

    def __init__(self, model=None, pred_len: int = 7):
        """
        Initialize XGBoost forecaster.

        Args:
            model: Pre-trained XGBoost model (or None to create new)
            pred_len: Prediction horizon
        """
        self.model = model
        self.pred_len = pred_len
        self.models = []  # One model per prediction day

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train XGBoost models (one per prediction day).

        Args:
            X: Input features of shape (n_samples, n_features)
            y: Targets of shape (n_samples, pred_len)
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.models = []
        params = V4Config.XGBOOST_PARAMS.copy()

        for day in range(self.pred_len):
            model = XGBRegressor(**params)
            model.fit(X, y[:, day])
            self.models.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples, pred_len)
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = np.zeros((X.shape[0], self.pred_len))
        for day, model in enumerate(self.models):
            predictions[:, day] = model.predict(X)

        return predictions

    def get_feature_importance(self) -> dict:
        """Get feature importance from first model."""
        if not self.models:
            return {}
        return dict(zip(range(len(self.models[0].feature_importances_)), self.models[0].feature_importances_))


# =============================================================================
# Ensemble Model
# =============================================================================


class EnsembleForecaster:
    """
    Ensemble forecaster combining XGBoost and Transformer.
    Uses climate-zone-specific weights.
    """

    def __init__(
        self,
        xgboost_model: XGBoostForecaster | None = None,
        transformer_model: TransformerForecaster | None = None,
        device: str = "cpu",
    ):
        """
        Initialize ensemble.

        Args:
            xgboost_model: Pre-trained XGBoost forecaster
            transformer_model: Pre-trained Transformer model
            device: PyTorch device
        """
        self.xgboost = xgboost_model
        self.transformer = transformer_model
        self.device = device

        if self.transformer is not None:
            self.transformer.to(device)
            self.transformer.eval()

    def predict(
        self,
        sequence: np.ndarray,
        climate_zone: str = "Temperate",
        return_individual: bool = False,
    ) -> np.ndarray | dict:
        """
        Make ensemble prediction.

        Args:
            sequence: Input sequence of shape (seq_len, n_features)
            climate_zone: Climate zone for weight selection
            return_individual: If True, return individual model predictions

        Returns:
            Ensemble predictions (or dict with individual predictions)
        """
        predictions = {}

        # XGBoost prediction
        if self.xgboost is not None:
            xgb_input = sequence.flatten().reshape(1, -1)
            predictions["xgboost"] = self.xgboost.predict(xgb_input)[0]

        # Transformer prediction
        if self.transformer is not None:
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                predictions["transformer"] = self.transformer(x).cpu().numpy()[0]

        # Get ensemble weights
        weights = V4Config.ENSEMBLE_WEIGHTS.get(climate_zone, (0.5, 0.5))
        xgb_weight, tf_weight = weights

        # Compute ensemble prediction
        if "xgboost" in predictions and "transformer" in predictions:
            ensemble = xgb_weight * predictions["xgboost"] + tf_weight * predictions["transformer"]
        elif "xgboost" in predictions:
            ensemble = predictions["xgboost"]
        elif "transformer" in predictions:
            ensemble = predictions["transformer"]
        else:
            raise ValueError("No models available for prediction")

        if return_individual:
            return {
                "ensemble": ensemble,
                "xgboost": predictions.get("xgboost"),
                "transformer": predictions.get("transformer"),
                "weights": {"xgboost": xgb_weight, "transformer": tf_weight},
            }

        return ensemble

    def predict_with_uncertainty(
        self, sequence: np.ndarray, climate_zone: str = "Temperate"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.

        Uses disagreement between models as uncertainty measure.

        Args:
            sequence: Input sequence
            climate_zone: Climate zone

        Returns:
            Tuple of (ensemble_pred, lower_bound, upper_bound)
        """
        result = self.predict(sequence, climate_zone, return_individual=True)

        ensemble = result["ensemble"]

        # Estimate uncertainty from model disagreement
        if result["xgboost"] is not None and result["transformer"] is not None:
            diff = np.abs(result["xgboost"] - result["transformer"])
            uncertainty = diff * 0.5  # Scale factor
        else:
            # Default uncertainty of ±2°C
            uncertainty = np.full_like(ensemble, 2.0)

        lower = ensemble - uncertainty
        upper = ensemble + uncertainty

        return ensemble, lower, upper


# =============================================================================
# Anomaly Detector
# =============================================================================


class AnomalyDetector:
    """
    Anomaly detection for input validation.
    Based on assessment Isolation Forest results.
    """

    def __init__(self, contamination: float = 0.05):
        """
        Initialize detector.

        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.model = None

    def fit(self, X: np.ndarray):
        """Fit Isolation Forest model."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("Scikit-learn not installed")

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self.model.fit(X)

    def is_anomaly(self, X: np.ndarray) -> bool:
        """Check if input is an anomaly."""
        if self.model is None:
            return False
        return self.model.predict(X.reshape(1, -1))[0] == -1

    def filter_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Return mask of non-anomalous samples."""
        if self.model is None:
            return np.ones(len(X), dtype=bool)
        return self.model.predict(X) == 1
