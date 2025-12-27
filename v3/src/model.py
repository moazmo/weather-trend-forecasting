"""
V3 Model Architecture
Climate-Aware Transformer with Gated Residual Networks.
"""

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for feature filtering.

    The GRN automatically suppresses irrelevant features through
    a learned gating mechanism, making the model robust to
    optional/missing user inputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection for dimension mismatch
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        residual = self.skip(x) if self.skip else x

        h = torch.relu(self.fc1(x))
        h = self.dropout(h)

        # Gated Linear Unit (GLU)
        gate = torch.sigmoid(self.gate(h))
        out = self.fc2(h) * gate

        return self.layer_norm(out + residual)


class V3ClimateTransformer(nn.Module):
    """
    Climate-Aware Transformer for weather forecasting.

    Architecture:
        1. Input GRN: Filters and projects features to d_model
        2. Positional Encoding: Learned positional embeddings
        3. Transformer Encoder: Multi-head attention layers
        4. Output GRN: Final feature refinement
        5. Prediction Head: Projects to forecast length

    Args:
        input_dim: Number of input features
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
        seq_len: Input sequence length
        pred_len: Output prediction length
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
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Input GRN (feature filtering)
        self.input_grn = GatedResidualNetwork(input_dim, d_model * 2, d_model, dropout)

        # Learnable Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output GRN
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # Prediction Head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temperature forecasting.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Predictions of shape (batch, pred_len)
        """
        # Feature projection and filtering
        x = self.input_grn(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer encoding
        x = self.transformer(x)

        # Take last timestep and refine
        x = self.output_grn(x[:, -1, :])  # (batch, d_model)

        # Project to prediction length
        return self.output_head(x)  # (batch, pred_len)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "V3ClimateTransformer":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to .pt file
            device: Device to load model on

        Returns:
            Loaded model in eval mode
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = cls(
            input_dim=checkpoint["input_dim"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            seq_len=checkpoint.get("seq_len", 14),
            pred_len=checkpoint.get("pred_len", 7),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model
