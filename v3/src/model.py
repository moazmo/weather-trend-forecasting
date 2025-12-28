import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for variable selection and feature gating."""

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
    """V3 Climate Transformer with Gated Residual Networks."""

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


class HybridClimateTransformer(nn.Module):
    """
    V3.1 Hybrid Static-Dynamic Transformer with Country Embeddings.
    Separates dynamic time-series features from static geographic features.
    """

    def __init__(
        self,
        num_countries,
        dyn_input_dim,
        stat_input_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.2,
        seq_len=14,
        pred_len=7,
    ):
        super().__init__()

        # 1. Feature Processors
        # Learnable vector per country to capture latent geography
        self.country_emb = nn.Embedding(num_countries, 16)

        # Dynamic Feature Encoder (Linear projection to d_model)
        self.dyn_encoder = nn.Linear(dyn_input_dim, d_model)

        # Static Feature Encoder (Linear projection to discrete size)
        # +16 for country embedding concatenation
        self.stat_encoder = nn.Linear(stat_input_dim + 16, d_model)

        # 2. Transformer Backbone (Processes only dynamic features over time)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Gating / Combination
        # Concatenate: [Target_Time_Context, Static_Context] -> Regressor
        self.output_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len),
        )

    def forward(self, x_dyn, x_stat, x_country):
        """
        Args:
            x_dyn: Dynamic features [Batch, Seq, Dyn_Dim]
            x_stat: Static features [Batch, Stat_Dim]
            x_country: Country IDs [Batch]
        """
        # A. Process Static Context
        c_emb = self.country_emb(x_country)  # [Batch, 16]
        stat_in = torch.cat([x_stat, c_emb], dim=1)  # [Batch, Stat_Dim + 16]
        stat_context = self.stat_encoder(stat_in)  # [Batch, d_model]

        # B. Process Dynamic Sequence
        dyn_emb = self.dyn_encoder(x_dyn)  # [Batch, Seq, d_model]
        dyn_emb = dyn_emb + self.pos_encoder  # Add Position info

        # C. Transformer Pass
        time_context = self.transformer(dyn_emb)  # [Batch, Seq, d_model]

        # Take hidden state of the last time step
        last_step = time_context[:, -1, :]  # [Batch, d_model]

        # D. Combine & Predict
        # Combine temporal context with static geographic context
        combined = torch.cat([last_step, stat_context], dim=1)  # [Batch, d_model*2]
        return self.output_head(combined)
