"""
V4 Model Tests.
Tests for ensemble model components.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v4.src.config import V4Config
from v4.src.models import GatedResidualNetwork, TransformerForecaster


class TestGatedResidualNetwork:
    """Test GatedResidualNetwork module."""

    def test_grn_output_shape(self):
        """GRN should produce correct output shape."""
        grn = GatedResidualNetwork(input_dim=10, hidden_dim=32, output_dim=20)
        x = torch.randn(4, 10)
        output = grn(x)
        assert output.shape == (4, 20)

    def test_grn_same_dim(self):
        """GRN with same input/output dim should work."""
        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32, output_dim=16)
        x = torch.randn(4, 16)
        output = grn(x)
        assert output.shape == (4, 16)

    def test_grn_gradient_flow(self):
        """GRN should allow gradient flow."""
        grn = GatedResidualNetwork(input_dim=10, hidden_dim=32, output_dim=10)
        x = torch.randn(4, 10, requires_grad=True)
        output = grn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestTransformerForecaster:
    """Test TransformerForecaster model."""

    def test_transformer_output_shape(self):
        """Transformer should produce correct output shape."""
        model = TransformerForecaster(
            input_dim=25, d_model=64, nhead=4, num_layers=2, seq_len=14, pred_len=7
        )
        x = torch.randn(4, 14, 25)
        output = model(x)
        assert output.shape == (4, 7)

    def test_transformer_eval_mode(self):
        """Transformer should work in eval mode."""
        model = TransformerForecaster(
            input_dim=25, d_model=64, nhead=4, num_layers=2
        )
        model.eval()
        x = torch.randn(1, 14, 25)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 7)

    def test_transformer_batch_independence(self):
        """Different batches should produce different outputs."""
        model = TransformerForecaster(input_dim=25, d_model=64, nhead=4, num_layers=2)
        model.eval()

        x1 = torch.randn(1, 14, 25)
        x2 = torch.randn(1, 14, 25)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert not torch.allclose(out1, out2)

    def test_transformer_deterministic(self):
        """Same input should produce same output in eval mode."""
        model = TransformerForecaster(input_dim=25, d_model=64, nhead=4, num_layers=2)
        model.eval()

        x = torch.randn(1, 14, 25)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2)


class TestV4Config:
    """Test V4Config class."""

    def test_climate_zone_tropical(self):
        """Equator should be tropical."""
        assert V4Config.get_climate_zone(0) == "Tropical"
        assert V4Config.get_climate_zone(10) == "Tropical"

    def test_climate_zone_subtropical(self):
        """~30° should be subtropical."""
        assert V4Config.get_climate_zone(30) == "Subtropical"
        assert V4Config.get_climate_zone(-30) == "Subtropical"

    def test_climate_zone_temperate(self):
        """~45° should be temperate."""
        assert V4Config.get_climate_zone(45) == "Temperate"

    def test_climate_zone_polar(self):
        """~80° should be polar."""
        assert V4Config.get_climate_zone(80) == "Polar"

    def test_hemisphere_detection(self):
        """Hemisphere should be correctly detected."""
        assert V4Config.get_hemisphere(30) == "Northern"
        assert V4Config.get_hemisphere(-30) == "Southern"
        assert V4Config.get_hemisphere(0) == "Northern"

    def test_top_features_count(self):
        """Should have correct number of top features."""
        assert len(V4Config.TOP_FEATURES) == 15

    def test_ensemble_weights_sum(self):
        """Ensemble weights should sum to 1."""
        for zone, weights in V4Config.ENSEMBLE_WEIGHTS.items():
            assert abs(sum(weights) - 1.0) < 0.001
