"""
Tests for V3 Model Architecture.
Tests GatedResidualNetwork and V3ClimateTransformer.
"""

import pytest
import torch


class TestGatedResidualNetwork:
    """Test suite for GRN component."""

    def test_grn_output_shape(self, sample_grn, sample_input):
        """GRN should output correct shape."""
        # GRN expects (batch, seq, input_dim)
        x = torch.randn(4, 14, 25)
        output = sample_grn(x)
        assert output.shape == (4, 14, 32), f"Expected (4, 14, 32), got {output.shape}"

    def test_grn_with_different_dims(self):
        """GRN should handle dimension mismatch via skip connection."""
        from v3.src.model import GatedResidualNetwork

        grn = GatedResidualNetwork(input_dim=10, hidden_dim=32, output_dim=20)
        x = torch.randn(2, 5, 10)
        output = grn(x)
        assert output.shape == (2, 5, 20)

    def test_grn_same_dims_no_skip(self):
        """GRN should work without skip when dims match."""
        from v3.src.model import GatedResidualNetwork

        grn = GatedResidualNetwork(input_dim=16, hidden_dim=32, output_dim=16)
        assert grn.skip is None, "Skip connection should be None when dims match"

    def test_grn_gradient_flow(self, sample_grn):
        """Gradients should flow through GRN."""
        x = torch.randn(2, 7, 25, requires_grad=True)
        output = sample_grn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow through GRN"


class TestV3ClimateTransformer:
    """Test suite for main model architecture."""

    def test_model_output_shape(self, sample_model, sample_input):
        """Model should output correct prediction shape."""
        output = sample_model(sample_input)
        batch_size = sample_input.shape[0]
        pred_len = 7
        assert output.shape == (
            batch_size,
            pred_len,
        ), f"Expected ({batch_size}, {pred_len}), got {output.shape}"

    def test_model_forward_no_error(self, sample_model, sample_input):
        """Model forward pass should not raise errors."""
        try:
            _ = sample_model(sample_input)
        except Exception as e:
            pytest.fail(f"Model forward pass raised: {e}")

    def test_model_eval_mode(self, sample_model, sample_input):
        """Model should work in eval mode."""
        sample_model.eval()
        with torch.no_grad():
            output = sample_model(sample_input)
        assert output.shape[1] == 7

    def test_model_gradient_flow(self, sample_model, sample_input):
        """Gradients should propagate through model."""
        sample_input.requires_grad_(True)
        output = sample_model(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_model_parameters_count(self, sample_model):
        """Model should have trainable parameters."""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        assert total_params > 0
        assert trainable_params == total_params  # All should be trainable

    def test_model_deterministic_eval(self, sample_model, sample_input):
        """Model should be deterministic in eval mode."""
        sample_model.eval()
        with torch.no_grad():
            out1 = sample_model(sample_input)
            out2 = sample_model(sample_input)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_model_batch_independence(self, sample_model):
        """Each sample in batch should be processed independently."""
        sample_model.eval()
        x = torch.randn(2, 14, 25)

        with torch.no_grad():
            full_out = sample_model(x)
            out1 = sample_model(x[0:1])
            out2 = sample_model(x[1:2])

        assert torch.allclose(full_out[0], out1[0], atol=1e-5)
        assert torch.allclose(full_out[1], out2[0], atol=1e-5)


class TestModelFromCheckpoint:
    """Test model loading from checkpoint."""

    def test_checkpoint_loading(self, models_exist):
        """Model should load from checkpoint if it exists."""
        if not models_exist:
            pytest.skip("Model checkpoint not found")

        from v3.src.config import MODELS_DIR
        from v3.src.model import V3ClimateTransformer

        model = V3ClimateTransformer.from_checkpoint(str(MODELS_DIR / "v3_climate_transformer.pt"))
        assert model is not None
        assert not model.training  # Should be in eval mode
