"""
Unit Tests for Mamba LVM Models - Phase 5
==========================================

Tests all 5 Mamba architectures:
- Shape correctness
- Gradient flow
- Parameter counts
- Numerical stability

Run:
    pytest tests/test_mamba_models.py -v
"""

import pytest
import torch

from app.lvm.mamba import (
    create_model,
    count_parameters,
    MambaS,
    MambaHybridLocal,
    MambaXL,
    MambaSandwich,
    MambaGR,
)


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class TestMambaS:
    """Test Mamba-S (Pure SSM, Small)."""

    def test_forward_shape(self, device):
        """Test output shape is correct."""
        model = MambaS(d_model=256, n_layers=8, d_state=128).to(device)

        # Input: [B=4, L=5, D=768]
        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768), f"Expected (4, 5, 768), got {out.shape}"

    def test_l2_normalized(self, device):
        """Test output is L2-normalized."""
        model = MambaS(d_model=256, n_layers=8).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        # Check L2 norm is 1.0 for each vector
        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flow(self, device):
        """Test gradients flow through model."""
        model = MambaS(d_model=256, n_layers=4).to(device)

        x = torch.randn(2, 5, 768, device=device)
        target = torch.randn(2, 5, 768, device=device)

        out = model(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_parameter_count(self):
        """Test parameter count is reasonable (~7M with I/O projections)."""
        model = MambaS(d_model=256, n_layers=8, d_state=128)
        n_params = count_parameters(model)

        # Note: Higher than PRD estimate due to 768 I/O projections
        # PRD ~1.3M was for core SSM only, actual ~7M includes I/O
        assert 6.0e6 < n_params < 9.0e6, f"Expected ~7M params, got {n_params:,}"
        print(f"MambaS params: {n_params:,}")


class TestMambaHybridLocal:
    """Test Mamba-H (Hybrid 80/20)."""

    def test_forward_shape(self, device):
        """Test output shape is correct."""
        model = MambaHybridLocal(
            d_model=320, n_layers=12, local_attn_every=4
        ).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)

    def test_local_attention_insertion(self):
        """Test local attention is inserted correctly."""
        model = MambaHybridLocal(
            d_model=320, n_layers=12, local_attn_every=4
        )

        # Should have 12 layers total
        assert len(model.layers) == 12

        # Every 4th layer should be LocalAttention
        from app.lvm.mamba.blocks import LocalAttention, MambaBlock
        for i, layer in enumerate(model.layers):
            if (i + 1) % 4 == 0:
                assert isinstance(layer, LocalAttention), f"Layer {i} should be LocalAttention"
            else:
                assert isinstance(layer, MambaBlock), f"Layer {i} should be MambaBlock"

    def test_parameter_count(self):
        """Test parameter count is reasonable (~13M with I/O projections)."""
        model = MambaHybridLocal(d_model=320, n_layers=12)
        n_params = count_parameters(model)

        # Note: Higher than PRD estimate due to 768 I/O projections
        assert 11.0e6 < n_params < 15.0e6, f"Expected ~13M params, got {n_params:,}"
        print(f"MambaHybridLocal params: {n_params:,}")


class TestMambaXL:
    """Test Mamba-XL (Deeper/Wider Pure SSM)."""

    def test_forward_shape(self, device):
        """Test output shape is correct."""
        model = MambaXL(d_model=384, n_layers=16, d_state=192).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)

    def test_parameter_count(self):
        """Test parameter count is reasonable (~31M with I/O projections)."""
        model = MambaXL(d_model=384, n_layers=16, d_state=192)
        n_params = count_parameters(model)

        # Note: Higher than PRD estimate due to 768 I/O projections
        assert 28.0e6 < n_params < 35.0e6, f"Expected ~31M params, got {n_params:,}"
        print(f"MambaXL params: {n_params:,}")


class TestMambaSandwich:
    """Test Mamba-Sandwich (Attn->SSM->Attn)."""

    def test_forward_shape(self, device):
        """Test output shape is correct."""
        model = MambaSandwich(
            d_model=320, n_layers_mamba=8, n_layers_local=4
        ).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)

    def test_layer_structure(self):
        """Test sandwich structure (front->trunk->back)."""
        model = MambaSandwich(
            d_model=320, n_layers_mamba=8, n_layers_local=4
        )

        # Should have 2 front, 8 trunk, 2 back
        assert len(model.front_layers) == 2
        assert len(model.trunk_layers) == 8
        assert len(model.back_layers) == 2

        from app.lvm.mamba.blocks import LocalAttention, MambaBlock
        for layer in model.front_layers:
            assert isinstance(layer, LocalAttention)
        for layer in model.trunk_layers:
            assert isinstance(layer, MambaBlock)
        for layer in model.back_layers:
            assert isinstance(layer, LocalAttention)

    def test_parameter_count(self):
        """Test parameter count is reasonable (~13M with I/O projections)."""
        model = MambaSandwich(d_model=320, n_layers_mamba=8, n_layers_local=4)
        n_params = count_parameters(model)

        # Note: Higher than PRD estimate due to 768 I/O projections
        assert 11.0e6 < n_params < 14.0e6, f"Expected ~13M params, got {n_params:,}"
        print(f"MambaSandwich params: {n_params:,}")


class TestMambaGR:
    """Test Mamba-GR (SSM + GRU Gate)."""

    def test_forward_shape(self, device):
        """Test output shape is correct."""
        model = MambaGR(
            d_model=288, n_layers=10, gru_hidden=256
        ).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)

    def test_gru_gates_exist(self):
        """Test GRU gates are present."""
        model = MambaGR(d_model=288, n_layers=10, gru_hidden=256)

        # Should have 10 Mamba layers and 10 GRU gates
        assert len(model.mamba_layers) == 10
        assert len(model.gru_gates) == 10

        from app.lvm.mamba.blocks import MambaBlock, GRUGate
        for layer in model.mamba_layers:
            assert isinstance(layer, MambaBlock)
        for gate in model.gru_gates:
            assert isinstance(gate, GRUGate)

    def test_parameter_count(self):
        """Test parameter count is reasonable (~16M with I/O projections)."""
        model = MambaGR(d_model=288, n_layers=10, gru_hidden=256)
        n_params = count_parameters(model)

        # Note: Higher than PRD estimate due to 768 I/O projections
        assert 14.0e6 < n_params < 18.0e6, f"Expected ~16M params, got {n_params:,}"
        print(f"MambaGR params: {n_params:,}")


class TestModelFactory:
    """Test create_model factory function."""

    def test_create_all_models(self):
        """Test all models can be created."""
        # Each model has specific parameters
        configs = {
            'mamba_s': {'d_model': 128, 'n_layers': 2},
            'mamba_hybrid_local': {'d_model': 128, 'n_layers': 4, 'local_attn_every': 2},
            'mamba_xl': {'d_model': 128, 'n_layers': 2},
            'mamba_sandwich': {'d_model': 128, 'n_layers_mamba': 2, 'n_layers_local': 2},
            'mamba_gr': {'d_model': 128, 'n_layers': 2, 'gru_hidden': 64},
        }

        for model_type, kwargs in configs.items():
            model = create_model(model_type, **kwargs)
            assert model is not None, f"Failed to create {model_type}"

    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError):
            create_model('invalid_model')


class TestAlignmentHead:
    """Test optional alignment head."""

    def test_alignment_head_enabled(self, device):
        """Test model with alignment head."""
        model = MambaS(
            d_model=256,
            n_layers=4,
            use_alignment_head=True,
            alignment_alpha=0.25,
        ).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)

        # Check alignment head exists
        assert hasattr(model, 'alignment_head')
        assert model.alignment_head is not None

    def test_alignment_head_disabled(self, device):
        """Test model without alignment head."""
        model = MambaS(
            d_model=256,
            n_layers=4,
            use_alignment_head=False,
        ).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert out.shape == (4, 5, 768)


class TestNumericalStability:
    """Test numerical stability."""

    @pytest.mark.parametrize('model_type,kwargs', [
        ('mamba_s', {'d_model': 128, 'n_layers': 2}),
        ('mamba_hybrid_local', {'d_model': 128, 'n_layers': 4, 'local_attn_every': 2}),
        ('mamba_xl', {'d_model': 128, 'n_layers': 2}),
        ('mamba_sandwich', {'d_model': 128, 'n_layers_mamba': 2, 'n_layers_local': 2}),
        ('mamba_gr', {'d_model': 128, 'n_layers': 2, 'gru_hidden': 64}),
    ])
    def test_no_nans(self, model_type, kwargs, device):
        """Test no NaNs in forward pass."""
        model = create_model(model_type, **kwargs).to(device)

        x = torch.randn(4, 5, 768, device=device)
        out = model(x)

        assert not torch.isnan(out).any(), f"NaN in output for {model_type}"
        assert not torch.isinf(out).any(), f"Inf in output for {model_type}"

    @pytest.mark.parametrize('model_type,kwargs', [
        ('mamba_s', {'d_model': 128, 'n_layers': 2}),
        ('mamba_hybrid_local', {'d_model': 128, 'n_layers': 4, 'local_attn_every': 2}),
        ('mamba_xl', {'d_model': 128, 'n_layers': 2}),
        ('mamba_sandwich', {'d_model': 128, 'n_layers_mamba': 2, 'n_layers_local': 2}),
        ('mamba_gr', {'d_model': 128, 'n_layers': 2, 'gru_hidden': 64}),
    ])
    def test_different_batch_sizes(self, model_type, kwargs, device):
        """Test model works with different batch sizes."""
        model = create_model(model_type, **kwargs).to(device)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 5, 768, device=device)
            out = model(x)
            assert out.shape == (batch_size, 5, 768)

    @pytest.mark.parametrize('model_type,kwargs', [
        ('mamba_s', {'d_model': 128, 'n_layers': 2}),
        ('mamba_hybrid_local', {'d_model': 128, 'n_layers': 4, 'local_attn_every': 2}),
        ('mamba_xl', {'d_model': 128, 'n_layers': 2}),
        ('mamba_sandwich', {'d_model': 128, 'n_layers_mamba': 2, 'n_layers_local': 2}),
        ('mamba_gr', {'d_model': 128, 'n_layers': 2, 'gru_hidden': 64}),
    ])
    def test_different_sequence_lengths(self, model_type, kwargs, device):
        """Test model works with different sequence lengths."""
        model = create_model(model_type, **kwargs).to(device)

        for seq_len in [1, 5, 16, 32]:
            x = torch.randn(4, seq_len, 768, device=device)
            out = model(x)
            assert out.shape == (4, seq_len, 768)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
