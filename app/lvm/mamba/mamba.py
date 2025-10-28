"""
Mamba LVM Models for LN-SP Phase-5
===================================

Five Mamba-based architectures for next-vector prediction:
- Model A: Mamba-S (Pure SSM, Small)
- Model B: Mamba-H (Hybrid 80/20: Mamba + Local-Attn)
- Model C: Mamba-XL (Deeper/Wider Pure SSM)
- Model D: Mamba-Sandwich (Attn→SSM→Attn)
- Model E: Mamba-GR (SSM + GRU Gate)

All models:
- Input: sequences of 768-D vectors
- Output: 768-D next-vector prediction (L2-normalized)
- Linear/near-linear memory scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from .blocks import MambaBlock, LocalAttention, GRUGate, MLPBlock, RMSNorm


ModelType = Literal["mamba_s", "mamba_hybrid_local", "mamba_xl", "mamba_sandwich", "mamba_gr"]


class BaseMambaLVM(nn.Module):
    """Base class for all Mamba LVM models.

    Provides:
    - Input embedding (768 → d_model)
    - Output head (d_model → 768 with L2 norm)
    - Optional alignment head (residual projection)
    """

    def __init__(
        self,
        d_model: int,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_alignment_head = use_alignment_head
        self.alignment_alpha = alignment_alpha

        # Input projection: 768 → d_model
        self.input_proj = nn.Linear(768, d_model)

        # Output head: d_model → 768
        self.output_head = nn.Linear(d_model, 768, bias=False)

        # Optional alignment head (residual)
        if use_alignment_head:
            self.alignment_head = nn.Linear(768, 768, bias=False)

    def _build_layers(self):
        """Override this to build model-specific layers."""
        raise NotImplementedError

    def _forward_layers(self, x):
        """Override this to implement layer-specific forward pass."""
        raise NotImplementedError

    def forward(self, x):
        """
        Args:
            x: [B, L, 768] input sequence of vectors

        Returns:
            [B, L, 768] predicted next vectors (L2-normalized)
        """
        # Project to internal dimension
        x = self.input_proj(x)  # [B, L, d_model]

        # Model-specific layers
        x = self._forward_layers(x)  # [B, L, d_model]

        # Output projection
        out = self.output_head(x)  # [B, L, 768]

        # Optional alignment head (residual)
        if self.use_alignment_head:
            residual = self.alignment_head(out)
            out = (1 - self.alignment_alpha) * out + self.alignment_alpha * residual

        # L2 normalize
        out = F.normalize(out, p=2, dim=-1)

        return out


class MambaS(BaseMambaLVM):
    """Model A: Mamba-S (Pure SSM, Small)

    - 8 pure Mamba blocks
    - d_model=256, d_state=128, expand=2
    - ~1.3M params
    - Target: R@5 52-54%, P95 ≤ 1.1ms
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 128,
        conv_sz: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__(d_model, use_alignment_head, alignment_alpha)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, conv_sz, expand, dropout)
            for _ in range(n_layers)
        ])

    def _forward_layers(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MambaHybridLocal(BaseMambaLVM):
    """Model B: Mamba-H (Hybrid 80/20: Mamba + Local-Attn)

    - 12 layers total: 3×(Mamba×3 → Local-Attn)
    - d_model=320, d_state=128, local window=8
    - ~2.6M params
    - Target: R@5 54-56%, P95 ≤ 1.3ms
    """

    def __init__(
        self,
        d_model: int = 320,
        n_layers: int = 12,
        d_state: int = 128,
        conv_sz: int = 4,
        expand: int = 2,
        local_attn_win: int = 8,
        local_attn_every: int = 4,
        n_heads: int = 4,
        dropout: float = 0.05,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__(d_model, use_alignment_head, alignment_alpha)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if (i + 1) % local_attn_every == 0:
                # Local attention every 4th layer
                self.layers.append(
                    LocalAttention(d_model, n_heads, local_attn_win, dropout)
                )
            else:
                # Mamba blocks
                self.layers.append(
                    MambaBlock(d_model, d_state, conv_sz, expand, dropout)
                )

    def _forward_layers(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MambaXL(BaseMambaLVM):
    """Model C: Mamba-XL (Deeper/Wider Pure SSM)

    - 16 pure Mamba blocks
    - d_model=384, d_state=192, expand=2
    - ~5.8M params
    - Target: R@5 55-57%, P95 ≤ 1.45ms
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 16,
        d_state: int = 192,
        conv_sz: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__(d_model, use_alignment_head, alignment_alpha)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, conv_sz, expand, dropout)
            for _ in range(n_layers)
        ])

    def _forward_layers(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MambaSandwich(BaseMambaLVM):
    """Model D: Mamba-Sandwich (Attn→SSM→Attn)

    - Front: 2 Local-Attn layers
    - Trunk: 8 Mamba blocks
    - Back: 2 Local-Attn layers
    - d_model=320, d_state=160, window=8
    - ~3.9M params
    - Target: R@5 56-59%, P95 ≤ 1.45ms
    """

    def __init__(
        self,
        d_model: int = 320,
        n_layers_mamba: int = 8,
        n_layers_local: int = 4,  # Total (2 front + 2 back)
        d_state: int = 160,
        conv_sz: int = 4,
        expand: int = 2,
        local_attn_win: int = 8,
        n_heads: int = 4,
        dropout: float = 0.05,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__(d_model, use_alignment_head, alignment_alpha)

        assert n_layers_local % 2 == 0, "n_layers_local must be even (front+back)"
        n_front = n_layers_local // 2
        n_back = n_layers_local // 2

        # Front: Local attention
        self.front_layers = nn.ModuleList([
            LocalAttention(d_model, n_heads, local_attn_win, dropout)
            for _ in range(n_front)
        ])

        # Trunk: Mamba blocks
        self.trunk_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, conv_sz, expand, dropout)
            for _ in range(n_layers_mamba)
        ])

        # Back: Local attention
        self.back_layers = nn.ModuleList([
            LocalAttention(d_model, n_heads, local_attn_win, dropout)
            for _ in range(n_back)
        ])

    def _forward_layers(self, x):
        # Front: attention
        for layer in self.front_layers:
            x = layer(x)

        # Trunk: Mamba
        for layer in self.trunk_layers:
            x = layer(x)

        # Back: attention
        for layer in self.back_layers:
            x = layer(x)

        return x


class MambaGR(BaseMambaLVM):
    """Model E: Mamba-GR (SSM + GRU Gate)

    - 10 layers: Mamba → GRU gate (per block)
    - d_model=288, d_state=144, gru_hidden=256
    - ~3.2M params
    - Target: R@5 53-55%, R@1 +0.3-0.8pp, P95 ≤ 1.4ms
    """

    def __init__(
        self,
        d_model: int = 288,
        n_layers: int = 10,
        d_state: int = 144,
        conv_sz: int = 4,
        expand: int = 2,
        gru_hidden: int = 256,
        dropout: float = 0.05,
        use_alignment_head: bool = False,
        alignment_alpha: float = 0.25,
    ):
        super().__init__(d_model, use_alignment_head, alignment_alpha)

        # Mamba + GRU blocks
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, conv_sz, expand, dropout)
            for _ in range(n_layers)
        ])

        self.gru_gates = nn.ModuleList([
            GRUGate(d_model, gru_hidden)
            for _ in range(n_layers)
        ])

    def _forward_layers(self, x):
        h_prev = None
        for mamba, gru in zip(self.mamba_layers, self.gru_gates):
            # Mamba block
            x = mamba(x)

            # GRU gate
            x, h_prev = gru(x, h_prev)

        return x


def create_model(
    model_type: ModelType,
    **kwargs,
) -> BaseMambaLVM:
    """Factory function to create Mamba models.

    Args:
        model_type: One of "mamba_s", "mamba_hybrid_local", "mamba_xl",
                    "mamba_sandwich", "mamba_gr"
        **kwargs: Model-specific parameters

    Returns:
        Initialized model

    Example:
        >>> model = create_model("mamba_s", d_model=256, n_layers=8)
        >>> model = create_model("mamba_sandwich", d_model=320, n_layers_mamba=8)
    """
    models = {
        "mamba_s": MambaS,
        "mamba_hybrid_local": MambaHybridLocal,
        "mamba_xl": MambaXL,
        "mamba_sandwich": MambaSandwich,
        "mamba_gr": MambaGR,
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose from: {list(models.keys())}"
        )

    return models[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Export all
__all__ = [
    'BaseMambaLVM',
    'MambaS',
    'MambaHybridLocal',
    'MambaXL',
    'MambaSandwich',
    'MambaGR',
    'create_model',
    'count_parameters',
    'ModelType',
]
