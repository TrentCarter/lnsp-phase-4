"""
Mamba LVM Package
=================

Five Mamba-based LVM architectures for LN-SP Phase-5.

Quick Start:
    >>> from app.lvm.mamba import create_model
    >>> model = create_model("mamba_s", d_model=256, n_layers=8)
    >>> output = model(input_vectors)  # [B, L, 768]

Available Models:
    - mamba_s: Pure SSM, Small (~1.3M params)
    - mamba_hybrid_local: Hybrid Mamba + Local-Attn (~2.6M params)
    - mamba_xl: Deeper/Wider Pure SSM (~5.8M params)
    - mamba_sandwich: Attn→SSM→Attn (~3.9M params)
    - mamba_gr: SSM + GRU Gate (~3.2M params)
"""

from .blocks import (
    RMSNorm,
    MambaBlock,
    LocalAttention,
    GRUGate,
    MLPBlock,
)

from .mamba import (
    BaseMambaLVM,
    MambaS,
    MambaHybridLocal,
    MambaXL,
    MambaSandwich,
    MambaGR,
    create_model,
    count_parameters,
    ModelType,
)

__all__ = [
    # Blocks
    'RMSNorm',
    'MambaBlock',
    'LocalAttention',
    'GRUGate',
    'MLPBlock',

    # Models
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
