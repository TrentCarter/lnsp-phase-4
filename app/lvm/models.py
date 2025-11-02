#!/usr/bin/env python3
"""
LVM Architecture Collection for Vector-Native Sequence Prediction
==================================================================

Four architectures for next-vector prediction from context:
1. LSTM Baseline - Simple recurrent baseline
2. GRU Stack - Stacked GRU with residuals (Mamba2 fallback)
3. Transformer - Full self-attention with causal mask
4. Attention Mixture Network (AMN) - Residual learning over linear baseline

All models:
- Input: [batch, context_len, 768]
- Output: [batch, 768] (L2 normalized)
- Loss: MSE between predicted and target vectors
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Extended context models (Experiment A & B)
try:
    # Try relative import first (when running from app/lvm/)
    from .hierarchical_gru import HierarchicalGRU
    from .memory_gru import MemoryAugmentedGRU
except ImportError:
    try:
        # Try absolute import (when running from project root)
        from app.lvm.hierarchical_gru import HierarchicalGRU
        from app.lvm.memory_gru import MemoryAugmentedGRU
    except (ImportError, ModuleNotFoundError):
        # If not available, define placeholders (not used by main 4 models)
        HierarchicalGRU = None
        MemoryAugmentedGRU = None


# ============================================================================
# 1. LSTM BASELINE
# ============================================================================

class LSTMBaseline(nn.Module):
    """Simple LSTM baseline for next-vector prediction"""

    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2, dropout=0.2, output_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_raw: bool = False):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            [batch, 768] - predicted next vector (L2 normalized)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_dim]

        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # Project to output
        raw = self.output_proj(last_hidden)  # [batch, 768]

        # Normalize
        cos = F.normalize(raw, p=2, dim=-1)

        if return_raw:
            return raw, cos
        return cos

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 2. GRU STACK (Mamba2 fallback)
# ============================================================================

class GRUBlock(nn.Module):
    """Single GRU block with residual connection"""

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # GRU forward
        gru_out, _ = self.gru(x)  # [batch, seq_len, d_model]

        # Residual connection
        if self.dropout is not None:
            gru_out = self.dropout(gru_out)
        out = self.norm(x + gru_out)

        return out


class GRUStack(nn.Module):
    """Stacked GRU with residuals (Mamba2 fallback)"""

    def __init__(self, input_dim=768, d_model=512, num_layers=4, dropout=0.0, output_dim=768):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stacked GRU blocks
        self.blocks = nn.ModuleList([
            GRUBlock(d_model, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, return_raw: bool = False):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            [batch, 768] - predicted next vector (L2 normalized)
        """
        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Apply GRU blocks
        for block in self.blocks:
            x = block(x)  # [batch, seq_len, d_model]

        # Take last timestep
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Project to output
        raw = self.output_proj(last_hidden)  # [batch, 768]

        # Normalize
        cos = F.normalize(raw, p=2, dim=-1)

        if return_raw:
            return raw, cos
        return cos

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 3. TRANSFORMER
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerVectorPredictor(nn.Module):
    """Transformer decoder for next-vector prediction"""

    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dropout=0.1, output_dim=768):
        super().__init__()
        self.d_model = d_model

        # Input projection (can be 768 or 769 if positional encoding used)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection (always outputs 768D target vectors)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x, return_raw: bool = False):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            [batch, 768] - predicted next vector (L2 normalized)
        """
        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer forward
        x = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask
        )  # [batch, seq_len, d_model]

        # Take last timestep
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Project to output
        raw = self.head(last_hidden)  # [batch, 768]

        # Normalize
        cos = F.normalize(raw, p=2, dim=-1)

        if return_raw:
            return raw, cos
        return cos

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 4. ATTENTION MIXTURE NETWORK (AMN) - NEW!
# ============================================================================

class AttentionMixtureNetwork(nn.Module):
    """
    Attention Mixture Network for Latent Space Next-Vector Prediction

    Key Innovation: Residual learning over linear baseline
    - Computes linear average of context (strong baseline: 0.546 cosine)
    - Uses lightweight attention to learn better mixture weights
    - Predicts residual correction to baseline
    - Output = normalize(baseline + residual)

    Why This Works for LNSP:
    - Wikipedia chunks are topically coherent (linear avg is strong)
    - Model learns when to deviate from average (topic shifts)
    - Residual forces model to beat baseline (not reinvent wheel)
    - Attention weights interpretable (can visualize semantic flow)

    Architecture:
    - Much simpler than transformer (~2M params vs 17M)
    - Single-head attention (enough for 5-vector context)
    - Small residual MLP (forces minimal correction)
    - Explicit baseline computation (interpretable)
    """

    def __init__(self, input_dim=768, d_model=256, hidden_dim=512, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # Context encoder (maps each position to d_model)
        self.context_encoder = nn.Linear(input_dim, d_model)

        # Query encoder (maps baseline to d_model for attention)
        self.query_encoder = nn.Linear(input_dim, d_model)

        # Attention scale
        self.scale = math.sqrt(d_model)

        # Residual predictor (corrects baseline)
        self.residual_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # [baseline, weighted_context]
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x, return_raw: bool = False, return_attention: bool = False):
        """
        Args:
            x: [batch, seq_len, 768] - context vectors
            return_raw: If True, return (raw, normalized)
            return_attention: If True, return attention weights
        Returns:
            [batch, 768] - predicted next vector (L2 normalized)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Compute linear baseline (strong baseline: 0.546 cosine)
        baseline = x.mean(dim=1)  # [batch, 768]

        # 2. Encode context and query for attention
        context_encoded = self.context_encoder(x)  # [batch, seq_len, d_model]
        query_encoded = self.query_encoder(baseline).unsqueeze(1)  # [batch, 1, d_model]

        # 3. Compute attention weights (learn better mixture than uniform)
        # Attention scores: query @ keys^T / sqrt(d_model)
        attn_scores = torch.bmm(
            query_encoded,
            context_encoded.transpose(1, 2)
        ) / self.scale  # [batch, 1, seq_len]

        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 1, seq_len]

        # 4. Compute weighted context (learned mixture)
        weighted_context = torch.bmm(
            attn_weights,
            x
        ).squeeze(1)  # [batch, 768]

        # 5. Predict residual correction
        # Concatenate baseline and weighted context
        residual_input = torch.cat([baseline, weighted_context], dim=-1)  # [batch, 1536]
        residual = self.residual_net(residual_input)  # [batch, 768]

        # 6. Add residual to baseline and normalize
        raw = baseline + residual  # [batch, 768]
        cos = F.normalize(raw, p=2, dim=-1)

        if return_attention:
            return cos, attn_weights.squeeze(1)  # [batch, seq_len]

        if return_raw:
            return raw, cos

        return cos

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, **kwargs):
    """
    Create LVM model by type.

    Args:
        model_type: One of ['lstm', 'gru', 'transformer', 'amn']
        **kwargs: Model-specific hyperparameters

    Returns:
        Model instance
    """
    if model_type == 'lstm':
        return LSTMBaseline(
            input_dim=kwargs.get('input_dim', 768),
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.2)
        )

    elif model_type == 'gru':
        return GRUStack(
            input_dim=kwargs.get('input_dim', 768),
            d_model=kwargs.get('d_model', 512),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.0)
        )

    elif model_type == 'transformer':
        return TransformerVectorPredictor(
            input_dim=kwargs.get('input_dim', 768),
            d_model=kwargs.get('d_model', 512),
            nhead=kwargs.get('nhead', 8),
            num_layers=kwargs.get('num_layers', 4),
            dropout=kwargs.get('dropout', 0.1)
        )

    elif model_type == 'amn':
        return AttentionMixtureNetwork(
            input_dim=kwargs.get('input_dim', 768),
            d_model=kwargs.get('d_model', 256),
            hidden_dim=kwargs.get('hidden_dim', 512)
        )

    elif model_type == 'hierarchical_gru':
        return HierarchicalGRU(
            d_model=kwargs.get('d_model', 768),
            hidden_dim=kwargs.get('hidden_dim', 512),
            chunk_size=kwargs.get('chunk_size', 10),
            num_chunks=kwargs.get('num_chunks', 10),
            local_layers=kwargs.get('local_layers', 2),
            global_layers=kwargs.get('global_layers', 2)
        )

    elif model_type == 'memory_gru':
        return MemoryAugmentedGRU(
            d_model=kwargs.get('d_model', 768),
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_layers=kwargs.get('num_layers', 4),
            memory_slots=kwargs.get('memory_slots', 2048),
            use_memory_write=kwargs.get('use_memory_write', True)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: lstm, gru, transformer, amn, hierarchical_gru, memory_gru")


# ============================================================================
# MODEL SPECS (for documentation/comparison)
# ============================================================================

MODEL_SPECS = {
    'lstm': {
        'name': 'LSTM Baseline',
        'params': '~1.6M',
        'description': 'Simple 2-layer LSTM baseline',
        'pros': ['Fast training', 'Low memory', 'Good for sequential data'],
        'cons': ['Limited long-range dependencies', 'Sequential bottleneck'],
        'best_for': 'Quick baseline experiments'
    },

    'gru': {
        'name': 'GRU Stack (Mamba2 fallback)',
        'params': '~2.4M',
        'description': 'Stacked GRU with residuals',
        'pros': ['Faster than LSTM', 'Residual connections', 'Stable training'],
        'cons': ['Still sequential', 'No attention mechanism'],
        'best_for': 'Strong recurrent baseline'
    },

    'transformer': {
        'name': 'Transformer Decoder',
        'params': '~17.8M',
        'description': 'Full self-attention with causal mask',
        'pros': ['Parallel training', 'Long-range dependencies', 'State-of-art'],
        'cons': ['Large model', 'Slower inference', 'May overfit'],
        'best_for': 'Maximum capacity, production systems'
    },

    'amn': {
        'name': 'Attention Mixture Network',
        'params': '~2.1M',
        'description': 'Residual learning over linear baseline',
        'pros': ['Beats baseline by design', 'Interpretable', 'Efficient'],
        'cons': ['Single attention head', 'Simpler than transformer'],
        'best_for': 'LNSP latent space, interpretable predictions'
    },

    'hierarchical_gru': {
        'name': 'Hierarchical GRU (Extended Context)',
        'params': '~8-10M',
        'description': 'Two-level processing: local chunks + global attention',
        'pros': ['100-vector context (2k tokens)', 'Hierarchical processing', 'Scales to 1000+ vectors'],
        'cons': ['More complex than baseline', 'Requires extended context data'],
        'best_for': 'Extended context experiments (Experiment A)'
    },

    'memory_gru': {
        'name': 'Memory-Augmented GRU (Extended Context)',
        'params': '~10-12M (+ 1.5M memory bank)',
        'description': 'GRU with external memory bank (2,048 slots)',
        'pros': ['Persistent knowledge', 'Content-based addressing', 'TMD-aware routing'],
        'cons': ['Memory overhead', 'Complex training dynamics'],
        'best_for': 'Extended context experiments (Experiment B)'
    }
}


if __name__ == '__main__':
    # Test all models
    print("=" * 80)
    print("LVM Model Collection - Architecture Test")
    print("=" * 80)
    print()

    batch_size = 4
    seq_len = 5
    input_dim = 768

    x = torch.randn(batch_size, seq_len, input_dim)

    for model_type in ['lstm', 'gru', 'transformer', 'amn']:
        print(f"Testing {model_type.upper()}...")
        model = create_model(model_type)

        # Forward pass
        with torch.no_grad():
            if model_type == 'amn':
                output, attn = model(x, return_attention=True)
                print(f"  Output shape: {output.shape}")
                print(f"  Attention shape: {attn.shape}")
            else:
                output = model(x)
                print(f"  Output shape: {output.shape}")

        # Check normalization
        norms = output.norm(dim=-1)
        print(f"  Output norms: {norms.mean():.4f} ± {norms.std():.4f}")

        # Parameters
        params = model.count_parameters()
        print(f"  Parameters: {params:,}")
        print()

    print("=" * 80)
    print("All models tested successfully!")
    print("=" * 80)
