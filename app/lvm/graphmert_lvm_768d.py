#!/usr/bin/env python3
"""
768-d Native GraphMERT-LVM Encoder
===================================

Key differences from standard GraphMERT:
1. NO projection layer (already in 768-d GTR-T5 space)
2. NO word embeddings (vector-native)
3. Direct transformer on 768-d vectors
4. Attention decay mechanism (λ=0.6)

This is a simplified version for benchmarking - NO KG leaves yet.
Just pure autoregressive vector prediction with GraphMERT-style attention.
"""

import torch
import torch.nn as nn
import math


class AttentionDecayMask(nn.Module):
    """
    Attention decay mask from GraphMERT paper (Sec 4.3)

    Exponential decay: λ^distance if distance > p
    - λ = 0.6 (base decay rate)
    - p = learnable threshold parameter
    """
    def __init__(self, lambda_decay=0.6, initial_threshold=2.0):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

    def forward(self, seq_len, device):
        """
        Build attention mask with exponential decay

        Returns:
            mask: (seq_len, seq_len) attention mask
        """
        # Distance matrix
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        distances = torch.abs(positions - positions.t())

        # Apply decay: -inf * (1 - λ^dist) for dist > p
        mask = torch.zeros(seq_len, seq_len, device=device)
        beyond_threshold = distances > self.threshold
        decay_factors = self.lambda_decay ** distances.float()
        mask[beyond_threshold] = -1e9 * (1.0 - decay_factors[beyond_threshold])

        return mask


class GraphMERTTransformerLayer(nn.Module):
    """
    Single transformer layer with attention decay

    Based on RoBERTa architecture (768-d native)
    """
    def __init__(
        self,
        d_model=768,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: (seq_len, seq_len) attention mask
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class GraphMERTLVM768D(nn.Module):
    """
    768-d Native GraphMERT-LVM Encoder

    Architecture:
    - Input: 5×768-d context vectors (no projection!)
    - 12 transformer layers (RoBERTa-style)
    - 8 attention heads
    - 2048 feed-forward hidden
    - Attention decay mask (λ=0.6)
    - Output: 768-d vector prediction

    Parameters: ~85M (similar to GraphMERT paper target)
    """
    def __init__(
        self,
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        lambda_decay=0.6,
    ):
        super().__init__()

        self.d_model = d_model

        # Attention decay mask
        self.decay_mask = AttentionDecayMask(lambda_decay)

        # Position embeddings (learned, like BERT)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)

        # Transformer layers (RoBERTa-style)
        self.layers = nn.ModuleList([
            GraphMERTTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head (predict 768-d vector)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights (RoBERTa-style)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, contexts, return_raw=False):
        """
        Forward pass

        Args:
            contexts: (batch, 5, 768) - 5 context vectors
            return_raw: If True, return (raw_pred, normalized_pred)

        Returns:
            predictions: (batch, 768) - predicted target vector
        """
        batch_size = contexts.size(0)
        device = contexts.device

        # Add position embeddings
        x = contexts + self.pos_embedding  # (batch, 5, 768)

        # Build attention decay mask
        attn_mask = self.decay_mask(5, device)  # (5, 5)

        # Transformer layers with decay mask
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # Pool sequence (mean pooling over 5 vectors)
        pooled = x.mean(dim=1)  # (batch, 768)

        # Output prediction
        pred_raw = self.output_head(pooled)  # (batch, 768)

        # Normalize to unit sphere (for cosine similarity)
        pred_normalized = torch.nn.functional.normalize(pred_raw, p=2, dim=1)

        if return_raw:
            return pred_raw, pred_normalized
        return pred_normalized

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_graphmert_lvm():
    """Test GraphMERT-LVM 768D architecture"""
    print("Testing GraphMERT-LVM 768D...")

    # Create model
    model = GraphMERTLVM768D(
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        lambda_decay=0.6
    )

    param_count = model.count_parameters()
    print(f"✓ Model created: {param_count:,} parameters")

    # Test forward pass
    batch_size = 4
    contexts = torch.randn(batch_size, 5, 768)

    predictions = model(contexts)
    print(f"✓ Forward pass: {contexts.shape} → {predictions.shape}")

    # Test return_raw
    pred_raw, pred_normalized = model(contexts, return_raw=True)
    print(f"✓ Return raw: raw={pred_raw.shape}, normalized={pred_normalized.shape}")

    # Verify normalization
    norms = torch.norm(pred_normalized, p=2, dim=1)
    print(f"✓ Normalized vectors: mean norm = {norms.mean():.6f} (should be ~1.0)")

    print("\n" + "="*60)
    print("GraphMERT-LVM 768D Test PASSED")
    print("="*60)


if __name__ == '__main__':
    test_graphmert_lvm()
