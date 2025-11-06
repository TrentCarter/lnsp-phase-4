"""
P7 "Directional Ranker" Model Architecture with Semantic Anchoring

Key features:
1. Unit-sphere output normalization (prevents magnitude drift)
2. Semantic anchoring: blend output with context subspace
3. Learnable anchor weight λ ∈ [0.6, 0.9]
4. Optional attention-weighted context blending

Author: Claude Code
Date: 2025-11-04
Status: P7 Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TransformerP7Ranker(nn.Module):
    """
    Transformer-based LVM with semantic anchoring for P7 ranking objective

    Architecture:
    - Input projection: 768D → d_model
    - Positional encoding
    - Transformer encoder (4 layers, 8 heads)
    - Output head: d_model → 768D
    - Unit sphere normalization
    - Semantic anchoring with learnable blend weight
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        anchor_lambda_init: float = 0.8,
        anchor_lambda_learnable: bool = True,
        anchor_lambda_min: float = 0.6,
        anchor_lambda_max: float = 0.9,
        use_attention_anchor: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.anchor_lambda_min = anchor_lambda_min
        self.anchor_lambda_max = anchor_lambda_max
        self.use_attention_anchor = use_attention_anchor

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learned)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model),
            requires_grad=False
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )

        # Semantic anchoring: learnable blend weight
        if anchor_lambda_learnable:
            # Initialize with logit transform to enable unconstrained optimization
            # logit(λ) where λ ∈ [min, max]
            # λ = min + (max - min) * sigmoid(logit_lambda)
            init_logit = self._inverse_sigmoid(
                (anchor_lambda_init - anchor_lambda_min) / (anchor_lambda_max - anchor_lambda_min)
            )
            self.anchor_lambda_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('anchor_lambda', torch.tensor(anchor_lambda_init))

        # Optional attention for context weighting
        if use_attention_anchor:
            self.context_attention = nn.Linear(d_model, 1)

    def _inverse_sigmoid(self, x: float) -> float:
        """Inverse sigmoid for parameter initialization"""
        x = max(min(x, 0.9999), 0.0001)  # Clamp to avoid infinities
        return torch.log(torch.tensor(x / (1.0 - x))).item()

    def get_anchor_lambda(self) -> float:
        """Get current anchor blend weight λ ∈ [min, max]"""
        if hasattr(self, 'anchor_lambda_logit'):
            # Transform from logit to [min, max] range
            sigmoid_val = torch.sigmoid(self.anchor_lambda_logit)
            lambda_val = self.anchor_lambda_min + (self.anchor_lambda_max - self.anchor_lambda_min) * sigmoid_val
            return lambda_val.item() if isinstance(lambda_val, torch.Tensor) else float(lambda_val)
        else:
            return self.anchor_lambda.item() if isinstance(self.anchor_lambda, torch.Tensor) else float(self.anchor_lambda)

    def forward(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with semantic anchoring

        Args:
            x: (B, K, D) - input context sequence (e.g., 5 chunks of 768D vectors)
            return_raw: if True, return raw prediction before anchoring
            return_attention: if True, return attention weights over context

        Returns:
            output: (B, D) - predicted next vector (unit-normalized, anchored)
            extras: dict with 'raw', 'anchor_lambda', 'attention' (if requested)
        """
        B, K, D = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (B, K, d_model)

        # Add positional encoding
        x_pos = x_proj + self.pos_encoder[:, :K, :]

        # Transformer encoding
        x_encoded = self.transformer(x_pos)  # (B, K, d_model)

        # Take last sequence position for next-token prediction
        x_last = x_encoded[:, -1, :]  # (B, d_model)

        # Output head
        output_raw = self.head(x_last)  # (B, output_dim)

        # Normalize to unit sphere (prevents magnitude drift)
        output_raw_norm = F.normalize(output_raw, dim=-1)

        # Semantic anchoring: blend with context subspace
        lambda_blend = self.get_anchor_lambda()

        if self.use_attention_anchor:
            # Attention-weighted context centroid
            attn_logits = self.context_attention(x_encoded).squeeze(-1)  # (B, K)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K)
            context_weighted = torch.bmm(
                attn_weights.unsqueeze(1),  # (B, 1, K)
                x  # (B, K, D) - use original input vectors
            ).squeeze(1)  # (B, D)
            context_norm = F.normalize(context_weighted, dim=-1)
        else:
            # Simple mean context centroid
            context_centroid = x.mean(dim=1)  # (B, D)
            context_norm = F.normalize(context_centroid, dim=-1)

        # Blend: q' = norm(λ·q_raw + (1-λ)·c)
        output_blended = lambda_blend * output_raw_norm + (1.0 - lambda_blend) * context_norm
        output_anchored = F.normalize(output_blended, dim=-1)

        if return_raw or return_attention:
            extras = {
                'raw': output_raw_norm,
                'anchor_lambda': lambda_blend.item() if torch.is_tensor(lambda_blend) else lambda_blend,
                'context_centroid': context_norm
            }
            if return_attention and self.use_attention_anchor:
                extras['attention'] = attn_weights
            return output_anchored, extras

        return output_anchored


class LSTMP7Ranker(nn.Module):
    """
    LSTM-based LVM with semantic anchoring for P7 ranking objective

    Simpler architecture for faster training, same anchoring principle.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        anchor_lambda_init: float = 0.8,
        anchor_lambda_learnable: bool = True,
        anchor_lambda_min: float = 0.6,
        anchor_lambda_max: float = 0.9
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.anchor_lambda_min = anchor_lambda_min
        self.anchor_lambda_max = anchor_lambda_max

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        # Semantic anchoring: learnable blend weight
        if anchor_lambda_learnable:
            init_logit = self._inverse_sigmoid(
                (anchor_lambda_init - anchor_lambda_min) / (anchor_lambda_max - anchor_lambda_min)
            )
            self.anchor_lambda_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self.register_buffer('anchor_lambda', torch.tensor(anchor_lambda_init))

    def _inverse_sigmoid(self, x: float) -> float:
        """Inverse sigmoid for parameter initialization"""
        x = max(min(x, 0.9999), 0.0001)
        return torch.log(torch.tensor(x / (1.0 - x))).item()

    def get_anchor_lambda(self) -> float:
        """Get current anchor blend weight λ ∈ [min, max]"""
        if hasattr(self, 'anchor_lambda_logit'):
            sigmoid_val = torch.sigmoid(self.anchor_lambda_logit)
            lambda_val = self.anchor_lambda_min + (self.anchor_lambda_max - self.anchor_lambda_min) * sigmoid_val
            return lambda_val.item() if isinstance(lambda_val, torch.Tensor) else float(lambda_val)
        else:
            return self.anchor_lambda.item() if isinstance(self.anchor_lambda, torch.Tensor) else float(self.anchor_lambda)

    def forward(
        self,
        x: torch.Tensor,
        return_raw: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with semantic anchoring

        Args:
            x: (B, K, D) - input context sequence
            return_raw: if True, return raw prediction before anchoring

        Returns:
            output: (B, D) - predicted next vector (unit-normalized, anchored)
            extras: dict with 'raw', 'anchor_lambda' (if requested)
        """
        B, K, D = x.shape

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, K, hidden_dim)

        # Take last time step
        x_last = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Output head
        output_raw = self.head(x_last)  # (B, output_dim)

        # Normalize to unit sphere
        output_raw_norm = F.normalize(output_raw, dim=-1)

        # Semantic anchoring
        lambda_blend = self.get_anchor_lambda()
        context_centroid = x.mean(dim=1)  # (B, D)
        context_norm = F.normalize(context_centroid, dim=-1)

        # Blend
        output_blended = lambda_blend * output_raw_norm + (1.0 - lambda_blend) * context_norm
        output_anchored = F.normalize(output_blended, dim=-1)

        if return_raw:
            extras = {
                'raw': output_raw_norm,
                'anchor_lambda': lambda_blend.item() if torch.is_tensor(lambda_blend) else lambda_blend,
                'context_centroid': context_norm
            }
            return output_anchored, extras

        return output_anchored


def create_p7_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    Factory function for creating P7 ranker models

    Args:
        model_type: 'transformer' or 'lstm'
        **kwargs: model-specific parameters

    Returns:
        model: P7 ranker model with semantic anchoring
    """
    if model_type == 'transformer':
        return TransformerP7Ranker(**kwargs)
    elif model_type == 'lstm':
        return LSTMP7Ranker(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
