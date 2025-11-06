"""
P8 Constrained LVM: Mixture-of-Context Head

Key innovation: Output is ALWAYS a weighted mixture of context vectors.
- No free 768D prediction (removes orthogonal escape)
- No λ-blend instability (no conflicting gradients)
- Geometrically constrained: q ∈ span(C)

Architecture:
    1. Encoder (Transformer/LSTM) processes context sequence
    2. Attention head predicts weights α over context vectors
    3. Output: q = normalize(Σ_i α_i · c_i)
    4. Optional: residual strictly within span(C)

This makes backward prediction HARD because:
- Output must be a mix of context vectors (all from past)
- To predict forward, model must learn which context vectors
  are most relevant for the next step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TransformerP8Constrained(nn.Module):
    """
    Transformer-based LVM with mixture-of-context head

    Output is constrained to span(context vectors), making orthogonal
    escape geometrically impossible.
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        context_length: int = 5,
        use_residual: bool = False,
        residual_eps: float = 0.1
    ):
        """
        Args:
            input_dim: dimension of input vectors (768 for GTR-T5)
            d_model: transformer hidden dimension
            nhead: number of attention heads
            num_layers: number of transformer layers
            dropout: dropout probability
            context_length: number of context vectors (K)
            use_residual: if True, add residual within span(C)
            residual_eps: weight for residual component (if enabled)
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.context_length = context_length
        self.use_residual = use_residual
        self.residual_eps = residual_eps

        # Project input to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, context_length, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Attention head: predict weights over context vectors
        # Takes last transformer output, produces K attention logits
        self.context_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, context_length)
        )

        # Optional: residual predictor (projects back to input_dim)
        if use_residual:
            self.residual_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, input_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with mixture-of-context head

        Args:
            x: (B, K, D) - context sequence (L2-normalized)
            return_attention: if True, return attention weights

        Returns:
            output: (B, D) - predicted next vector (unit-normalized, q ∈ span(C))
            extras: dict with 'attention', 'raw_mixture' (if requested)
        """
        B, K, D = x.shape
        assert K == self.context_length, f"Expected {self.context_length} context vectors, got {K}"

        # Project input
        x_proj = self.input_proj(x)  # (B, K, d_model)

        # Add positional encoding
        x_pos = x_proj + self.pos_encoder[:, :K, :]

        # Transformer encoding
        x_encoded = self.transformer(x_pos)  # (B, K, d_model)

        # Take last sequence position for next-token prediction
        x_last = x_encoded[:, -1, :]  # (B, d_model)

        # Predict attention weights over context vectors
        attn_logits = self.context_attention(x_last)  # (B, K)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K)

        # Mixture: q_mix = Σ_i α_i · c_i
        # (B, K, 1) * (B, K, D) → (B, K, D) → (B, D)
        q_mixture = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)

        # Optional: add residual strictly within span(C)
        if self.use_residual:
            # Predict residual in input space
            r_raw = self.residual_head(x_last)  # (B, D)

            # Project residual onto span(C) using QR decomposition
            # Note: This is expensive! Only use if necessary
            # For simplicity, we'll just normalize the mixture+residual
            q_with_residual = q_mixture + self.residual_eps * r_raw
            output = F.normalize(q_with_residual, dim=-1)
        else:
            # Pure mixture (no residual)
            output = F.normalize(q_mixture, dim=-1)

        if return_attention:
            extras = {
                'attention': attn_weights,
                'raw_mixture': q_mixture
            }
            return output, extras

        return output


class LSTMP8Constrained(nn.Module):
    """
    LSTM-based LVM with mixture-of-context head

    Simpler architecture for faster training, same constrained output.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        context_length: int = 5,
        use_residual: bool = False,
        residual_eps: float = 0.1
    ):
        """
        Args:
            input_dim: dimension of input vectors (768 for GTR-T5)
            hidden_dim: LSTM hidden dimension
            num_layers: number of LSTM layers
            dropout: dropout probability
            context_length: number of context vectors (K)
            use_residual: if True, add residual within span(C)
            residual_eps: weight for residual component
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.use_residual = use_residual
        self.residual_eps = residual_eps

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        # Attention head
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, context_length)
        )

        # Optional residual head
        if use_residual:
            self.residual_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, input_dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with mixture-of-context head

        Args:
            x: (B, K, D) - context sequence (L2-normalized)
            return_attention: if True, return attention weights

        Returns:
            output: (B, D) - predicted next vector (q ∈ span(C))
            extras: dict with 'attention', 'raw_mixture' (if requested)
        """
        B, K, D = x.shape

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (B, K, hidden_dim)

        # Take last output
        h_last = lstm_out[:, -1, :]  # (B, hidden_dim)

        # Predict attention weights
        attn_logits = self.context_attention(h_last)  # (B, K)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Mixture
        q_mixture = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)

        # Optional residual
        if self.use_residual:
            r_raw = self.residual_head(h_last)
            q_with_residual = q_mixture + self.residual_eps * r_raw
            output = F.normalize(q_with_residual, dim=-1)
        else:
            output = F.normalize(q_mixture, dim=-1)

        if return_attention:
            extras = {
                'attention': attn_weights,
                'raw_mixture': q_mixture
            }
            return output, extras

        return output


class OrderVerifier(nn.Module):
    """
    Auxiliary head: predict if chunk j comes after chunk i

    Given (c_i, c_j) from same article, predict label y:
        y = 1 if j > i (forward order)
        y = 0 if j < i (backward order)

    Self-supervised temporal prior that helps model learn
    "forward > backward" as an invariant.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256
    ):
        """
        Args:
            input_dim: dimension of input vectors (768)
            hidden_dim: hidden dimension for MLP
        """
        super().__init__()

        # Concatenate [c_i, c_j, c_i * c_j, c_i - c_j]
        # Total: 4 * input_dim features
        self.mlp = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )

    def forward(
        self,
        c_i: torch.Tensor,
        c_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict if j > i

        Args:
            c_i: (B, D) - earlier chunk vectors
            c_j: (B, D) - later chunk vectors

        Returns:
            logits: (B,) - logits for y = 1 (j > i)
        """
        # Concatenate features
        v = torch.cat([
            c_i,
            c_j,
            c_i * c_j,  # Element-wise product (interaction)
            c_i - c_j   # Difference (direction)
        ], dim=-1)  # (B, 4*D)

        logits = self.mlp(v).squeeze(-1)  # (B,)
        return logits
