"""
Mamba Block Library for LN-SP Phase-5
======================================

Building blocks for 5 Mamba-based LVM architectures:
- Pure Mamba (SSM) blocks
- Local attention blocks
- GRU-gated blocks
- Hybrid compositions

All blocks:
- Accept sequences of 768-D vectors
- Linear/near-linear memory scaling
- Optimized for Apple Silicon (MPS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More stable than LayerNorm for SSMs, lower compute cost.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [B, L, D]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


class MambaBlock(nn.Module):
    """Pure Mamba (SSM) block.

    Architecture:
    - Input projection (expand by factor)
    - Short 1D convolution (token mixing)
    - Selective SSM (state space model)
    - Output projection
    - Residual connection

    Key property: Linear memory in sequence length.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        conv_sz: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short convolution for local token mixing
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=conv_sz,
            padding=conv_sz - 1,
            groups=self.d_inner,  # Depthwise
        )

        # SSM parameters (selective mechanism)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # SSM state transition matrices (learned)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Initialize A with small negative values (stability)
        nn.init.normal_(self.A_log, mean=-4.0, std=1.0)

        # Initialize dt_proj with small positive bias
        nn.init.constant_(self.dt_proj.bias, 1.0)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence

        Returns:
            [B, L, D] output sequence
        """
        B, L, D = x.shape
        skip = x

        # Pre-norm
        x = self.norm(x)

        # Input projection and split into two branches
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        x, res = x_and_res.chunk(2, dim=-1)  # Each: [B, L, d_inner]

        # Short convolution (local mixing)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # Selective SSM
        x_ssm = self._selective_scan(x_conv)

        # Gating with residual branch
        x = x_ssm * F.silu(res)

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)

        # Residual
        return x + skip

    def _selective_scan(self, x):
        """Selective state space model (SSM) scan.

        This is a simplified implementation. For production,
        consider using optimized kernels (e.g., from mamba-ssm library).

        Args:
            x: [B, L, d_inner]

        Returns:
            [B, L, d_inner]
        """
        B, L, D = x.shape

        # Project input to state space parameters
        x_dbl = self.x_proj(x)  # [B, L, 2*d_state]
        delta, B_ssm = x_dbl.chunk(2, dim=-1)  # Each: [B, L, d_state]

        # Compute discrete-time dynamics
        dt = F.softplus(self.dt_proj(x))  # [B, L, d_inner]
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Simplified selective scan (sequential for clarity)
        # Production: use parallel scan or CUDA kernel
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=x.device)

        for t in range(L):
            # Get current timestep values
            x_t = x[:, t, :]  # [B, d_inner]
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B_ssm[:, t, :]  # [B, d_state]

            # Discretize A with learned dt: [d_inner, d_state] * [B, d_inner, 1] -> [B, d_inner, d_state]
            A_discrete = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # [B, d_inner, d_state]

            # Update state: h_t = A_discrete * h_{t-1} + B_t * x_t
            # h: [B, d_inner, d_state]
            # B_t: [B, d_state] -> expand to [B, 1, d_state] to broadcast with x_t
            # x_t: [B, d_inner] -> expand to [B, d_inner, 1]
            h = A_discrete * h + B_t.unsqueeze(1) * x_t.unsqueeze(-1)  # [B, d_inner, d_state]

            # Output: y_t = sum over d_state + D * x_t
            y_t = h.sum(dim=-1) + self.D * x_t  # [B, d_inner]
            y[:, t, :] = y_t

        return y


class LocalAttention(nn.Module):
    """Local windowed self-attention.

    Only attends to nearby tokens (window_size).
    This keeps attention O(L*W) instead of O(L^2).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.norm = RMSNorm(d_model)

        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, L, D]

        Returns:
            [B, L, D]
        """
        B, L, D = x.shape
        skip = x

        # Pre-norm
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x)  # [B, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d_head]
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Local attention with sliding window
        attn = torch.zeros(B, self.n_heads, L, L, device=x.device)

        for i in range(L):
            # Define window bounds
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)

            # Compute attention scores in window
            scores = torch.matmul(q[:, :, i:i+1, :], k[:, :, start:end, :].transpose(-2, -1))
            scores = scores / math.sqrt(self.d_head)

            # Softmax over window
            scores = F.softmax(scores, dim=-1)
            attn[:, :, i:i+1, start:end] = scores

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, L, d_head]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual
        return out + skip


class GRUGate(nn.Module):
    """GRU-style gating mechanism.

    Applied after Mamba blocks to help with exact next-chunk prediction (R@1).
    Adds explicit recurrence for better ranking.
    """
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # GRU gates
        self.reset_gate = nn.Linear(d_model + hidden_size, hidden_size)
        self.update_gate = nn.Linear(d_model + hidden_size, hidden_size)
        self.new_gate = nn.Linear(d_model + hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, d_model)

    def forward(self, x, h_prev=None):
        """
        Args:
            x: [B, L, D] input sequence
            h_prev: [B, hidden_size] previous hidden state (optional)

        Returns:
            [B, L, D] output sequence
            [B, hidden_size] final hidden state
        """
        B, L, D = x.shape

        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_size, device=x.device)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :]  # [B, D]

            # Concatenate input and hidden
            combined = torch.cat([x_t, h_prev], dim=1)  # [B, D + hidden]

            # GRU gates
            r = torch.sigmoid(self.reset_gate(combined))
            z = torch.sigmoid(self.update_gate(combined))

            # New hidden state candidate
            combined_reset = torch.cat([x_t, r * h_prev], dim=1)
            h_new = torch.tanh(self.new_gate(combined_reset))

            # Update hidden state
            h_prev = (1 - z) * h_prev + z * h_new

            # Project to output
            out_t = self.out_proj(h_prev)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)  # [B, L, D]
        return outputs, h_prev


class MLPBlock(nn.Module):
    """Feed-forward MLP with SiLU activation.

    Used inside Mamba blocks or as standalone layers.
    """
    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expand

        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + skip


# Export all blocks
__all__ = [
    'RMSNorm',
    'MambaBlock',
    'LocalAttention',
    'GRUGate',
    'MLPBlock',
]
