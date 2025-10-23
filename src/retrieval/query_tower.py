from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryTower(nn.Module):
    """GRU → 768D + LayerNorm + L2‑normalize.
    Expects input: (B, T, 768) float32 unit‑norm vectors.
    """
    def __init__(self, hidden_size: int = 768, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=False)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (B, T, 768)
        out, h = self.gru(x)  # out: (B, T, H)
        # mean‑pool over T
        q = out.mean(dim=1)
        q = self.ln(q)
        # L2 normalize
        q = F.normalize(q, p=2, dim=-1)
        return q  # (B, 768)
