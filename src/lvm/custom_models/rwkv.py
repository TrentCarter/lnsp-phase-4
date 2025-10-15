import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class EmbeddingAdapter(nn.Module):
    """Novel adapter to bridge sentence-transformers to vec2text embeddings."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Custom mapping: affine transform + norm
        x = self.linear(x)
        x = self.norm(x)
        return x

class CustomRWKV(nn.Module):
    """RWKV with custom tweaks for compatibility."""
    def __init__(self, embed_dim=768, hidden_dim=512, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, hidden=None):
        x = self.adapter(x)  # Apply compatibility layer
        x, hidden = self.rnn(x, hidden)
        x = self.dropout(x)
        x = self.proj(x[:, -1, :])  # Last hidden state
        return torch.nn.functional.normalize(x, p=2, dim=-1)
