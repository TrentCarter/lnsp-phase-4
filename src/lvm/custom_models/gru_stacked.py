import torch
import torch.nn as nn

class EmbeddingAdapter(nn.Module):
    """Reused adapter for GRU Stacked."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class CustomGRUStacked(nn.Module):
    """Stacked GRU with custom tweaks."""
    def __init__(self, embed_dim=768, hidden_dim=256, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x, hidden=None):
        x = self.adapter(x)
        x, hidden = self.gru(x, hidden)
        x = self.proj(x[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=-1)
