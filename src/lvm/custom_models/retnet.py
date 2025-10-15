import torch
import torch.nn as nn
import math

class EmbeddingAdapter(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class RetNetBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Simplified retention mechanism
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        x = torch.matmul(attn, v)
        x = self.proj(x)
        x = self.dropout(x)
        return residual + x

class CustomRetNet(nn.Module):
    def __init__(self, embed_dim=768, num_layers=4, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.layers = nn.ModuleList([RetNetBlock(embed_dim, hidden_dim) for _ in range(num_layers)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.adapter(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x[:, -1, :])  # Last timestep
        return torch.nn.functional.normalize(x, p=2, dim=-1)
