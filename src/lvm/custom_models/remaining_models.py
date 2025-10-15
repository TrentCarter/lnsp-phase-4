import torch
import torch.nn as nn

class EmbeddingAdapter(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x

class CustomDistilGPT2(nn.Module):
    def __init__(self, embed_dim=768, num_layers=6, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, hidden_dim, batch_first=True),
            num_layers
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.adapter(x)
        x = self.transformer(x)
        x = self.proj(x[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=-1)

class CustomPerformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.adapter(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.proj(x[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=-1)

class CustomLinformer(nn.Module):
    def __init__(self, embed_dim=768, seq_len=5, k=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.proj = nn.Linear(seq_len * embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.adapter(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

class CustomS4(nn.Module):
    def __init__(self, embed_dim=768, state_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(embed_dim, state_dim))
        self.C = nn.Parameter(torch.randn(state_dim, embed_dim))
        
    def forward(self, x):
        x = self.adapter(x)
        # Simplified S4 recurrence
        h = torch.zeros(x.size(0), self.A.size(0), device=x.device)
        for t in range(x.size(1)):
            h = torch.matmul(h, self.A) + torch.matmul(x[:, t, :], self.B)
        x = torch.matmul(h, self.C)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

class CustomHybridMambaAttn(nn.Module):
    def __init__(self, embed_dim=768, mamba_layers=3, attn_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.mamba = nn.GRU(embed_dim, embed_dim, mamba_layers, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        x = self.adapter(x)
        x, _ = self.mamba(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.proj(x[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=-1)
