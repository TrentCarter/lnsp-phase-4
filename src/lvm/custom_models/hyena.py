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

class HyenaOperator(nn.Module):
    def __init__(self, dim, order=3):
        super().__init__()
        self.order = order
        self.filters = nn.Parameter(torch.randn(order, dim, dim))
        
    def forward(self, x):
        # Simplified: use element-wise multiplication for speed
        B, T, C = x.shape
        out = torch.zeros_like(x)
        for i in range(self.order):
            filter = self.filters[i]
            out += torch.matmul(x, filter)
        return out

class CustomHyena(nn.Module):
    def __init__(self, embed_dim=768, num_layers=4, order=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter = EmbeddingAdapter(embed_dim)
        self.layers = nn.ModuleList([HyenaOperator(embed_dim, order) for _ in range(num_layers)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def generate(self, seed_vectors, max_length=5, temperature=1.0):
        """Generative mode: autoregressively generate next vectors."""
        generated = list(seed_vectors)
        for _ in range(max_length):
            input_seq = torch.stack(generated[-5:]).unsqueeze(0)  # (1, 5, 768)
            with torch.no_grad():
                next_vec = self.forward(input_seq)  # (1, 768)
                if next_vec is not None:
                    # Add noise for diversity
                    next_vec = next_vec + temperature * torch.randn_like(next_vec)
                    generated.append(next_vec.squeeze(0))
                else:
                    break
        return torch.stack(generated)
        
    def forward(self, x):
        x = self.adapter(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x[:, -1, :])
        return torch.nn.functional.normalize(x, p=2, dim=-1)
