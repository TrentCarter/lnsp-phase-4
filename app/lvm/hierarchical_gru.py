"""
Hierarchical GRU for Extended Context

Two-level architecture:
- Level 1 (Local): Process 10 chunks of 10 vectors each
- Level 2 (Global): Attend over chunk summaries with GRU

Total context: 100 vectors (2,000 tokens) vs baseline 5 vectors (100 tokens)

Architecture:
    Input: [batch, 100, 768]
    ├─> Chunk into 10 groups of [batch, 10, 768]
    ├─> Local GRU: [batch, 10, 768] -> [batch, 512] (chunk summary)
    ├─> Global GRU: [batch, 10, 512] -> [batch, 512] (context summary)
    └─> Output Projection: [batch, 512] -> [batch, 768]

Benefits:
- Hierarchical processing reduces quadratic attention cost
- Local-global structure mirrors document structure (paragraphs -> document)
- Can extend to 1000+ vectors by adding more hierarchy levels

Created: 2025-10-19 (Extended Context Experiments - Experiment A)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalChunkEncoder(nn.Module):
    """
    Encode a local chunk of vectors into a summary vector.

    Args:
        d_model: Input/output dimension (768 for GTR-T5)
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
    """
    def __init__(self, d_model=768, hidden_dim=512, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, chunk):
        """
        Args:
            chunk: [batch, chunk_size, d_model] (e.g., [batch, 10, 768])

        Returns:
            summary: [batch, hidden_dim] (e.g., [batch, 512])
        """
        # GRU processes sequence
        output, hidden = self.gru(chunk)  # output: [batch, chunk_size, hidden_dim]

        # Use final hidden state as chunk summary
        summary = hidden[-1]  # [batch, hidden_dim]

        # Normalize
        summary = self.norm(summary)

        return summary


class GlobalContextEncoder(nn.Module):
    """
    Encode chunk summaries into global context representation.

    Args:
        hidden_dim: Dimension of chunk summaries
        num_layers: Number of GRU layers
    """
    def __init__(self, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, chunk_summaries):
        """
        Args:
            chunk_summaries: [batch, num_chunks, hidden_dim] (e.g., [batch, 10, 512])

        Returns:
            global_context: [batch, hidden_dim]
        """
        # GRU over chunk summaries
        output, hidden = self.gru(chunk_summaries)

        # Use final hidden state as global context
        global_context = hidden[-1]  # [batch, hidden_dim]

        # Normalize
        global_context = self.norm(global_context)

        return global_context


class HierarchicalGRU(nn.Module):
    """
    Hierarchical GRU with two-level processing for extended context.

    Experiment A from Extended Context PRD:
    - Process 100-vector context in 10 chunks of 10 vectors
    - Local encoder summarizes each chunk
    - Global encoder attends over chunk summaries

    Args:
        d_model: Vector dimension (768 for GTR-T5)
        hidden_dim: GRU hidden dimension
        chunk_size: Vectors per chunk (default 10)
        num_chunks: Number of chunks (default 10)
        local_layers: GRU layers in local encoder
        global_layers: GRU layers in global encoder
    """
    def __init__(
        self,
        d_model=768,
        hidden_dim=512,
        chunk_size=10,
        num_chunks=10,
        local_layers=2,
        global_layers=2
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.context_length = chunk_size * num_chunks

        # Two-level hierarchy
        self.local_encoder = LocalChunkEncoder(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=local_layers
        )

        self.global_encoder = GlobalContextEncoder(
            hidden_dim=hidden_dim,
            num_layers=global_layers
        )

        # Output projection: hidden_dim -> d_model
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, d_model)
        )

        # Residual connection (optional, for stability)
        self.use_residual = True
        if self.use_residual:
            self.residual_proj = nn.Linear(d_model, d_model)

    def forward(self, x, tmd=None, return_raw=False):
        """
        Args:
            x: [batch, context_length, d_model] (e.g., [batch, 100, 768])
            tmd: Optional TMD vector [batch, 16] (not used in baseline, reserved for Phase 2)
            return_raw: If True, return (raw, normalized) tuple

        Returns:
            output: [batch, d_model] - predicted next vector (L2 normalized)
            OR (raw, normalized) if return_raw=True
        """
        batch_size, seq_len, _ = x.shape

        # Verify input shape
        assert seq_len == self.context_length, \
            f"Expected context_length={self.context_length}, got {seq_len}"

        # Level 1: Local chunk encoding
        chunk_summaries = []
        for i in range(self.num_chunks):
            start_idx = i * self.chunk_size
            end_idx = start_idx + self.chunk_size
            chunk = x[:, start_idx:end_idx, :]  # [batch, chunk_size, d_model]

            summary = self.local_encoder(chunk)  # [batch, hidden_dim]
            chunk_summaries.append(summary)

        # Stack chunk summaries: [batch, num_chunks, hidden_dim]
        chunk_summaries = torch.stack(chunk_summaries, dim=1)

        # Level 2: Global context encoding
        global_context = self.global_encoder(chunk_summaries)  # [batch, hidden_dim]

        # Output projection
        raw = self.output_proj(global_context)  # [batch, d_model]

        # Optional residual connection (from last input vector)
        if self.use_residual:
            residual = self.residual_proj(x[:, -1, :])  # [batch, d_model]
            raw = raw + residual

        # Normalize
        normalized = torch.nn.functional.normalize(raw, p=2, dim=-1)

        if return_raw:
            return raw, normalized
        return normalized

    def get_num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        """Alias for get_num_params() to match trainer interface."""
        return self.get_num_params()


def test_hierarchical_gru():
    """Test Hierarchical GRU with sample data."""
    print("Testing Hierarchical GRU...")

    # Create model
    model = HierarchicalGRU(
        d_model=768,
        hidden_dim=512,
        chunk_size=10,
        num_chunks=10,
        local_layers=2,
        global_layers=2
    )

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Context length: {model.context_length} vectors (~{model.context_length * 20} tokens)")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 100, 768)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, 768), f"Expected [4, 768], got {output.shape}"

    print("✓ Hierarchical GRU test passed!")


if __name__ == '__main__':
    test_hierarchical_gru()
