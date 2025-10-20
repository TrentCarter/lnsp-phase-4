"""
Memory-Augmented GRU for Extended Context

Architecture:
- Core GRU processes input sequence
- External memory bank (2,048 slots × 768D) stores persistent concepts
- Content-based read/write operations
- TMD-aware memory routing (optional for Phase 2)

Memory Operations:
1. Read: Query memory with current hidden state → retrieve relevant concepts
2. Process: GRU combines input + memory content
3. Write: Update memory with new information

Benefits:
- Persistent knowledge across sequences (like a "working memory")
- Scales beyond fixed context windows
- Can specialize memory slots by domain (with TMD routing)

Created: 2025-10-19 (Extended Context Experiments - Experiment B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAddressableMemory(nn.Module):
    """
    External memory bank with content-based addressing.

    Memory operations inspired by Neural Turing Machines and Differentiable Neural Computers.

    Args:
        num_slots: Number of memory slots (default 2048)
        slot_dim: Dimension per slot (default 768, matches GTR-T5)
        query_dim: Dimension of query vector (default 512, GRU hidden dim)
    """
    def __init__(self, num_slots=2048, slot_dim=768, query_dim=512):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.query_dim = query_dim

        # Memory bank: [num_slots, slot_dim]
        # Initialize with small random values
        self.register_buffer(
            'memory',
            torch.randn(num_slots, slot_dim) * 0.01
        )

        # Query projection: query_dim -> slot_dim
        self.query_proj = nn.Linear(query_dim, slot_dim)

        # Write gate: Decide how much to update memory
        self.write_gate = nn.Linear(query_dim, num_slots)

    def read(self, query):
        """
        Read from memory using content-based addressing.

        Args:
            query: [batch, query_dim] - Query vector (e.g., GRU hidden state)

        Returns:
            content: [batch, slot_dim] - Retrieved memory content
            weights: [batch, num_slots] - Attention weights (for visualization)
        """
        batch_size = query.size(0)

        # Project query to slot dimension
        query_projected = self.query_proj(query)  # [batch, slot_dim]

        # Compute cosine similarity with all memory slots
        # Normalize query and memory
        query_norm = F.normalize(query_projected, p=2, dim=1)  # [batch, slot_dim]
        memory_norm = F.normalize(self.memory, p=2, dim=1)  # [num_slots, slot_dim]

        # Similarity: [batch, num_slots]
        similarity = torch.matmul(query_norm, memory_norm.t())

        # Softmax to get attention weights
        weights = F.softmax(similarity, dim=1)  # [batch, num_slots]

        # Weighted sum of memory slots
        content = torch.matmul(weights, self.memory)  # [batch, slot_dim]

        return content, weights

    def write(self, query, value):
        """
        Write to memory using soft attention.

        Args:
            query: [batch, query_dim] - Query vector (where to write)
            value: [batch, slot_dim] - Value to write
        """
        batch_size = query.size(0)

        # Compute write weights (where to write)
        write_logits = self.write_gate(query)  # [batch, num_slots]
        write_weights = F.softmax(write_logits, dim=1)  # [batch, num_slots]

        # Blend new value into memory (batch average for simplicity)
        # In practice, use per-example write with erase/add mechanism (like NTM)
        write_weights_mean = write_weights.mean(dim=0)  # [num_slots]
        value_mean = value.mean(dim=0)  # [slot_dim]

        # Update memory: weighted blend of old and new
        update = torch.outer(write_weights_mean, value_mean)  # [num_slots, slot_dim]
        self.memory.data = 0.9 * self.memory.data + 0.1 * update


class MemoryAugmentedGRU(nn.Module):
    """
    GRU with external memory bank for extended context.

    Experiment B from Extended Context PRD:
    - Core GRU processes input sequence (100 vectors)
    - External memory provides persistent knowledge (2,048 slots)
    - Read-process-write cycle at each step

    Args:
        d_model: Vector dimension (768 for GTR-T5)
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
        memory_slots: Number of external memory slots
        use_memory_write: Enable memory updates during training
    """
    def __init__(
        self,
        d_model=768,
        hidden_dim=512,
        num_layers=4,
        memory_slots=2048,
        use_memory_write=True
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.memory_slots = memory_slots
        self.use_memory_write = use_memory_write

        # Core GRU
        self.gru = nn.GRU(
            input_size=d_model + d_model,  # Input + memory content
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # External memory
        self.memory = ContentAddressableMemory(
            num_slots=memory_slots,
            slot_dim=d_model,
            query_dim=hidden_dim
        )

        # Input projection (optional)
        self.input_norm = nn.LayerNorm(d_model)

        # Fusion layer: Combine GRU output + memory
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Output projection: hidden_dim -> d_model
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, d_model)
        )

    def forward(self, x, tmd=None, return_raw=False):
        """
        Args:
            x: [batch, seq_len, d_model] (e.g., [batch, 100, 768])
            tmd: Optional TMD vector [batch, 16] (reserved for Phase 2 TMD-aware routing)
            return_raw: If True, return (raw, normalized) tuple

        Returns:
            output: [batch, d_model] - predicted next vector (L2 normalized)
            OR (raw, normalized) if return_raw=True
        """
        batch_size, seq_len, _ = x.shape

        # Normalize input
        x = self.input_norm(x)

        # Read from memory (initial read with zero query)
        initial_query = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        memory_content, _ = self.memory.read(initial_query)

        # Augment each input vector with memory content
        # For simplicity, use same memory read for all steps (can make per-step in Phase 2)
        memory_expanded = memory_content.unsqueeze(1).expand(-1, seq_len, -1)
        x_augmented = torch.cat([x, memory_expanded], dim=-1)  # [batch, seq_len, d_model*2]

        # GRU processing
        gru_output, gru_hidden = self.gru(x_augmented)

        # Final hidden state
        final_hidden = gru_hidden[-1]  # [batch, hidden_dim]

        # Read from memory again with final hidden state (query)
        final_memory_content, attention_weights = self.memory.read(final_hidden)

        # Fuse GRU output + memory content
        fused = torch.cat([final_hidden, final_memory_content], dim=-1)
        fused = self.fusion(fused)  # [batch, hidden_dim]

        # Output projection
        raw = self.output_proj(fused)  # [batch, d_model]

        # Normalize
        normalized = torch.nn.functional.normalize(raw, p=2, dim=-1)

        # Write to memory (update for next iteration)
        if self.training and self.use_memory_write:
            self.memory.write(final_hidden, raw)

        if return_raw:
            return raw, normalized
        return normalized

    def get_num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        """Alias for get_num_params() to match trainer interface."""
        return self.get_num_params()

    def get_memory_stats(self):
        """Get memory bank statistics (for debugging/visualization)."""
        memory_data = self.memory.memory.data  # [num_slots, slot_dim]

        stats = {
            'num_slots': self.memory_slots,
            'slot_dim': self.d_model,
            'memory_norm_mean': memory_data.norm(dim=1).mean().item(),
            'memory_norm_std': memory_data.norm(dim=1).std().item(),
            'memory_sparsity': (memory_data.abs() < 0.01).float().mean().item()
        }

        return stats


def test_memory_gru():
    """Test Memory-Augmented GRU with sample data."""
    print("Testing Memory-Augmented GRU...")

    # Create model
    model = MemoryAugmentedGRU(
        d_model=768,
        hidden_dim=512,
        num_layers=4,
        memory_slots=2048,
        use_memory_write=True
    )

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Memory slots: {model.memory_slots}")

    # Test forward pass
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 768)

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, 768), f"Expected [4, 768], got {output.shape}"

    # Check memory stats
    stats = model.get_memory_stats()
    print(f"Memory stats: {stats}")

    print("✓ Memory-Augmented GRU test passed!")


if __name__ == '__main__':
    test_memory_gru()
