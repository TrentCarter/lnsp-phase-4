"""
LSTM-based LVM (unintended but functional baseline).

This is the architecture from models/lvm_wordnet.pt - kept for comparison.
Input: 784D vectors (768D GTR-T5 + 16D TMD)
Output: 784D vector (next in sequence)
"""

import torch
import torch.nn as nn


class LatentLSTM(nn.Module):
    """
    LSTM-based sequence model for 784D vectors.

    Architecture:
    - Input projection: 784D → hidden_dim
    - 2-layer LSTM: hidden_dim → hidden_dim
    - Output head: hidden_dim → 784D

    This is simpler than Mamba but serves as a strong baseline.
    """

    def __init__(
        self,
        d_input: int = 784,
        d_hidden: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )

        # Output head
        self.output_head = nn.Linear(d_hidden, d_input)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, 784) - Context vectors
            mask: (batch, seq_len) - Optional attention mask

        Returns:
            (batch, 784) - Predicted next vector
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_hidden)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, d_hidden)

        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch, d_hidden)

        # Project to output
        output = self.output_head(final_hidden)  # (batch, 784)

        return output

    def get_num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_lstm_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained LSTM model from checkpoint."""
    model = LatentLSTM()

    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test LSTM model
    model = LatentLSTM(d_input=784, d_hidden=512, n_layers=2)

    print(f"LatentLSTM")
    print(f"  Input dim: {model.d_input}")
    print(f"  Hidden dim: {model.d_hidden}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Parameters: {model.get_num_params():,}")

    # Test forward pass
    batch_size = 4
    seq_len = 7
    x = torch.randn(batch_size, seq_len, 784)

    output = model(x)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (batch_size, 784), "Output shape mismatch!"
    print("\n✓ LSTM model test passed!")
