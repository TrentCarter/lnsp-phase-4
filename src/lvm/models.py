"""
Latent Vector Model (LVM) - Tokenless Mamba-2 Architecture

Processes 768D vectors directly (no text tokens!)
"""

import torch
import torch.nn as nn


class LatentMamba(nn.Module):
    """
    Simplified Mamba model for vector sequence processing.

    Uses standard LSTM for initial version (Mamba-SSM can be added later).
    Focus: Get training working first, then optimize architecture.
    """

    def __init__(
        self,
        d_input: int = 784,      # Input dim (768 semantic + 16 TMD)
        d_hidden: int = 512,     # Hidden dimension
        n_layers: int = 2,       # Number of layers
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)

        # LSTM layers (will replace with Mamba later)
        self.lstm = nn.LSTM(
            d_hidden,
            d_hidden,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Output head (project to 784D)
        self.output_head = nn.Linear(d_hidden, d_input)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: Input vectors [batch, seq_len, 784]
            mask: Attention mask [batch, seq_len] (optional)

        Returns:
            output: Predicted next vector [batch, 784]
        """
        # Input projection
        h = self.input_proj(x)  # [batch, seq_len, d_hidden]
        h = self.dropout(h)

        # LSTM forward
        lstm_out, _ = self.lstm(h)  # [batch, seq_len, d_hidden]

        # Get last valid position
        if mask is not None:
            # Get last non-masked position for each sequence
            lengths = mask.sum(dim=1).long() - 1  # [batch]
            last_h = lstm_out[torch.arange(lstm_out.size(0)), lengths]
        else:
            last_h = lstm_out[:, -1, :]  # [batch, d_hidden]

        # Project to output
        output = self.output_head(last_h)  # [batch, 784]

        return output

    def get_num_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = LatentMamba(d_input=784, d_hidden=512, n_layers=2)
    x = torch.randn(4, 10, 784)
    mask = torch.ones(4, 10)
    mask[:, 7:] = 0

    output = model(x, mask=mask)
    print(f"✅ Model has {model.get_num_params():,} parameters")
    print(f"✅ Output shape: {output.shape}")
    assert output.shape == (4, 784)
