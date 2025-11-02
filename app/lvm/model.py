"""
Unified LVM (Latent Vector Model) Architectures

Contains all production LVM models:
- AMN: Additive Memory Network (fastest, best OOD)
- GRU: Gated Recurrent Unit (stacked with residuals)
- LSTM: Long Short-Term Memory (balanced)
- Transformer: Self-attention decoder (best accuracy)

All models predict next 768D vector from context sequence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# AMN - Additive Memory Network
# ============================================================================

class AMNModel(nn.Module):
    """
    Additive Memory Network for next-vector prediction.

    Architecture:
    - Separate encoders for context and query
    - Additive memory pooling
    - Residual MLP for prediction

    Best for: Zero-shot generalization, low latency
    Performance: OOD 0.6375, 0.62ms/query
    """
    def __init__(self, input_dim=768, d_model=256, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Context and query encoders
        self.context_encoder = nn.Linear(input_dim, d_model)
        self.query_encoder = nn.Linear(input_dim, d_model)

        # Residual network: [context + query] → prediction
        self.residual_net = nn.Sequential(
            nn.Linear(d_model * 6, hidden_dim),  # 6x for concatenated context
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            prediction: [batch, 768]
        """
        # Encode context (all but last)
        context_vecs = x[:, :-1, :]  # [batch, seq_len-1, 768]
        context_encoded = self.context_encoder(context_vecs)  # [batch, seq_len-1, d_model]

        # Pool context (mean)
        context_pooled = context_encoded.mean(dim=1)  # [batch, d_model]

        # Encode query (last position)
        query_vec = x[:, -1, :]  # [batch, 768]
        query_encoded = self.query_encoder(query_vec)  # [batch, d_model]

        # Concatenate representations (additive memory)
        # Note: Original uses 6x concatenation for richer representation
        combined = torch.cat([
            context_pooled, query_encoded,
            context_pooled, query_encoded,
            context_pooled, query_encoded
        ], dim=1)  # [batch, 6*d_model]

        # Predict
        prediction = self.residual_net(combined)  # [batch, 768]

        # L2 normalize
        prediction = F.normalize(prediction, p=2, dim=-1)

        return prediction


# ============================================================================
# GRU - Gated Recurrent Unit (Stacked)
# ============================================================================

class GRUBlock(nn.Module):
    """Single GRU block with residual connection and layer norm."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.norm(out + x)  # Residual connection


class GRUModel(nn.Module):
    """
    Stacked GRU with residual connections.

    Architecture:
    - Input projection to d_model
    - 4 stacked GRU blocks with residuals
    - Output projection to 768D

    Best for: Balanced accuracy and speed
    Performance: OOD 0.6295, 2.11ms/query
    """
    def __init__(self, input_dim=768, d_model=512, num_layers=4, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stacked GRU blocks
        self.blocks = nn.ModuleList([
            GRUBlock(d_model) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            prediction: [batch, 768]
        """
        # Project to hidden dim
        h = self.input_proj(x)  # [batch, seq_len, d_model]
        h = self.dropout(h)

        # Pass through GRU blocks
        for block in self.blocks:
            h = block(h)

        # Get last position
        last_hidden = h[:, -1, :]  # [batch, d_model]

        # Project to output
        prediction = self.output_proj(last_hidden)  # [batch, 768]

        # L2 normalize
        prediction = F.normalize(prediction, p=2, dim=-1)

        return prediction


# ============================================================================
# LSTM - Long Short-Term Memory
# ============================================================================

class LSTMBlock(nn.Module):
    """Single LSTM block with residual connection and layer norm."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        identity = x
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        out = self.norm(lstm_out + identity)  # Residual connection + layer norm
        return out


class LSTMModel(nn.Module):
    """Stacked LSTM with residual connections for next-vector prediction.

    Architecture:
    - Input projection to d_model
    - 4 stacked LSTM blocks with residuals
    - Output projection to 768D
    """
    def __init__(self, input_dim=768, d_model=512, num_layers=4, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stacked LSTM blocks
        self.blocks = nn.ModuleList([
            LSTMBlock(d_model) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            prediction: [batch, 768]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        # Apply LSTM blocks
        for block in self.blocks:
            x = block(x)  # [batch, seq_len, d_model]

        # Get last hidden state and project to output
        last_hidden = x[:, -1, :]  # [batch, d_model]
        output = self.output_proj(last_hidden)  # [batch, 768]
        return output


# ============================================================================
# Transformer - Self-Attention Decoder
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer decoder for next-vector prediction.

    Architecture:
    - Input projection + positional encoding
    - 4-layer transformer decoder with causal masking
    - Output head with GELU activation

    Best for: Highest accuracy (in-distribution)
    Performance: Val cosine 0.5820, 2.68ms/query
    """
    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, 768]
        Returns:
            prediction: [batch, 768]
        """
        # Project to d_model
        h = self.input_proj(x)  # [batch, seq_len, d_model]
        h = self.pos_encoder(h)

        # Causal mask
        seq_len = h.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(h.device)

        # Transformer forward
        h = self.transformer_decoder(tgt=h, memory=h, tgt_mask=causal_mask, memory_mask=causal_mask)

        # Get last position
        last_hidden = h[:, -1, :]  # [batch, d_model]

        # Output head
        prediction = self.head(last_hidden)  # [batch, 768]

        # L2 normalize
        prediction = F.normalize(prediction, p=2, dim=-1)

        return prediction


# ============================================================================
# Model Loading Utility
# ============================================================================

def load_lvm_model(model_type: str, checkpoint_path: str, device: str = "cpu"):
    """
    Load LVM model from checkpoint.

    Args:
        model_type: "amn", "gru", "lstm", or "transformer"
        checkpoint_path: Path to .pt checkpoint
        device: "cpu", "mps", or "cuda"

    Returns:
        model: Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config
    config = checkpoint['model_config']

    # Create model with parameter name handling
    if model_type == "amn":
        model = AMNModel(**config)
    elif model_type == "gru":
        model = GRUModel(**config)
    elif model_type == "lstm":
        # Handle parameter name differences for LSTM
        lstm_config = config.copy()
        if 'hidden_dim' in lstm_config and 'd_model' not in lstm_config:
            lstm_config['d_model'] = lstm_config.pop('hidden_dim')
        model = LSTMModel(**lstm_config)
    elif model_type == "transformer":
        # Handle both old and new transformer architectures
        transformer_config = config.copy()
        # Old TransformerModel doesn't accept output_dim, filter it out
        transformer_config.pop('output_dim', None)
        model = TransformerModel(**transformer_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Quick test
    print("Testing LVM models...")

    x = torch.randn(2, 5, 768)  # Batch=2, seq_len=5, dim=768

    # Test AMN
    amn = AMNModel()
    out_amn = amn(x)
    print(f"✅ AMN: {out_amn.shape}, norm={out_amn.norm(dim=-1).mean():.4f}")

    # Test GRU
    gru = GRUModel()
    out_gru = gru(x)
    print(f"✅ GRU: {out_gru.shape}, norm={out_gru.norm(dim=-1).mean():.4f}")

    # Test LSTM
    lstm = LSTMModel()
    out_lstm = lstm(x)
    print(f"✅ LSTM: {out_lstm.shape}, norm={out_lstm.norm(dim=-1).mean():.4f}")

    # Test Transformer
    transformer = TransformerModel()
    out_transformer = transformer(x)
    print(f"✅ Transformer: {out_transformer.shape}, norm={out_transformer.norm(dim=-1).mean():.4f}")
