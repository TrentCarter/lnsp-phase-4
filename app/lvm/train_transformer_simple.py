#!/usr/bin/env python3
"""
Simple Transformer trainer for LVM - using cosine similarity loss only
Similar to Simple MLP but with Transformer architecture
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SimpleTransformerPredictor(nn.Module):
    """Simple Transformer for next-vector prediction"""

    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layers (simpler than decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

        self.normalize = True

    def forward(self, x):
        # x: [batch, seq_len, input_dim]

        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Take last position
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Project to output
        out = self.head(last_hidden)  # [batch, input_dim]

        # Normalize to unit sphere
        if self.normalize:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)

        return out


class SequenceDataset(Dataset):
    """Dataset for sequence prediction"""

    def __init__(self, contexts, targets):
        self.contexts = torch.from_numpy(contexts).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


def cosine_similarity_loss(predictions, targets):
    """1 - cosine similarity (minimize)"""
    # Normalize
    pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
    target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=1)

    # Loss: 1 - cosine (minimize)
    loss = 1.0 - cos_sim
    return loss.mean()


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_cosine = 0.0

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(contexts)

        # Loss
        loss = cosine_similarity_loss(predictions, targets)

        # Backward
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()

        # Compute cosine for monitoring
        with torch.no_grad():
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_cosine


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(contexts)

            # Loss
            loss = cosine_similarity_loss(predictions, targets)

            # Cosine
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_cosine


def main():
    parser = argparse.ArgumentParser(description="Train simple Transformer for vector prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to training_sequences_ctx5.npz")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    data = np.load(args.data)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Train/val split (90/10)
    num_samples = len(contexts)
    train_size = int(0.9 * num_samples)

    train_contexts = contexts[:train_size]
    train_targets = targets[:train_size]
    val_contexts = contexts[train_size:]
    val_targets = targets[train_size:]

    print(f"Training samples: {len(train_contexts)}")
    print(f"Validation samples: {len(val_contexts)}")

    # Create datasets
    train_dataset = SequenceDataset(train_contexts, train_targets)
    val_dataset = SequenceDataset(val_contexts, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device(args.device)
    model = SimpleTransformerPredictor(
        input_dim=768,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input: 5x768D sequence")
    print(f"  Transformer: {args.num_layers} layers, d_model={args.d_model}, {args.nhead} heads")
    print(f"  Output: 768D (normalized)")
    print(f"  Parameters: {num_params:,}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        train_loss, train_cosine = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_cosine = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f} | Train Cosine: {train_cosine:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_cosine': float(train_cosine),
            'val_loss': float(val_loss),
            'val_cosine': float(val_cosine)
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
