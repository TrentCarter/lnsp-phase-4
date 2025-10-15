#!/usr/bin/env python3
"""
LSTM model for LVM - sequential modeling with recurrent layers
Architecture: 5x768D → LSTM → 768D
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class LSTMPredictor(nn.Module):
    """LSTM-based vector predictor"""

    def __init__(self, vector_dim=768, hidden_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=vector_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim)
        )

        self.normalize = True

    def forward(self, x):
        # x shape: (batch_size, seq_len=5, vector_dim=768)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Project to output
        out = self.output_proj(last_hidden)

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
    parser = argparse.ArgumentParser(description="Train LSTM for vector prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to training_sequences_ctx5.npz")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
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
    model = LSTMPredictor(
        vector_dim=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel architecture:")
    print(f"  Input: 5x768D sequence")
    print(f"  LSTM: {args.num_layers} layers, {args.hidden_dim}D hidden")
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
