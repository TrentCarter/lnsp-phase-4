#!/usr/bin/env python3
"""
Train Alignment Head
====================

Trains a tiny MLP to align predicted vectors with true vectors.

Architecture: 768→256→768 with residual connection
Loss: 0.7 * cosine + 0.3 * MSE
Training: 1-3 epochs, early stopping on val cosine plateau
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class AlignmentHead(nn.Module):
    """Tiny MLP with residual for vector alignment."""

    def __init__(self, dim=768, hidden_dim=256, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # x: [B, 768]
        # Residual connection: v' = norm(v + α * head(v))
        residual = self.net(x)
        v_out = x + self.alpha * residual
        # L2 normalize
        v_out = F.normalize(v_out, p=2, dim=-1)
        return v_out


class AlignmentDataset(Dataset):
    def __init__(self, v_pred, v_true):
        self.v_pred = torch.from_numpy(v_pred).float()
        self.v_true = torch.from_numpy(v_true).float()

    def __len__(self):
        return len(self.v_pred)

    def __getitem__(self, idx):
        return self.v_pred[idx], self.v_true[idx]


def combined_loss(v_pred, v_true, w_cosine=0.7, w_mse=0.3):
    """Combined cosine + MSE loss."""
    # Cosine loss (1 - cosine similarity)
    cos_sim = F.cosine_similarity(v_pred, v_true, dim=-1)
    cos_loss = (1.0 - cos_sim).mean()

    # MSE loss
    mse_loss = F.mse_loss(v_pred, v_true)

    return w_cosine * cos_loss + w_mse * mse_loss, cos_sim.mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_cosine = 0
    n_batches = 0

    for v_pred, v_true in loader:
        v_pred, v_true = v_pred.to(device), v_true.to(device)

        # Forward
        v_aligned = model(v_pred)
        loss, cosine = combined_loss(v_aligned, v_true)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cosine += cosine.item()
        n_batches += 1

    return total_loss / n_batches, total_cosine / n_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_cosine = 0
    n_batches = 0

    with torch.no_grad():
        for v_pred, v_true in loader:
            v_pred, v_true = v_pred.to(device), v_true.to(device)

            v_aligned = model(v_pred)
            loss, cosine = combined_loss(v_aligned, v_true)

            total_loss += loss.item()
            total_cosine += cosine.item()
            n_batches += 1

    return total_loss / n_batches, total_cosine / n_batches


def main():
    print("=" * 80)
    print("TRAINING ALIGNMENT HEAD")
    print("=" * 80)
    print()

    # Config
    device = torch.device("cpu")  # CPU training (fast enough for small model)
    batch_size = 1024
    lr = 1e-3
    weight_decay = 1e-4
    max_epochs = 3
    patience = 2

    # Load data
    print("Loading training data...")
    data_path = Path("artifacts/lvm/alignment_training_data.npz")
    data = np.load(data_path)

    train_dataset = AlignmentDataset(data["v_pred_train"], data["v_true_train"])
    val_dataset = AlignmentDataset(data["v_pred_val"], data["v_true_val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset):,} pairs")
    print(f"  Val:   {len(val_dataset):,} pairs")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}\n")

    # Create model
    print("Creating model...")
    model = AlignmentHead(dim=768, hidden_dim=256, alpha=0.5)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    print("Training...")
    best_val_cosine = 0.0
    no_improve = 0
    history = []

    for epoch in range(max_epochs):
        train_loss, train_cosine = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_cosine = validate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_cosine": float(train_cosine),
            "val_loss": float(val_loss),
            "val_cosine": float(val_cosine),
        })

        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"  Train: loss={train_loss:.4f}, cosine={train_cosine:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, cosine={val_cosine:.4f}")

        # Save best model
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            no_improve = 0

            model_path = Path("artifacts/lvm/alignment_head.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_cosine": val_cosine,
                "config": {"dim": 768, "hidden_dim": 256, "alpha": 0.5},
            }, model_path)
            print(f"  ✓ Saved best model (val_cosine={val_cosine:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        print()

        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs\n")
            break

    # Save history
    history_path = Path("artifacts/lvm/alignment_head_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best val cosine: {best_val_cosine:.4f}")
    print(f"Model saved to: artifacts/lvm/alignment_head.pt")
    print()


if __name__ == "__main__":
    main()
