"""
Train LVM on ordered ontology sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import time

from src.lvm.models import LatentMamba


def load_training_data(data_path: str):
    """Load prepared training data."""
    data = np.load(data_path)

    return {
        'train': (data['train_contexts'], data['train_targets'], data['train_masks']),
        'val': (data['val_contexts'], data['val_targets'], data['val_masks']),
        'test': (data['test_contexts'], data['test_targets'], data['test_masks'])
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_ctx, batch_tgt, batch_mask in dataloader:
        batch_ctx = batch_ctx.to(device)
        batch_tgt = batch_tgt.to(device)
        batch_mask = batch_mask.to(device)

        # Forward pass
        pred = model(batch_ctx, mask=batch_mask)

        # MSE loss (predict next vector)
        loss = F.mse_loss(pred, batch_tgt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_ctx, batch_tgt, batch_mask in dataloader:
            batch_ctx = batch_ctx.to(device)
            batch_tgt = batch_tgt.to(device)
            batch_mask = batch_mask.to(device)

            pred = model(batch_ctx, mask=batch_mask)
            loss = F.mse_loss(pred, batch_tgt)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_lvm(
    data_path: str = "artifacts/lvm/wordnet_training_sequences.npz",
    output_path: str = "models/lvm_wordnet.pt",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "mps"
):
    """Train LVM end-to-end."""
    print("=== Training Tokenless Mamba LVM ===")

    # Load data
    print(f"\n1. Loading data from {data_path}...")
    data = load_training_data(data_path)
    train_ctx, train_tgt, train_mask = data['train']
    val_ctx, val_tgt, val_mask = data['val']

    print(f"   Train: {len(train_ctx)} sequences")
    print(f"   Val: {len(val_ctx)} sequences")
    print(f"   Context shape: {train_ctx.shape}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_ctx).float(),
        torch.from_numpy(train_tgt).float(),
        torch.from_numpy(train_mask).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_ctx).float(),
        torch.from_numpy(val_tgt).float(),
        torch.from_numpy(val_mask).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    print(f"\n2. Creating model...")
    d_input = train_ctx.shape[2]  # 784D
    model = LatentMamba(d_input=d_input, d_hidden=512, n_layers=2)
    model = model.to(device)
    print(f"   Parameters: {model.get_num_params():,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\n3. Training for {epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        elapsed = time.time() - start

        print(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s)")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"  ✅ Saved best model (val_loss={val_loss:.6f})")

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    train_lvm(
        data_path="artifacts/lvm/wordnet_training_sequences.npz",
        output_path="models/lvm_wordnet.pt",
        epochs=10,
        batch_size=64,
        device="mps"  # Use "cuda" if available, "cpu" otherwise
    )
