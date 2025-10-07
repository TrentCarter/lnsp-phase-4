#!/usr/bin/env python3
"""
Train both LSTM and Mamba models side-by-side for comparison.

This script trains:
1. LatentLSTM (unintended baseline, simpler architecture)
2. LatentMamba (intended tokenless architecture)

Both models train on same data, same hyperparameters, same device.
Results saved to separate checkpoints for A/B testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lvm.models import LatentMamba
from src.lvm.models_lstm import LatentLSTM


def load_training_data(data_path: str):
    """Load prepared training data."""
    data = np.load(data_path)
    return {
        'train': (data['train_contexts'], data['train_targets'], data['train_masks']),
        'val': (data['val_contexts'], data['val_targets'], data['val_masks']),
        'test': (data['test_contexts'], data['test_targets'], data['test_masks'])
    }


def train_epoch(model, dataloader, optimizer, device, model_name="Model"):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()

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

    elapsed = time.time() - start_time
    avg_loss = total_loss / len(dataloader)

    return avg_loss, elapsed


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


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    output_path
):
    """Train a model and track metrics."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, LR: {lr}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        # Train
        train_loss, train_time = train_epoch(model, train_loader, optimizer, device, model_name)
        history['train_loss'].append(train_loss)
        history['epoch_time'].append(train_time)

        # Validate
        val_loss = evaluate(model, val_loader, device)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={train_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {output_path}")

    return history, best_val_loss


def main():
    # Configuration
    data_path = "artifacts/lvm/wordnet_training_sequences.npz"
    epochs = 10
    batch_size = 64
    lr = 1e-3
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("="*60)
    print("LSTM vs Mamba Side-by-Side Training")
    print("="*60)
    print(f"\nData: {data_path}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}\n")

    # Load data
    print("Loading training data...")
    data = load_training_data(data_path)
    train_ctx, train_tgt, train_mask = data['train']
    val_ctx, val_tgt, val_mask = data['val']
    test_ctx, test_tgt, test_mask = data['test']

    print(f"  Train: {len(train_ctx)} sequences")
    print(f"  Val: {len(val_ctx)} sequences")
    print(f"  Test: {len(test_ctx)} sequences")
    print(f"  Context shape: {train_ctx.shape}")

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
    test_dataset = TensorDataset(
        torch.from_numpy(test_ctx).float(),
        torch.from_numpy(test_tgt).float(),
        torch.from_numpy(test_mask).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ========================================================================
    # Train LSTM
    # ========================================================================
    lstm_model = LatentLSTM(d_input=784, d_hidden=512, n_layers=2).to(device)
    lstm_history, lstm_best_val = train_model(
        model=lstm_model,
        model_name="LatentLSTM",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        output_path="models/lvm_lstm_retrained.pt"
    )

    # ========================================================================
    # Train Mamba
    # ========================================================================
    mamba_model = LatentMamba(d_input=784, d_hidden=512, n_layers=4).to(device)
    mamba_history, mamba_best_val = train_model(
        model=mamba_model,
        model_name="LatentMamba",
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        output_path="models/lvm_mamba_new.pt"
    )

    # ========================================================================
    # Final Comparison
    # ========================================================================
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    # Test both models
    lstm_model.load_state_dict(torch.load("models/lvm_lstm_retrained.pt", map_location=device))
    mamba_model.load_state_dict(torch.load("models/lvm_mamba_new.pt", map_location=device))

    lstm_test_loss = evaluate(lstm_model, test_loader, device)
    mamba_test_loss = evaluate(mamba_model, test_loader, device)

    print(f"\n{'Model':<20} {'Params':<15} {'Best Val Loss':<15} {'Test Loss':<15}")
    print("-"*65)
    print(f"{'LatentLSTM':<20} {lstm_model.get_num_params():<15,} {lstm_best_val:<15.4f} {lstm_test_loss:<15.4f}")
    print(f"{'LatentMamba':<20} {mamba_model.get_num_params():<15,} {mamba_best_val:<15.4f} {mamba_test_loss:<15.4f}")

    # Declare winner
    print("\n" + "="*60)
    if mamba_test_loss < lstm_test_loss:
        improvement = (lstm_test_loss - mamba_test_loss) / lstm_test_loss * 100
        print(f"✅ Winner: Mamba ({improvement:.1f}% better test loss)")
    elif lstm_test_loss < mamba_test_loss:
        improvement = (mamba_test_loss - lstm_test_loss) / mamba_test_loss * 100
        print(f"✅ Winner: LSTM ({improvement:.1f}% better test loss)")
    else:
        print(f"🤝 Tie: Both models perform equally")

    print("\nSaved models:")
    print("  - models/lvm_lstm_retrained.pt")
    print("  - models/lvm_mamba_new.pt")
    print("\nReady for inference testing!")


if __name__ == "__main__":
    main()
