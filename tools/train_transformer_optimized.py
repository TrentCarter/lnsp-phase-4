#!/usr/bin/env python3
"""
Optimized Transformer Training
===============================

Re-train Transformer with consultant's optimization suggestions:
1. 5% LR warmup (1 epoch)
2. Cosine annealing LR schedule
3. Early stopping (patience=4)

This tests whether LR scheduling can improve the Transformer's performance
beyond the baseline 0.5774 in-dist / 0.6214 OOD results.

Usage:
    python tools/train_transformer_optimized.py
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add app/lvm to path
sys.path.insert(0, 'app/lvm')

from models import create_model, MODEL_SPECS
from loss_utils import LossWeights, compute_losses
from train_helpers import (
    CycleConfig,
    compute_batch_stats,
    compute_mmd_rbf,
    mean_variance_penalty,
    maybe_cycle_penalty,
    sample_anchors,
)


class VectorSequenceDataset(Dataset):
    """Dataset for autoregressive vector prediction"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.contexts = torch.FloatTensor(data['context_sequences'])
        self.targets = torch.FloatTensor(data['target_vectors'])

        print(f"Loaded {len(self.contexts)} training pairs")
        print(f"Context shape: {self.contexts.shape}")
        print(f"Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup + cosine annealing"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        epoch = self.current_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""

    def __init__(self, patience=4, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def cosine_similarity(pred, target):
    """Compute cosine similarity between predictions and targets"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_weights: LossWeights,
):
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    stats_acc = {
        "loss_mse": 0.0,
    }

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        pred_raw, pred_cos = model(contexts, return_raw=True)

        loss, stats = compute_losses(pred_raw, pred_cos, targets, loss_weights)
        stats_acc["loss_mse"] += stats.get("loss_mse", 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            cosine = cosine_similarity(pred_cos, targets)
            total_cosine += cosine.item()

        if batch_idx % 100 == 0:
            print(
                "  Batch {}/{} | Loss: {:.6f} | MSE: {:.6f} | Cosine: {:.4f}".format(
                    batch_idx,
                    len(dataloader),
                    loss.item(),
                    stats.get("loss_mse", 0.0),
                    cosine.item(),
                )
            )

    denom = len(dataloader)
    avg_stats = {k: v / denom for k, v in stats_acc.items() if denom > 0}
    return total_loss / denom, total_cosine / denom, avg_stats


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_cosine = 0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            predictions = model(contexts)
            loss = nn.functional.mse_loss(predictions, targets)
            cosine = cosine_similarity(predictions, targets)

            total_loss += loss.item()
            total_cosine += cosine.item()

    return total_loss / len(dataloader), total_cosine / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Optimized Transformer Training')

    # Data and training
    parser.add_argument('--data', default='artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', default=None)

    # Optimization params
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Number of warmup epochs (5% of 20 = 1)')
    parser.add_argument('--patience', type=int, default=4, help='Early stopping patience')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing')

    args = parser.parse_args()

    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'artifacts/lvm/models/transformer_optimized_{timestamp}'

    print("=" * 80)
    print("OPTIMIZED TRANSFORMER TRAINING")
    print("=" * 80)
    print("Optimizations:")
    print(f"  1. LR Warmup: {args.warmup_epochs} epochs (5%)")
    print(f"  2. Cosine Annealing: epochs {args.warmup_epochs+1}-{args.epochs}")
    print(f"  3. Early Stopping: patience={args.patience}")
    print()
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} → {args.lr_min}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    dataset = VectorSequenceDataset(args.data)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Create Transformer model
    print("Creating Transformer model...")
    device = torch.device(args.device)
    model_config = {
        'input_dim': 768,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1
    }
    model = create_model('transformer', **model_config).to(device)

    actual_params = model.count_parameters()
    print(f"Parameters: {actual_params:,}")
    print()

    # Optimizer with warmup + cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        lr_min=args.lr_min
    )
    early_stopping = EarlyStopping(patience=args.patience)

    # Training loop
    best_val_loss = float('inf')
    best_val_cosine = 0.0
    history = []
    loss_weights = LossWeights(
        tau=0.07,
        mse=1.0,
        info_nce=0.0,
        moment=0.0,
        variance=0.0,
    )

    print("Starting training...")
    print()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")

        # Train
        train_loss, train_cosine, train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
        )

        # Validate
        val_loss, val_cosine = evaluate(model, val_loader, device)

        # Learning rate schedule
        scheduler.step()

        # Log
        print(
            "  Train Loss: {:.6f} | Train Cosine: {:.4f} | MSE: {:.6f}".format(
                train_loss,
                train_cosine,
                train_stats["loss_mse"],
            )
        )
        print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")
        print()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            'train_loss_mse': train_stats['loss_mse'],
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cosine = val_cosine
            torch.save({
                'epoch': epoch + 1,
                'model_type': 'transformer',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'model_config': model_config,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f}, val_cosine: {val_cosine:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"    No improvement for {args.patience} epochs")
            break

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_type': 'transformer',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'model_config': model_config,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'model_type': 'transformer_optimized',
            'model_name': 'Transformer (Optimized)',
            'optimizations': [
                f'{args.warmup_epochs} epoch LR warmup',
                'Cosine annealing LR schedule',
                f'Early stopping (patience={args.patience})'
            ],
            'history': history,
            'best_val_loss': best_val_loss,
            'best_val_cosine': best_val_cosine,
            'final_params': actual_params,
            'stopped_early': early_stopping.early_stop,
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("=" * 80)
    print("Training Complete!")
    print(f"Model: Transformer (Optimized)")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val cosine: {best_val_cosine:.4f}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Early stopped: {early_stopping.early_stop}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
