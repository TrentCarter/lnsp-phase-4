#!/usr/bin/env python3
"""
Two-Tower v4 STABLE Training: macOS ARM64 Safety Rails
Based on train_twotower_v4.py with crash mitigations:
- Python 3.11 + PyTorch 2.5.0 (avoid 3.13 + 2.7.x edge cases)
- Single-threaded CPU (no MKL/BLAS races)
- AdamW with foreach=False (avoid ARM64 state corruption)
- No DataLoader workers (avoid fork/spawn + FAISS issues)
- Optional: LSTM instead of GRU (more stable on ARM)
- Faulthandler enabled (get stack traces on crashes)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import time
import os
import faulthandler

# ============================================================
# SAFETY RAILS: Enable before any heavy imports
# ============================================================
faulthandler.enable()  # Get stack traces on segfaults

# Single-threaded execution (avoid MKL races on macOS ARM)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# Disable MKL-DNN (can cause crashes on ARM64)
try:
    torch.backends.mkldnn.enabled = False
except:
    pass

print("=" * 60)
print("STABLE TRAINING MODE")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Threads: {torch.get_num_threads()} (compute), {torch.get_num_interop_threads()} (interop)")
print(f"MKL-DNN: {torch.backends.mkldnn.is_available()}")
print("=" * 60)
print()


# ============================================================
# Model Architecture
# ============================================================

class GRUPoolQuery(torch.nn.Module):
    """Query tower with GRU + mean pooling"""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(d_model, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, dim=-1)


class LSTMPoolQuery(torch.nn.Module):
    """Query tower with LSTM + mean pooling (more stable on ARM64)"""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(d_model, hidden_dim, num_layers,
                                   batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, dim=-1)


class IdentityDocTower(torch.nn.Module):
    """Document tower (just L2 normalization)"""
    def forward(self, x):
        return F.normalize(x, dim=-1)


# ============================================================
# Memory Bank
# ============================================================

class MemoryBank:
    """FIFO queue of recent document vectors"""
    def __init__(self, max_size=50000, dim=768, device='cpu'):
        self.max_size = max_size
        self.dim = dim
        self.device = device
        self.vectors = torch.zeros(0, dim, device=device)

    def add(self, vecs):
        """Add vectors to bank (FIFO)"""
        vecs = vecs.detach()
        self.vectors = torch.cat([self.vectors, vecs], dim=0)
        if len(self.vectors) > self.max_size:
            self.vectors = self.vectors[-self.max_size:]

    def get_all(self):
        """Get all vectors in bank"""
        return self.vectors


# ============================================================
# Hard Negative Mining (DISABLED for initial stability test)
# ============================================================

def mine_hard_negatives_disabled():
    """Mining disabled for stability testing"""
    print("  ⚠️  Hard negative mining DISABLED (stability mode)")
    return None


# ============================================================
# Loss Function
# ============================================================

def info_nce_loss_simple(q, d_pos, tau=0.05):
    """
    Simple InfoNCE without memory bank or hard negatives
    Args:
        q: (B, D) query vectors
        d_pos: (B, D) positive doc vectors
        tau: temperature
    """
    B = q.size(0)

    # Compute similarities (B, B)
    logits = torch.matmul(q, d_pos.T) / tau

    # Labels: diagonal (i-th query matches i-th doc)
    labels = torch.arange(B, device=q.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch_stable(model_q, model_d, train_loader, optimizer, device='cpu', tau=0.05):
    """
    STABLE training loop: no mining, no memory bank, no accumulation
    """
    model_q.train()
    model_d.train()

    total_loss = 0.0
    num_batches = 0

    for X_batch, Y_batch in tqdm(train_loader, desc="  Training", leave=False):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Encode
        q = model_q(X_batch)
        d_pos = model_d(Y_batch)

        # Simple InfoNCE loss
        loss = info_nce_loss_simple(q, d_pos, tau=tau)

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\n⚠️  Non-finite loss detected: {loss.item()}")
            print(f"   q range: [{q.min():.4f}, {q.max():.4f}]")
            print(f"   d_pos range: [{d_pos.min():.4f}, {d_pos.max():.4f}]")
            raise RuntimeError("Non-finite loss")

        # Backward + step (no accumulation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_simple(model_q, model_d, val_loader, device='cpu', tau=0.05):
    """Simple validation: just compute loss"""
    model_q.eval()
    model_d.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            q = model_q(X_batch)
            d_pos = model_d(Y_batch)

            loss = info_nce_loss_simple(q, d_pos, tau=tau)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, required=True)
    parser.add_argument('--bank', type=str, required=True)
    parser.add_argument('--out', type=str, default='runs/twotower_stable')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--use-lstm', action='store_true', help='Use LSTM instead of GRU (more stable)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Pairs: {args.pairs}")
    print(f"Bank: {args.bank}")
    print(f"Output: {args.out}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.wd}")
    print(f"Temperature: {args.tau}")
    print(f"Device: {args.device}")
    print(f"Model: {'LSTM' if args.use_lstm else 'GRU'}")
    print(f"Hidden dim: {args.hidden_dim}")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    pairs = np.load(args.pairs)

    X_train = torch.from_numpy(pairs['X_train'])
    Y_train = torch.from_numpy(pairs['Y_train'])
    X_val = torch.from_numpy(pairs['X_val'])
    Y_val = torch.from_numpy(pairs['Y_val'])

    print(f"  Train: {len(X_train)} pairs")
    print(f"  Val: {len(X_val)} pairs")
    print()

    # Create datasets (no shuffle, no workers, no pin_memory)
    from torch.utils.data import TensorDataset, DataLoader

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)

    # STABLE DataLoader: no workers, no pin_memory, drop_last
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,  # Keep shuffle for generalization
        num_workers=0,  # NO multiprocessing
        persistent_workers=False,
        pin_memory=False,  # NO pinned memory
        prefetch_factor=None,
        drop_last=True  # Avoid odd-sized last batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )

    print("Building models...")
    # Choose query tower
    if args.use_lstm:
        model_q = LSTMPoolQuery(d_model=768, hidden_dim=args.hidden_dim, num_layers=1)
        print("  Query tower: LSTM (stable mode)")
    else:
        model_q = GRUPoolQuery(d_model=768, hidden_dim=args.hidden_dim, num_layers=1)
        print("  Query tower: GRU")

    model_d = IdentityDocTower()

    model_q = model_q.to(args.device)
    model_d = model_d.to(args.device)

    num_params = sum(p.numel() for p in model_q.parameters())
    print(f"  Query tower params: {num_params:,}")
    print()

    # STABLE Optimizer: foreach=False, capturable=False
    optimizer = torch.optim.AdamW(
        model_q.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        foreach=False,  # NO fused foreach (causes ARM64 crashes)
        capturable=False
    )

    print("=" * 60)
    print("TRAINING (STABLE MODE)")
    print("=" * 60)
    print("Safety features:")
    print("  ✓ No hard negative mining")
    print("  ✓ No gradient accumulation")
    print("  ✓ No memory bank")
    print("  ✓ No DataLoader workers")
    print("  ✓ AdamW with foreach=False")
    print("  ✓ Single-threaded CPU")
    print("=" * 60)
    print()

    # Save config
    config = vars(args)
    config['model_type'] = 'LSTM' if args.use_lstm else 'GRU'
    config['stable_mode'] = True
    config['foreach'] = False
    with open(out_dir / 'config.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Config saved: {out_dir / 'config.yaml'}")
    print()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_one_epoch_stable(
            model_q, model_d, train_loader, optimizer,
            device=args.device, tau=args.tau
        )

        # Validate
        val_loss = validate_simple(
            model_q, model_d, val_loader,
            device=args.device, tau=args.tau
        )

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_q': model_q.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, out_dir / 'best_model.pt')
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

        print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {out_dir / 'best_model.pt'}")
    print()


if __name__ == '__main__':
    main()
