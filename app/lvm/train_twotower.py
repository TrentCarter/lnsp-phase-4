#!/usr/bin/env python3
"""
Two-Tower Retriever (Option 4 - Contractor Pivot)
===================================================

Separate Q (query) and P (payload) towers with symmetric InfoNCE.

Architecture:
- Q tower: context (5×768) → query vector (768D, L2=1)
- P tower: target chunk (768D) → payload vector (768D, L2=1)
- Loss: InfoNCE on cos(q, p) with in-batch negatives
- No AR, no projection head, no shared weights

Usage:
    python app/lvm/train_twotower.py \
        --arch-q mamba_s --arch-p mamba_s \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --epochs 2 --eval-every 1 \
        --save-dir artifacts/lvm/models/twotower_mamba_s
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.lvm.mamba import create_model


class QueryTower(nn.Module):
    """Q tower: context sequence → query vector."""

    def __init__(self, backbone_type='mamba_s', d_model=768, **kwargs):
        super().__init__()
        self.backbone = create_model(
            model_type=backbone_type,
            d_model=d_model,
            **kwargs
        )

    def forward(self, context):
        """
        Args:
            context: [B, seq_len, 768] context sequence
        Returns:
            q: [B, 768] L2-normalized query vector
        """
        # Backbone output
        h = self.backbone(context)  # [B, seq_len, 768] or [B, 768]

        # Take last position if sequence output
        if len(h.shape) == 3:
            h = h[:, -1, :]  # [B, 768]

        # L2 normalize
        q = F.normalize(h, p=2, dim=-1)
        return q


class PayloadTower(nn.Module):
    """P tower: target chunk → payload vector."""

    def __init__(self, backbone_type='mamba_s', d_model=768, **kwargs):
        super().__init__()
        # Simple MLP for now (can use Mamba if desired)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, target):
        """
        Args:
            target: [B, 768] target chunk vector
        Returns:
            p: [B, 768] L2-normalized payload vector
        """
        h = self.mlp(target)
        p = F.normalize(h, p=2, dim=-1)
        return p


class TwoTowerInfoNCE(nn.Module):
    """Symmetric InfoNCE loss for two-tower retrieval."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, p):
        """
        Args:
            q: [B, 768] query vectors (L2-normalized)
            p: [B, 768] payload vectors (L2-normalized)

        Returns:
            loss, metrics_dict
        """
        B = q.shape[0]

        # Compute similarity matrix: [B, B]
        # logits[i,j] = cos(q_i, p_j) / τ
        logits = torch.mm(q, p.t()) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(B, device=q.device)

        # InfoNCE = cross-entropy
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            # Positive cosines (diagonal)
            pos_cos = torch.diag(logits).mean() * self.temperature

            # Negative cosines (off-diagonal)
            mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
            neg_cos = logits[mask].mean() * self.temperature

            # Separation
            separation = pos_cos - neg_cos

        metrics = {
            'loss': loss.item(),
            'pos_cos': pos_cos.item(),
            'neg_cos': neg_cos.item(),
            'separation': separation.item(),
        }

        return loss, metrics


class TwoTowerDataset(Dataset):
    """Dataset for two-tower training."""

    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = torch.from_numpy(self.contexts[idx]).float()
        tgt = torch.from_numpy(self.targets[idx]).float()
        return ctx, tgt


def train_epoch(q_tower, p_tower, dataloader, loss_fn, optimizer, device):
    """Train one epoch."""
    print("    train_epoch: Setting models to train mode...")
    q_tower.train()
    p_tower.train()

    total_loss = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_sep = 0.0
    num_batches = 0

    print("    train_epoch: Starting DataLoader iteration...")
    for contexts, targets in dataloader:
        print(f"    Batch {num_batches + 1}...")
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Forward
        q = q_tower(contexts)
        p = p_tower(targets)

        # Loss
        loss, metrics = loss_fn(q, p)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += metrics['loss']
        total_pos += metrics['pos_cos']
        total_neg += metrics['neg_cos']
        total_sep += metrics['separation']
        num_batches += 1

    return {
        'train_loss': total_loss / num_batches,
        'train_pos_cos': total_pos / num_batches,
        'train_neg_cos': total_neg / num_batches,
        'train_separation': total_sep / num_batches,
    }


@torch.no_grad()
def validate(q_tower, p_tower, dataloader, device):
    """Validate on held-out data."""
    q_tower.eval()
    p_tower.eval()

    all_q = []
    all_p = []

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        q = q_tower(contexts)
        p = p_tower(targets)

        all_q.append(q.cpu())
        all_p.append(p.cpu())

    all_q = torch.cat(all_q, dim=0)
    all_p = torch.cat(all_p, dim=0)

    # Compute cosine similarity
    cos = F.cosine_similarity(all_q, all_p, dim=1)

    return {
        'val_cosine': float(cos.mean()),
        'val_cosine_std': float(cos.std()),
    }


def main():
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument('--arch-q', type=str, default='mamba_s',
                        help='Q tower backbone')
    parser.add_argument('--arch-p', type=str, default='mamba_s',
                        help='P tower backbone')
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--d-state', type=int, default=128)
    parser.add_argument('--conv-sz', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Loss
    parser.add_argument('--temperature', type=float, default=0.07)

    # Data
    parser.add_argument('--train-npz', type=str, required=True)
    parser.add_argument('--val-split', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=500)

    # System
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--eval-every', type=int, default=1)

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TWO-TOWER RETRIEVER TRAINING")
    print("=" * 80)
    print(f"Q tower: {args.arch_q}")
    print(f"P tower: {args.arch_p}")
    print(f"Loss: InfoNCE (τ={args.temperature})")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    data = np.load(args.train_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Split
    n_val = int(len(contexts) * args.val_split)
    n_train = len(contexts) - n_val
    indices = np.random.permutation(len(contexts))

    train_dataset = TwoTowerDataset(
        contexts[indices[:n_train]],
        targets[indices[:n_train]]
    )
    val_dataset = TwoTowerDataset(
        contexts[indices[n_train:]],
        targets[indices[n_train:]]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print()

    # Create towers
    print("Creating towers...")
    q_tower = QueryTower(
        backbone_type=args.arch_q,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        conv_sz=args.conv_sz,
        expand=args.expand,
        dropout=args.dropout,
    ).to(args.device)

    p_tower = PayloadTower(
        backbone_type=args.arch_p,
        d_model=args.d_model,
    ).to(args.device)

    print(f"  Q params: {sum(p.numel() for p in q_tower.parameters()):,}")
    print(f"  P params: {sum(p.numel() for p in p_tower.parameters()):,}")
    print()

    # Loss
    loss_fn = TwoTowerInfoNCE(temperature=args.temperature)

    # Optimizer (both towers)
    optimizer = torch.optim.AdamW(
        list(q_tower.parameters()) + list(p_tower.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    print("Computing total steps...")
    total_steps = len(train_loader) * args.epochs
    print(f"  Total steps: {total_steps}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup_steps
    )
    print("✓ Scheduler created")

    # Training loop
    history = []

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n>>> Epoch {epoch} starting...")
        epoch_start = time.time()

        # Train
        print("  Calling train_epoch...")
        train_metrics = train_epoch(
            q_tower, p_tower, train_loader, loss_fn, optimizer, args.device
        )
        print("  train_epoch returned")

        # Validate
        val_metrics = validate(q_tower, p_tower, val_loader, args.device)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train Δ (pos-neg): {train_metrics['train_separation']:.4f}")
        print(f"  Val cosine: {val_metrics['val_cosine']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'q_tower_state_dict': q_tower.state_dict(),
            'p_tower_state_dict': p_tower.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(args),
        }

        # Convert Paths to strings
        checkpoint['args']['save_dir'] = str(checkpoint['args']['save_dir'])
        checkpoint['args']['train_npz'] = str(checkpoint['args']['train_npz'])

        torch.save(checkpoint, Path(args.save_dir) / f'epoch{epoch}.pt')
        print(f"  ✅ Saved epoch{epoch}.pt")
        print()

        # Step scheduler
        for _ in range(len(train_loader)):
            scheduler.step()

    # Save history
    with open(Path(args.save_dir) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
