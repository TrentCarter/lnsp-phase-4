#!/usr/bin/env python3
"""
Pure InfoNCE Training (Phase 5.2 - Contractor Option 1)
========================================================

PURE InfoNCE ONLY - No AR loss, no projection head.

Key changes:
1. NO projection head
2. NO AR cosine loss (was causing conflict)
3. Pure InfoNCE on raw 768D L2-normalized vectors
4. 1-epoch smoke test → if R@5 still 0%, pivot to two-tower
5. Same regularization (article dropout, span corruption)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python app/lvm/train_mamba_contrastive_v2.py \
        --model-type mamba_s \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --d-model 768 --n-layers 8 --d-state 128 \
        --batch-size 256 --grad-accum-steps 4 \
        --lambda-con 0.85 --lambda-ar 0.15 \
        --device mps \
        --save-dir artifacts/lvm/models/mamba_s_contrastive_v2
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model, count_parameters


class PureInfoNCELoss(nn.Module):
    """Pure InfoNCE only (no AR loss, no projection head)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 768] model predictions
            target: [B, 768] ground truth vectors

        Returns:
            loss, loss_dict
        """
        B = pred.shape[0]

        # InfoNCE contrastive loss (on raw 768D vectors)
        # Normalize predictions and targets
        h_pred = F.normalize(pred, p=2, dim=-1)  # [B, 768]

        # Stop-grad on targets
        with torch.no_grad():
            h_target = F.normalize(target, p=2, dim=-1)  # [B, 768]

        # Compute all pairwise similarities (full batch negatives)
        # h_pred @ h_target.T = [B, B]
        logits = torch.mm(h_pred, h_target.t()) / self.temperature  # [B, B]

        # Labels: diagonal elements are positives (i→i)
        labels = torch.arange(B, device=pred.device)

        # InfoNCE = NT-Xent = cross-entropy with positives on diagonal
        loss = F.cross_entropy(logits, labels)

        # Compute AR cosine for monitoring only (not in loss)
        with torch.no_grad():
            ar_cosine = F.cosine_similarity(pred, target, dim=-1).mean().item()

        loss_dict = {
            'infonce': loss.item(),
            'ar_cosine_monitor': ar_cosine,  # For tracking, not optimized
            'total': loss.item(),
        }

        return loss, loss_dict


class SequenceDataset(Dataset):
    """Dataset with article dropout and span corruption."""

    def __init__(
        self,
        context_sequences,
        target_vectors,
        article_dropout: float = 0.2,
        span_corruption: float = 0.1,
    ):
        self.contexts = context_sequences
        self.targets = target_vectors
        self.article_dropout = article_dropout
        self.span_corruption = span_corruption

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = self.contexts[idx].copy()
        tgt = self.targets[idx].copy()

        # Article dropout: zero last k positions
        if np.random.rand() < self.article_dropout:
            k = np.random.randint(1, len(ctx) + 1)
            ctx[-k:] = 0.0

        # Span corruption: replace random position with different article
        if np.random.rand() < self.span_corruption and len(self.contexts) > 1:
            pos = np.random.randint(0, len(ctx))
            other_idx = np.random.randint(0, len(self.contexts))
            if other_idx != idx:
                ctx[pos] = self.contexts[other_idx][pos]

        return torch.from_numpy(ctx).float(), torch.from_numpy(tgt).float()


def train_epoch(model, dataloader, loss_fn, optimizer, device, grad_accum_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_infonce = 0.0
    total_ar_monitor = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for i, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Forward pass
        preds = model(contexts)
        if len(preds.shape) == 3:
            preds = preds[:, -1, :]

        # Loss
        loss, loss_dict = loss_fn(preds, targets)
        loss = loss / grad_accum_steps

        # Backward
        loss.backward()

        # Step optimizer every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict['total']
        total_infonce += loss_dict['infonce']
        total_ar_monitor += loss_dict['ar_cosine_monitor']
        num_batches += 1

    return {
        'train_loss': total_loss / num_batches,
        'train_infonce': total_infonce / num_batches,
        'train_ar_cosine_monitor': total_ar_monitor / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate on held-out data."""
    model.eval()
    cosines = []

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        preds = model(contexts)
        if len(preds.shape) == 3:
            preds = preds[:, -1, :]

        # Cosine similarity
        cos = F.cosine_similarity(preds, targets, dim=-1)
        cosines.append(cos.cpu().numpy())

    all_cosines = np.concatenate(cosines)
    return float(all_cosines.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='mamba_s')
    parser.add_argument('--train-npz', type=str, required=True)
    parser.add_argument('--val-split', type=float, default=0.2)

    # Model architecture
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--d-state', type=int, default=128)
    parser.add_argument('--conv-sz', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Pure InfoNCE loss (no AR)
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature')

    # Regularization
    parser.add_argument('--article-dropout', type=float, default=0.2)
    parser.add_argument('--span-corruption', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--early-stop-patience', type=int, default=3)

    # System
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=Path, required=True)

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PURE InfoNCE TRAINING (Contractor Option 1)")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Training data: {args.train_npz}")
    print(f"Save dir: {args.save_dir}")
    print()
    print("Loss Function:")
    print(f"  PURE InfoNCE only (NO AR loss)")
    print(f"  τ: {args.temperature}")
    print(f"  NO projection head (raw 768D L2-normalized)")
    print()
    print("Regularization:")
    print(f"  Article dropout: {args.article_dropout}")
    print(f"  Span corruption: {args.span_corruption}")
    print()
    print("Training:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accum steps: {args.grad_accum_steps}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum_steps}")
    print(f"  Epochs: {args.epochs} (1-epoch smoke test)")
    print(f"  Device: {args.device}")
    print()
    print("⚠️  SMOKE TEST: If R@5 = 0% after epoch 1, pivot to two-tower (Option 4)")
    print("=" * 80)
    print()

    # Load data
    print("Loading training data...")
    data = np.load(args.train_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']
    print(f"  Loaded {len(contexts)} sequences")
    print(f"  Context shape: {contexts.shape}")
    print(f"  Target shape: {targets.shape}")

    # Train/val split
    n_val = int(len(contexts) * args.val_split)
    n_train = len(contexts) - n_val
    indices = np.random.permutation(len(contexts))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = SequenceDataset(
        contexts[train_indices],
        targets[train_indices],
        article_dropout=args.article_dropout,
        span_corruption=args.span_corruption,
    )
    val_dataset = SequenceDataset(
        contexts[val_indices],
        targets[val_indices],
        article_dropout=0.0,  # No augmentation for validation
        span_corruption=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    # Create model
    print("Creating model...")
    model = create_model(
        model_type=args.model_type,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        conv_sz=args.conv_sz,
        expand=args.expand,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    print(f"  Parameters: {count_parameters(model):,}")
    print()

    # Loss function (pure InfoNCE, no AR, no projection head!)
    loss_fn = PureInfoNCELoss(temperature=args.temperature)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing with warmup
    total_steps = len(train_loader) // args.grad_accum_steps * args.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    # Training loop
    best_val_cosine = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, args.device,
            grad_accum_steps=args.grad_accum_steps,
        )

        # Validate
        val_cosine = validate(model, val_loader, args.device)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'epoch': epoch,
            **train_metrics,
            'val_loss': 1.0 - val_cosine,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        history.append(metrics)

        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss (InfoNCE): {train_metrics['train_loss']:.4f}")
        print(f"  Train AR cosine (monitor): {train_metrics['train_ar_cosine_monitor']:.4f}")
        print(f"  Val cosine: {val_cosine:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save history
        with open(args.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save best checkpoint
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine': val_cosine,
                'val_loss': 1.0 - val_cosine,
                'args': vars(args),
            }

            # Convert Path to string for JSON serialization
            checkpoint['args']['save_dir'] = str(checkpoint['args']['save_dir'])
            checkpoint['args']['train_npz'] = str(checkpoint['args']['train_npz'])

            # Use Path() since args.save_dir is still Path type here
            torch.save(checkpoint, Path(args.save_dir) / 'best.pt')
            print(f"  ✅ New best: {val_cosine:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

        print()

        # Step scheduler
        for _ in range(len(train_loader) // args.grad_accum_steps):
            scheduler.step()

    # Save final args
    args_dict = vars(args).copy()
    args_dict['save_dir'] = str(args_dict['save_dir'])
    args_dict['train_npz'] = str(args_dict['train_npz'])
    with open(Path(args.save_dir) / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best val_cosine: {best_val_cosine:.4f}")
    print(f"Checkpoint: {args.save_dir / 'best.pt'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
