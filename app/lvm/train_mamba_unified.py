#!/usr/bin/env python3
"""
Unified Training Script for Mamba LVM Models - Phase 5
======================================================

Trains all 5 Mamba variants with consistent setup:
- Loss: 0.7 * cosine + 0.3 * MSE (+ optional InfoNCE)
- Optimizer: AdamW with warmup + cosine decay
- Data: Wikipedia sequences (context=5)

Usage Examples:
    # Model A: Mamba-S
    python app/lvm/train_mamba_unified.py --model-type mamba_s \
        --d-model 256 --n-layers 8 --d-state 128 --device mps

    # Model B: Mamba-H (Hybrid)
    python app/lvm/train_mamba_unified.py --model-type mamba_hybrid_local \
        --d-model 320 --n-layers 12 --d-state 128 \
        --local-attn-win 8 --local-attn-every 4 --device mps

    # Model C: Mamba-XL
    python app/lvm/train_mamba_unified.py --model-type mamba_xl \
        --d-model 384 --n-layers 16 --d-state 192 --batch-size 768 --device mps

    # Model D: Mamba-Sandwich
    python app/lvm/train_mamba_unified.py --model-type mamba_sandwich \
        --d-model 320 --n-layers-mamba 8 --n-layers-local 4 \
        --local-attn-win 8 --batch-size 896 --device mps

    # Model E: Mamba-GR
    python app/lvm/train_mamba_unified.py --model-type mamba_gr \
        --d-model 288 --n-layers 10 --d-state 144 --gru-hidden 256 --device mps
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

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model, count_parameters


class VectorSequenceDataset(Dataset):
    """Dataset for vector sequences with next-vector targets."""

    def __init__(self, npz_path: str):
        """Load vector sequences from NPZ file.

        Expected format:
            - 'context_sequences' or 'sequences': [N, context_len, 768]
            - 'target_vectors' or 'targets': [N, 768]
        """
        self.data = np.load(npz_path)

        # Support both naming conventions
        if 'context_sequences' in self.data:
            self.sequences = self.data['context_sequences']
        elif 'sequences' in self.data:
            self.sequences = self.data['sequences']
        else:
            raise KeyError("NPZ must contain 'context_sequences' or 'sequences'")

        if 'target_vectors' in self.data:
            self.targets = self.data['target_vectors']
        elif 'targets' in self.data:
            self.targets = self.data['targets']
        else:
            raise KeyError("NPZ must contain 'target_vectors' or 'targets'")

        # Convert to tensors
        self.sequences = torch.from_numpy(self.sequences).float()
        self.targets = torch.from_numpy(self.targets).float()

        print(f" Loaded {len(self.sequences)} sequences from {npz_path}")
        print(f"  Context shape: {self.sequences.shape}")
        print(f"  Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'target': self.targets[idx],
        }


class CombinedLoss(nn.Module):
    """Combined loss: 0.7 * cosine + 0.3 * MSE + optional InfoNCE."""

    def __init__(
        self,
        cosine_weight: float = 0.7,
        mse_weight: float = 0.3,
        use_infonce: bool = False,
        infonce_weight: float = 0.1,
        infonce_temp: float = 0.07,
        infonce_negs: int = 8,
    ):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.use_infonce = use_infonce
        self.infonce_weight = infonce_weight
        self.infonce_temp = infonce_temp
        self.infonce_negs = infonce_negs

    def forward(self, pred, target):
        """
        Args:
            pred: [B, L, 768] or [B, 768]
            target: [B, L, 768] or [B, 768]

        Returns:
            total_loss, loss_dict
        """
        # Take last timestep if sequence
        if pred.dim() == 3:
            pred = pred[:, -1, :]
        if target.dim() == 3:
            target = target[:, -1, :]

        # Cosine loss (1 - cosine_similarity)
        loss_cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

        # MSE loss
        loss_mse = F.mse_loss(pred, target)

        total_loss = self.cosine_weight * loss_cos + self.mse_weight * loss_mse

        loss_dict = {
            'cosine': loss_cos.item(),
            'mse': loss_mse.item(),
        }

        # Optional InfoNCE
        if self.use_infonce:
            B = pred.shape[0]
            pos_sim = F.cosine_similarity(pred, target, dim=-1) / self.infonce_temp

            # Sample negatives from batch
            neg_sims = []
            for _ in range(self.infonce_negs):
                neg_idx = torch.randperm(B, device=pred.device)
                neg_sim = F.cosine_similarity(pred, target[neg_idx], dim=-1) / self.infonce_temp
                neg_sims.append(neg_sim)

            neg_sims = torch.stack(neg_sims, dim=1)  # [B, num_negatives]
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=pred.device)
            loss_info = F.cross_entropy(logits, labels)

            total_loss += self.infonce_weight * loss_info
            loss_dict['infonce'] = loss_info.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {}

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)
        loss, loss_dict = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Accumulate loss components
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        # Log periodically
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {avg_loss:.6f}")

    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    all_cosines = []

    for batch in dataloader:
        sequences = batch['sequence'].to(device)
        targets = batch['target'].to(device)

        outputs = model(sequences)
        loss, loss_dict = criterion(outputs, targets)

        total_loss += loss.item()

        # Accumulate loss components
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        # Compute cosine similarity
        if outputs.dim() == 3:
            outputs = outputs[:, -1, :]
        if targets.dim() == 3:
            targets = targets[:, -1, :]

        cosines = F.cosine_similarity(outputs, targets, dim=-1)
        all_cosines.append(cosines.cpu())

    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    avg_cosine = torch.cat(all_cosines).mean().item()

    return avg_loss, avg_components, avg_cosine


def main():
    parser = argparse.ArgumentParser(description="Train Mamba LVM models (Phase 5)")

    # Model selection
    parser.add_argument(
        '--model-type', type=str, required=True,
        choices=['mamba_s', 'mamba_hybrid_local', 'mamba_xl', 'mamba_sandwich', 'mamba_gr'],
        help='Model architecture to train'
    )

    # Model parameters (shared)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--d-state', type=int, default=128)
    parser.add_argument('--conv-sz', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.05)

    # Hybrid-specific
    parser.add_argument('--local-attn-win', type=int, default=8)
    parser.add_argument('--local-attn-every', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)

    # Sandwich-specific
    parser.add_argument('--n-layers-mamba', type=int, default=8)
    parser.add_argument('--n-layers-local', type=int, default=4)

    # GR-specific
    parser.add_argument('--gru-hidden', type=int, default=256)

    # Alignment head
    parser.add_argument('--use-alignment-head', action='store_true')
    parser.add_argument('--alignment-alpha', type=float, default=0.25)

    # Data
    parser.add_argument('--train-npz', type=str, default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)

    # Loss
    parser.add_argument('--cosine-weight', type=float, default=0.7)
    parser.add_argument('--mse-weight', type=float, default=0.3)
    parser.add_argument('--use-infonce', action='store_true')
    parser.add_argument('--infonce-weight', type=float, default=0.1)

    # Device
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'])

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='artifacts/lvm/models')
    parser.add_argument('--early-stop-patience', type=int, default=3)

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir) / args.model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Training Mamba LVM: {args.model_type}")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2))
    print("=" * 80)

    # Device
    device = torch.device(args.device)
    print(f"\n Using device: {device}")

    # Load data
    print("\nLoading data...")
    full_dataset = VectorSequenceDataset(args.train_npz)

    # Split train/val
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")

    # Create model
    print("\nCreating model...")
    model_kwargs = {
        'd_model': args.d_model,
        'dropout': args.dropout,
        'use_alignment_head': args.use_alignment_head,
        'alignment_alpha': args.alignment_alpha,
    }

    if args.model_type == 'mamba_s':
        model_kwargs.update({
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'conv_sz': args.conv_sz,
            'expand': args.expand,
        })
    elif args.model_type == 'mamba_hybrid_local':
        model_kwargs.update({
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'conv_sz': args.conv_sz,
            'expand': args.expand,
            'local_attn_win': args.local_attn_win,
            'local_attn_every': args.local_attn_every,
            'n_heads': args.n_heads,
        })
    elif args.model_type == 'mamba_xl':
        model_kwargs.update({
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'conv_sz': args.conv_sz,
            'expand': args.expand,
        })
    elif args.model_type == 'mamba_sandwich':
        model_kwargs.update({
            'n_layers_mamba': args.n_layers_mamba,
            'n_layers_local': args.n_layers_local,
            'd_state': args.d_state,
            'conv_sz': args.conv_sz,
            'expand': args.expand,
            'local_attn_win': args.local_attn_win,
            'n_heads': args.n_heads,
        })
    elif args.model_type == 'mamba_gr':
        model_kwargs.update({
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'conv_sz': args.conv_sz,
            'expand': args.expand,
            'gru_hidden': args.gru_hidden,
        })

    model = create_model(args.model_type, **model_kwargs)
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Loss and optimizer
    criterion = CombinedLoss(
        cosine_weight=args.cosine_weight,
        mse_weight=args.mse_weight,
        use_infonce=args.use_infonce,
        infonce_weight=args.infonce_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    print(f"\n Training steps: {num_training_steps} (warmup: {num_warmup_steps})")

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    best_val_cosine = -1.0
    patience_counter = 0
    history = []

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Evaluate
        val_loss, val_components, val_cosine = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train loss: {train_loss:.6f} (cos: {train_components['cosine']:.4f}, mse: {train_components['mse']:.6f})")
        print(f"  Val loss: {val_loss:.6f} | Val cosine: {val_cosine:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        })

        # Save best model
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine': val_cosine,
                'val_loss': val_loss,
                'args': vars(args),
            }, save_dir / 'best.pt')

            print(f"  �  Saved best model (val_cosine={val_cosine:.4f})")
        else:
            patience_counter += 1
            print(f"  � No improvement ({patience_counter}/{args.early_stop_patience})")

            if patience_counter >= args.early_stop_patience:
                print("\n� Early stopping triggered!")
                break

    # Save final model
    torch.save(model.state_dict(), save_dir / 'final.pt')

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print(" Training complete!")
    print(f"  Best val cosine: {best_val_cosine:.4f}")
    print(f"  Saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
