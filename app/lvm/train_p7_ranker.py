"""
Training script for P7 "Directional Ranker" LVM

Implements ranking-based training with:
- InfoNCE contrastive loss
- Prev-repel margin
- Semantic anchoring
- Directional gating
- In-batch negatives

Author: Claude Code
Date: 2025-11-04
Status: P7 Architecture
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import P7 components
from app.lvm.models_p7_ranker import create_p7_model
from app.lvm.losses_ranking import (
    p7_combined_loss,
    compute_directional_metrics
)


class P7RankingDataset(Dataset):
    """Dataset for P7 ranking training with negative sampling"""

    def __init__(
        self,
        sequences_npz: str,
        context_length: int = 5,
        num_hard_negatives: int = 2
    ):
        """
        Args:
            sequences_npz: path to NPZ file with 'contexts' and 'targets'
            context_length: number of context vectors (default 5)
            num_hard_negatives: number of in-article hard negatives (default 2)
        """
        data = np.load(sequences_npz)
        self.contexts = torch.from_numpy(data['contexts']).float()  # (N, K, 768)
        self.targets = torch.from_numpy(data['targets']).float()  # (N, 768)

        assert self.contexts.shape[1] == context_length
        assert self.contexts.shape[0] == self.targets.shape[0]

        self.context_length = context_length
        self.num_hard_negatives = num_hard_negatives

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            context: (K, 768) - context sequence
            target_next: (768,) - next chunk (positive)
            target_prev: (768,) - previous chunk (hard negative)
            hard_negatives: (M, 768) - in-article hard negatives
        """
        context = self.contexts[idx]  # (K, 768)
        target_next = self.targets[idx]  # (768,)
        target_prev = context[-1]  # Last context vector is "previous"

        # Sample hard negatives from nearby sequences (same article if available)
        # For now, sample randomly from dataset (TODO: use article boundaries)
        hard_neg_indices = torch.randint(0, len(self), (self.num_hard_negatives,))
        hard_negatives = self.targets[hard_neg_indices]  # (M, 768)

        return {
            'context': context,
            'target_next': target_next,
            'target_prev': target_prev,
            'hard_negatives': hard_negatives
        }


def create_negative_pool(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Create negative pool combining:
    1. Previous chunk (hard negative)
    2. In-article hard negatives
    3. In-batch negatives (all other targets in batch)

    Args:
        batch: dictionary with 'target_next', 'target_prev', 'hard_negatives'
        device: torch device

    Returns:
        negatives: (B, N, 768) - negative samples pool
    """
    B = batch['target_next'].shape[0]

    # 1. Previous chunks (B, 768)
    prev_negs = batch['target_prev'].unsqueeze(1)  # (B, 1, 768)

    # 2. In-article hard negatives (B, M, 768)
    hard_negs = batch['hard_negatives']  # (B, M, 768)

    # 3. In-batch negatives: all other targets in batch
    # For each sample i, negatives are all j ≠ i
    targets = batch['target_next']  # (B, 768)

    # Build in-batch negatives manually (exclude self)
    inbatch_negs_list = []
    for i in range(B):
        # Get all targets except i-th
        neg_indices = torch.cat([torch.arange(i, device=targets.device),
                                torch.arange(i+1, B, device=targets.device)])
        negs_i = targets[neg_indices]  # (B-1, 768)
        inbatch_negs_list.append(negs_i)
    inbatch_negs = torch.stack(inbatch_negs_list, dim=0)  # (B, B-1, 768)

    # Combine all negatives
    all_negatives = torch.cat([prev_negs, hard_negs, inbatch_negs], dim=1)  # (B, 1+M+B-1, 768)

    return all_negatives


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    epoch: int,
    warmup_epochs: int = 2
) -> Dict[str, float]:
    """
    Train one epoch with P7 ranking objective

    Returns:
        metrics: dictionary with average training metrics
    """
    model.train()

    total_loss = 0.0
    total_loss_rank = 0.0
    total_loss_margin = 0.0
    total_loss_teacher = 0.0
    total_gate_weight = 0.0
    total_teacher_violations = 0

    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        context = batch_on_device['context']
        target_next = batch_on_device['target_next']
        target_prev = batch_on_device['target_prev']

        # Create negative pool
        negatives = create_negative_pool(batch_on_device, device)

        # Forward pass
        query = model(context)  # (B, 768) - anchored prediction

        # Compute P7 loss
        loss_dict = p7_combined_loss(
            query=query,
            positive=target_next,
            previous=target_prev,
            negatives=negatives,
            context_vectors=context,
            weights_dict=loss_weights,
            epoch=epoch,
            warmup_epochs=warmup_epochs
        )

        loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_loss_rank += loss_dict['loss_rank'].item()
        total_loss_margin += loss_dict['loss_margin'].item()
        total_loss_teacher += loss_dict['loss_teacher'].item()
        total_gate_weight += loss_dict['gate_weight_mean'].item()
        total_teacher_violations += loss_dict['n_teacher_violations']
        num_batches += 1

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, "
                  f"rank={loss_dict['loss_rank'].item():.4f}, "
                  f"margin={loss_dict['loss_margin'].item():.4f}")

    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'loss_rank': total_loss_rank / num_batches,
        'loss_margin': total_loss_margin / num_batches,
        'loss_teacher': total_loss_teacher / num_batches,
        'gate_weight_mean': total_gate_weight / num_batches,
        'teacher_violations_per_batch': total_teacher_violations / num_batches
    }

    return avg_metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate model with directional metrics

    Returns:
        metrics: dictionary with validation metrics
    """
    model.eval()

    all_cos_next = []
    all_cos_prev = []
    all_cos_anchor = []

    with torch.no_grad():
        for batch in dataloader:
            context = batch['context'].to(device)
            target_next = batch['target_next'].to(device)
            target_prev = batch['target_prev'].to(device)

            # Forward pass
            query = model(context)

            # Compute directional metrics
            metrics = compute_directional_metrics(
                query=query,
                positive=target_next,
                previous=target_prev,
                context_vectors=context
            )

            all_cos_next.append(metrics['cos_next'])
            all_cos_prev.append(metrics['cos_prev'])
            all_cos_anchor.append(metrics['cos_anchor'])

    # Average metrics
    avg_metrics = {
        'cos_next': np.mean(all_cos_next),
        'cos_prev': np.mean(all_cos_prev),
        'margin': np.mean(all_cos_next) - np.mean(all_cos_prev),
        'cos_anchor': np.mean(all_cos_anchor)
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train P7 Directional Ranker LVM')

    # Data
    parser.add_argument('--train-npz', required=True, help='Training sequences NPZ')
    parser.add_argument('--val-npz', required=True, help='Validation sequences NPZ')
    parser.add_argument('--context-length', type=int, default=5)

    # Model
    parser.add_argument('--model-type', default='transformer', choices=['transformer', 'lstm'])
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--anchor-lambda', type=float, default=0.8)
    parser.add_argument('--anchor-learnable', action='store_true', default=True)

    # Loss weights
    parser.add_argument('--w-rank', type=float, default=1.0)
    parser.add_argument('--w-margin', type=float, default=0.5)
    parser.add_argument('--w-teacher', type=float, default=0.2)
    parser.add_argument('--margin', type=float, default=0.07)
    parser.add_argument('--temperature', type=float, default=0.07)

    # Gating
    parser.add_argument('--gate-threshold', type=float, default=0.03)
    parser.add_argument('--gate-weak-weight', type=float, default=0.25)
    parser.add_argument('--floor-threshold', type=float, default=0.20)

    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--device', default='mps', choices=['cpu', 'mps', 'cuda'])

    # Output
    parser.add_argument('--output-dir', default='artifacts/lvm/models')
    parser.add_argument('--exp-name', default='p7_ranker')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load datasets
    print("Loading datasets...")
    train_dataset = P7RankingDataset(args.train_npz, context_length=args.context_length)
    val_dataset = P7RankingDataset(args.val_npz, context_length=args.context_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # Create model
    print(f"Creating {args.model_type} model with semantic anchoring...")
    model = create_p7_model(
        model_type=args.model_type,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        anchor_lambda_init=args.anchor_lambda,
        anchor_lambda_learnable=args.anchor_learnable
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get initial anchor lambda and convert if tensor
    initial_anchor_lambda = model.get_anchor_lambda()
    if isinstance(initial_anchor_lambda, torch.Tensor):
        initial_anchor_lambda = initial_anchor_lambda.item()
    print(f"Initial anchor λ: {initial_anchor_lambda:.3f}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Loss weights
    loss_weights = {
        'w_rank': args.w_rank,
        'w_margin': args.w_margin,
        'w_teacher': args.w_teacher,
        'margin': args.margin,
        'temperature': args.temperature,
        'gate_threshold': args.gate_threshold,
        'gate_weak_weight': args.gate_weak_weight,
        'floor_threshold': args.floor_threshold
    }

    print(f"\nLoss weights: {loss_weights}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    training_history = []
    best_margin = -float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            loss_weights, epoch, args.warmup_epochs
        )

        print(f"\nTrain metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Rank: {train_metrics['loss_rank']:.4f}")
        print(f"  Margin: {train_metrics['loss_margin']:.4f}")
        print(f"  Teacher: {train_metrics['loss_teacher']:.4f}")
        print(f"  Gate weight: {train_metrics['gate_weight_mean']:.3f}")

        # Get anchor lambda and convert if tensor
        anchor_lambda_print = model.get_anchor_lambda()
        if isinstance(anchor_lambda_print, torch.Tensor):
            anchor_lambda_print = anchor_lambda_print.item()
        print(f"  Anchor λ: {anchor_lambda_print:.3f}")

        # Validate
        val_metrics = validate(model, val_loader, device)

        print(f"\nVal metrics:")
        print(f"  cos(pred, next): {val_metrics['cos_next']:.4f}")
        print(f"  cos(pred, prev): {val_metrics['cos_prev']:.4f}")
        print(f"  Margin (Δ): {val_metrics['margin']:.4f}")
        print(f"  cos(pred, anchor): {val_metrics['cos_anchor']:.4f}")

        # Check for orthogonal drift
        if val_metrics['cos_anchor'] < 0.05:
            print("\n⚠️  WARNING: Orthogonal drift detected (cos_anchor < 0.05)!")
            print("   Predictions are drifting away from context subspace")

        # Save checkpoint
        anchor_lambda = model.get_anchor_lambda()
        if isinstance(anchor_lambda, torch.Tensor):
            anchor_lambda = anchor_lambda.item()

        epoch_data = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'anchor_lambda': float(anchor_lambda)
        }
        training_history.append(epoch_data)

        # Save best model
        if val_metrics['margin'] > best_margin:
            best_margin = val_metrics['margin']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': {
                    'model_type': args.model_type,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'anchor_lambda_init': args.anchor_lambda,
                    'anchor_lambda_learnable': args.anchor_learnable
                },
                'metrics': val_metrics
            }, output_dir / 'best_model.pt')
            print(f"✅ Saved best model (margin: {best_margin:.4f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'model_type': args.model_type,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'anchor_lambda_init': args.anchor_lambda,
            'anchor_lambda_learnable': args.anchor_learnable
        },
        'final_metrics': val_metrics
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Best margin: {best_margin:.4f}")
    print(f"Final margin: {val_metrics['margin']:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
