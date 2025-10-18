#!/usr/bin/env python3
"""
Unified LVM Trainer - Train Any Architecture
=============================================

Train any of the 4 LVM architectures with consistent hyperparameters:
1. LSTM Baseline (~5M params)
2. GRU Stack (~7M params)
3. Transformer (~18M params)
4. Attention Mixture Network (~2M params) [RECOMMENDED for LNSP]

Usage:
    python app/lvm/train_unified.py --model-type amn --epochs 20
    python app/lvm/train_unified.py --model-type transformer --epochs 20
    python app/lvm/train_unified.py --model-type lstm --epochs 20
    python app/lvm/train_unified.py --model-type gru --epochs 20

Key Features:
- MSE loss by default (fixed from InfoNCE bug)
- Consistent train/val split (90/10)
- Model-specific hyperparameters
- Progress logging and checkpointing
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    *,
    anchors: torch.Tensor | None = None,
    anchor_sigma: float | None = None,
    lambda_mmd: float = 0.0,
    stats_mean: torch.Tensor | None = None,
    stats_std: torch.Tensor | None = None,
    lambda_stat: float = 0.0,
    cycle_cfg: CycleConfig | None = None,
    cycle_metrics: list[float] | None = None,
    rng: random.Random | None = None,
):
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    stats_acc = {
        "loss_info": 0.0,
        "loss_moment": 0.0,
        "loss_variance": 0.0,
        "loss_mse": 0.0,
        "loss_mmd": 0.0,
        "loss_stat": 0.0,
        "loss_cycle": 0.0,
    }
    rng = rng or random

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        pred_raw, pred_cos = model(contexts, return_raw=True)

        loss, stats = compute_losses(pred_raw, pred_cos, targets, loss_weights)
        for key, value in stats.items():
            if key in stats_acc:
                stats_acc[key] += value

        if lambda_mmd > 0.0 and anchors is not None and anchor_sigma is not None:
            idx = torch.randint(0, anchors.size(0), (pred_cos.size(0),), device=device)
            anchor_batch = anchors[idx]
            mmd = compute_mmd_rbf(pred_cos, anchor_batch, anchor_sigma)
            loss = loss + lambda_mmd * mmd
            stats_acc["loss_mmd"] += float(mmd.detach().item())

        if lambda_stat > 0.0 and stats_mean is not None and stats_std is not None:
            stat_penalty = mean_variance_penalty(pred_cos, stats_mean, stats_std)
            loss = loss + lambda_stat * stat_penalty
            stats_acc["loss_stat"] += float(stat_penalty.detach().item())

        if cycle_cfg is not None and cycle_cfg.enabled():
            cycle_penalty, cycle_cos = maybe_cycle_penalty(pred_raw[0], cycle_cfg, rng)
            if cycle_penalty is not None:
                loss = loss + cycle_penalty.to(device)
                stats_acc["loss_cycle"] += float(cycle_penalty.detach().item())
                if cycle_metrics is not None and cycle_cos is not None:
                    cycle_metrics.append(cycle_cos)

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


def get_model_config(model_type: str):
    """Get model-specific hyperparameters"""
    configs = {
        'lstm': {
            'input_dim': 768,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.2
        },
        'gru': {
            'input_dim': 768,
            'd_model': 512,
            'num_layers': 4,
            'dropout': 0.0
        },
        'transformer': {
            'input_dim': 768,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1
        },
        'amn': {
            'input_dim': 768,
            'd_model': 256,
            'hidden_dim': 512
        }
    }
    return configs.get(model_type, {})


def main():
    parser = argparse.ArgumentParser(description='Unified LVM Trainer')

    # Model selection
    parser.add_argument('--model-type', required=True, choices=['lstm', 'gru', 'transformer', 'amn'],
                        help='Model architecture to train')

    # Data and training
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', default=None, help='Output directory (auto-generated if not specified)')

    # Loss weights (MSE is primary)
    parser.add_argument('--lambda-mse', type=float, default=1.0, help='MSE loss weight (PRIMARY)')
    parser.add_argument('--lambda-info', type=float, default=0.0, help='InfoNCE loss weight (disabled by default)')
    parser.add_argument('--lambda-moment', type=float, default=0.0, help='Moment matching weight')
    parser.add_argument('--lambda-variance', type=float, default=0.0, help='Variance penalty weight')
    parser.add_argument('--tau', type=float, default=0.07, help='Temperature for InfoNCE')

    # Optional regularization
    parser.add_argument('--lambda-mmd', type=float, default=0.0)
    parser.add_argument('--mmd-anchors', type=int, default=0)
    parser.add_argument('--lambda-stat', type=float, default=0.0)

    # Cycle consistency (experimental)
    parser.add_argument('--cycle-pct', type=float, default=0.0)
    parser.add_argument('--cycle-lambda', type=float, default=0.0)
    parser.add_argument('--cycle-steps', type=int, default=1)
    parser.add_argument('--cycle-timeout', type=float, default=30.0)
    parser.add_argument('--decoder-endpoint', default='http://127.0.0.1:8766/decode')
    parser.add_argument('--encoder-endpoint', default='http://127.0.0.1:8767/embed')

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'artifacts/lvm/models/{args.model_type}_{timestamp}'

    print("=" * 80)
    print(f"Unified LVM Trainer - {args.model_type.upper()}")
    print("=" * 80)
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Description: {MODEL_SPECS[args.model_type]['description']}")
    print(f"Expected params: {MODEL_SPECS[args.model_type]['params']}")
    print()
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()
    print("Loss Configuration:")
    print(f"  MSE weight: {args.lambda_mse} (PRIMARY)")
    print(f"  InfoNCE weight: {args.lambda_info} (disabled)")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    dataset = VectorSequenceDataset(args.data)
    targets_np = dataset.targets.numpy()

    # Optional regularization setup
    anchor_tensor = None
    anchor_sigma = None
    if args.lambda_mmd > 0.0 and args.mmd_anchors > 0:
        anchor_tensor, anchor_sigma = sample_anchors(targets_np, args.mmd_anchors)
        print(f"Anchor set prepared: {anchor_tensor.shape[0]} vectors (sigma={anchor_sigma:.4f})")

    stats_mean_tensor = None
    stats_std_tensor = None
    if args.lambda_stat > 0.0:
        mean_np, std_np = compute_batch_stats(targets_np)
        stats_mean_tensor = torch.from_numpy(mean_np)
        stats_std_tensor = torch.from_numpy(std_np)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print()

    # Create model
    print("Creating model...")
    device = torch.device(args.device)
    model_config = get_model_config(args.model_type)
    model = create_model(args.model_type, **model_config).to(device)

    actual_params = model.count_parameters()
    print(f"Actual parameters: {actual_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if anchor_tensor is not None:
        anchor_tensor = anchor_tensor.to(device)
    if stats_mean_tensor is not None:
        stats_mean_tensor = stats_mean_tensor.to(device)
        stats_std_tensor = stats_std_tensor.to(device)

    cycle_cfg = CycleConfig(
        pct=args.cycle_pct,
        weight=args.cycle_lambda,
        steps=args.cycle_steps,
        decoder_endpoint=args.decoder_endpoint,
        encoder_endpoint=args.encoder_endpoint,
        timeout=args.cycle_timeout,
    )
    cycle_metrics: list[float] = []
    rng = random.Random(42)

    # Training loop
    best_val_loss = float('inf')
    history = []
    loss_weights = LossWeights(
        tau=args.tau,
        mse=args.lambda_mse,
        info_nce=args.lambda_info,
        moment=args.lambda_moment,
        variance=args.lambda_variance,
    )

    print("Starting training...")
    print()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_cosine, train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            anchors=anchor_tensor,
            anchor_sigma=anchor_sigma,
            lambda_mmd=args.lambda_mmd,
            stats_mean=stats_mean_tensor,
            stats_std=stats_std_tensor,
            lambda_stat=args.lambda_stat,
            cycle_cfg=cycle_cfg,
            cycle_metrics=cycle_metrics,
            rng=rng,
        )

        # Validate
        val_loss, val_cosine = evaluate(model, val_loader, device)

        # Learning rate schedule
        scheduler.step(val_loss)

        # Log
        mse_val = train_stats.get("loss_mse", 0.0)
        info_val = train_stats.get("loss_info", 0.0)

        print(
            "  Train Loss: {:.6f} | Train Cosine: {:.4f} | MSE: {:.6f}".format(
                train_loss,
                train_cosine,
                mse_val,
            )
        )
        print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")
        print()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            'train_loss_mse': train_stats.get('loss_mse', 0.0),
            'train_loss_info': train_stats.get('loss_info', 0.0),
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'model_config': model_config,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_type': args.model_type,
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
            'model_type': args.model_type,
            'model_name': MODEL_SPECS[args.model_type]['name'],
            'history': history,
            'best_val_loss': best_val_loss,
            'final_params': actual_params,
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("=" * 80)
    print("Training Complete!")
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final val cosine: {val_cosine:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
