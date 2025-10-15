#!/usr/bin/env python3
"""
Transformer Baseline Trainer for Vector-Native LVM
===================================================

Trains a Transformer decoder to predict next vector from context.

Architecture:
- Input: sequence of 768D vectors (context window)
- Transformer decoder: 4 layers, d_model=512, 8 heads
- Output: next 768D vector prediction
- Loss: MSE between predicted and target vectors

Why Transformer:
- Self-attention captures long-range dependencies
- Parallel processing (faster than RNN)
- Foundation of modern LLMs (GPT, etc.)
- O(n²) complexity but good for short sequences
"""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        self.contexts = torch.FloatTensor(data['context_sequences'])  # [N, ctx_len, 768]
        self.targets = torch.FloatTensor(data['target_vectors'])      # [N, 768]

        print(f"Loaded {len(self.contexts)} training pairs")
        print(f"Context shape: {self.contexts.shape}")
        print(f"Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerVectorPredictor(nn.Module):
    """Transformer decoder for next-vector prediction"""

    def __init__(self, input_dim=768, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x, return_raw: bool = False):
        """
        Args:
            x: [batch, seq_len, 768] - context vectors
        Returns:
            [batch, 768] - predicted next vector (L2 normalized to match GTR-T5)
        """
        # Project to d_model
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask (prevent attending to future positions)
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer forward (using memory = x for self-attention)
        x = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask
        )  # [batch, seq_len, d_model]

        # Take last timestep
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Project to output
        raw = self.head(last_hidden)  # [batch, 768]

        cos = nn.functional.normalize(raw, p=2, dim=-1)
        if return_raw:
            return raw, cos
        return cos

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
                "  Batch {}/{} | Loss: {:.6f} | InfoNCE: {:.4f} | Cosine: {:.4f}".format(
                    batch_idx,
                    len(dataloader),
                    loss.item(),
                    stats.get("loss_info", 0.0),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='artifacts/lvm/models/transformer')
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--lambda-mse', type=float, default=0.0)
    parser.add_argument('--lambda-moment', type=float, default=0.0)
    parser.add_argument('--lambda-variance', type=float, default=0.0)
    parser.add_argument('--lambda-mmd', type=float, default=0.0)
    parser.add_argument('--mmd-anchors', type=int, default=0)
    parser.add_argument('--lambda-stat', type=float, default=0.0)
    parser.add_argument('--cycle-pct', type=float, default=0.0)
    parser.add_argument('--cycle-lambda', type=float, default=0.0)
    parser.add_argument('--cycle-steps', type=int, default=1)
    parser.add_argument('--cycle-timeout', type=float, default=30.0)
    parser.add_argument('--decoder-endpoint', default='http://127.0.0.1:8766/decode')
    parser.add_argument('--encoder-endpoint', default='http://127.0.0.1:8767/embed')
    args = parser.parse_args()

    print("=" * 80)
    print("Transformer Baseline Trainer")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    dataset = VectorSequenceDataset(args.data)
    targets_np = dataset.targets.numpy()

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
    model = TransformerVectorPredictor(
        input_dim=768,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
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
        moment=args.lambda_moment,
        variance=args.lambda_variance,
    )

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
        info_val = train_stats.get("loss_info", 0.0)
        moment_val = train_stats.get("loss_moment", 0.0)
        variance_val = train_stats.get("loss_variance", 0.0)
        mmd_val = train_stats.get("loss_mmd", 0.0)
        stat_val = train_stats.get("loss_stat", 0.0)
        cycle_val = train_stats.get("loss_cycle", 0.0)

        print(
            "  Train Loss: {:.6f} | Train Cosine: {:.4f} | InfoNCE: {:.4f} | Moment: {:.6f} | Var: {:.6f}".format(
                train_loss,
                train_cosine,
                info_val,
                moment_val,
                variance_val,
            )
        )
        if args.lambda_mmd > 0.0:
            print(f"    MMD: {mmd_val:.6f}")
        if args.lambda_stat > 0.0:
            print(f"    Stat penalty: {stat_val:.6f}")
        if cycle_cfg.enabled():
            mean_cycle = sum(cycle_metrics) / len(cycle_metrics) if cycle_metrics else 0.0
            print(f"    Cycle loss: {cycle_val:.6f} (avg cos {mean_cycle:.4f})")
        print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")
        print()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            'train_loss_info': train_stats.get('loss_info', 0.0),
            'train_loss_moment': train_stats.get('loss_moment', 0.0),
            'train_loss_variance': train_stats.get('loss_variance', 0.0),
            'train_loss_mse': train_stats.get('loss_mse', 0.0),
            'train_loss_mmd': train_stats.get('loss_mmd', 0.0),
            'train_loss_stat': train_stats.get('loss_stat', 0.0),
            'train_loss_cycle': train_stats.get('loss_cycle', 0.0),
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'history': history,
            'best_val_loss': best_val_loss,
            'final_params': model.count_parameters(),
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("=" * 80)
    print("Training Complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
