#!/usr/bin/env python3
"""
GraphMERT-LVM Training - 10k Benchmark
=======================================

Train 768-d native GraphMERT-LVM on 10k samples to benchmark training time.

Usage:
    # Single GPU
    python app/lvm/train_graphmert_lvm_benchmark.py --device cuda:0 --epochs 3

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=8 app/lvm/train_graphmert_lvm_benchmark.py --epochs 3

Key Features:
- 768-d native (no projection layer)
- MSE loss (vector prediction)
- Multi-GPU support via DDP
- Attention decay mask (λ=0.6)
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from graphmert_lvm_768d import GraphMERTLVM768D


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


def train_epoch(model, dataloader, optimizer, device, rank=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    num_batches = len(dataloader)

    epoch_start = time.time()

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_raw, pred_normalized = model(contexts, return_raw=True)

        # MSE loss (on raw predictions)
        loss = nn.functional.mse_loss(pred_raw, targets)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            cosine = cosine_similarity(pred_normalized, targets)
            total_cosine += cosine.item()

        # Log progress (rank 0 only)
        if rank == 0 and batch_idx % 50 == 0:
            elapsed = time.time() - epoch_start
            batches_done = batch_idx + 1
            batches_remaining = num_batches - batches_done
            eta = (elapsed / batches_done) * batches_remaining

            print(
                f"  Batch {batch_idx}/{num_batches} | "
                f"Loss: {loss.item():.6f} | "
                f"Cosine: {cosine.item():.4f} | "
                f"ETA: {eta:.0f}s"
            )

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    avg_cosine = total_cosine / num_batches

    return avg_loss, avg_cosine, epoch_time


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


def setup_ddp(rank, world_size):
    """Setup DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='GraphMERT-LVM 10k Benchmark')

    # Data and training
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5_10k.npz')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps')
    parser.add_argument('--output-dir', default='artifacts/lvm/models/graphmert_lvm_benchmark')

    # Model architecture
    parser.add_argument('--n-layers', type=int, default=12)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--d-ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lambda-decay', type=float, default=0.6)

    args = parser.parse_args()

    # Check for DDP
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device)

    if rank == 0:
        print("=" * 80)
        print("GraphMERT-LVM 10k Benchmark")
        print("=" * 80)
        print(f"Data: {args.data}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Device: {device}")
        print(f"World size: {world_size} GPUs")
        print(f"Output: {args.output_dir}")
        print()

    # Load data
    if rank == 0:
        print("Loading dataset...")
    dataset = VectorSequenceDataset(args.data)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create samplers for DDP
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )

    if rank == 0:
        print(f"Train: {len(train_dataset)} samples")
        print(f"Val: {len(val_dataset)} samples")
        print()

    # Create model
    if rank == 0:
        print("Creating GraphMERT-LVM 768D model...")

    model = GraphMERTLVM768D(
        d_model=768,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lambda_decay=args.lambda_decay
    ).to(device)

    if rank == 0:
        param_count = model.count_parameters()
        print(f"Parameters: {param_count:,}")
        print()

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training loop
    if rank == 0:
        print("Starting training...")
        print()

    history = []
    total_start = time.time()

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")

        # Set epoch for sampler (DDP)
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_cosine, epoch_time = train_epoch(
            model, train_loader, optimizer, device, rank
        )

        # Validate (rank 0 only)
        if rank == 0:
            val_loss, val_cosine = evaluate(model, val_loader, device)

            print(
                f"  Train Loss: {train_loss:.6f} | "
                f"Train Cosine: {train_cosine:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")
            print()

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_cosine': train_cosine,
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'epoch_time': epoch_time
            })

    total_time = time.time() - total_start

    # Save results (rank 0 only)
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_to_save = model.module if world_size > 1 else model
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
            'history': history
        }, output_dir / 'benchmark_model.pt')

        # Save training history
        benchmark_results = {
            'model_type': 'GraphMERT-LVM-768D',
            'dataset_size': len(dataset),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'world_size': world_size,
            'total_training_time': total_time,
            'avg_epoch_time': total_time / args.epochs,
            'parameters': param_count,
            'history': history,
            'trained_at': datetime.now().isoformat()
        }

        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        print("=" * 80)
        print("Benchmark Complete!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Avg epoch time: {total_time/args.epochs:.1f}s")
        print(f"Final val cosine: {history[-1]['val_cosine']:.4f}")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)

    # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()


if __name__ == '__main__':
    main()
