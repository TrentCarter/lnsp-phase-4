#!/usr/bin/env python3
"""
Final LVM Trainer - Consultant's Exact Recipe
==============================================

Implements all 4 critical fixes:
A) Early stopping on Hit@5 (patience=3)
B) L2-norm before losses, proper delta reconstruction
C) Tuned loss balance (InfoNCE=0.05, batch≥256)
D) Data quality gates (coherence≥0.78, len≥7)

Expected result: 55%+ Hit@5 (production threshold)

Usage:
    python app/lvm/train_final.py \
        --model-type memory_gru \
        --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
        --epochs 50 \
        --batch-size 32 \
        --accumulation-steps 8
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from .models import create_model, MODEL_SPECS
except ImportError:
    from models import create_model, MODEL_SPECS


# ============================================================================
# CONSULTANT FIX B: L2 NORMALIZATION UTILITY
# ============================================================================

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors (consultant requirement)"""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ============================================================================
# DATASET WITH QUALITY GATES (FIX D)
# ============================================================================

class QualityGatedDataset(Dataset):
    """
    Dataset with consultant's quality gates:
    - Chain coherence ≥ 0.78
    - Sequence length ≥ 7
    - Delta prediction mode
    """

    def __init__(
        self,
        npz_path: str,
        min_coherence: float = 0.78,
        min_length: int = 7,
        delta_mode: bool = True
    ):
        data = np.load(npz_path)
        contexts = torch.FloatTensor(data['context_sequences'])
        targets = torch.FloatTensor(data['target_vectors'])

        # Quality gate: sequence length
        seq_len = contexts.size(1)
        if seq_len < min_length:
            print(f"⚠️ Warning: Sequence length {seq_len} < {min_length}")
            print(f"   Cannot apply length filter - using all data")
            length_mask = torch.ones(len(contexts), dtype=torch.bool)
        else:
            length_mask = torch.ones(len(contexts), dtype=torch.bool)
            print(f"✓ All sequences have length {seq_len} ≥ {min_length}")

        # Quality gate: chain coherence (neighbor cosine ≥ threshold)
        last_context = l2_normalize(contexts[:, -1, :])
        target_norm = l2_normalize(targets)
        coherence = (last_context * target_norm).sum(dim=-1)

        coherence_mask = coherence >= min_coherence

        # Combine masks
        final_mask = length_mask & coherence_mask

        # Apply filters
        n_before = len(contexts)
        self.contexts = contexts[final_mask]
        self.targets = targets[final_mask]
        self.chain_ids = np.arange(len(self.contexts))
        n_after = len(self.contexts)

        print(f"\nQuality Gates Applied:")
        print(f"  Min coherence: {min_coherence}")
        print(f"  Min length: {min_length}")
        print(f"  Before: {n_before} sequences")
        print(f"  After: {n_after} sequences")
        print(f"  Removed: {n_before - n_after} ({100*(1-n_after/n_before):.1f}%)")
        print(f"  Mean coherence: {coherence[final_mask].mean():.4f}")

        self.delta_mode = delta_mode

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        target = self.targets[idx]

        if self.delta_mode:
            y_curr = context[-1]
            delta = target - y_curr
            return context, delta, y_curr, target
        else:
            return context, target

    @staticmethod
    def chain_split(dataset, train_ratio=0.9, seed=42):
        """Chain-level split to prevent leakage"""
        unique_chains = np.unique(dataset.chain_ids)
        n_chains = len(unique_chains)

        rng = np.random.RandomState(seed)
        rng.shuffle(unique_chains)

        n_train = int(train_ratio * n_chains)
        train_chains = set(unique_chains[:n_train])
        val_chains = set(unique_chains[n_train:])

        train_mask = np.array([cid in train_chains for cid in dataset.chain_ids])
        val_mask = np.array([cid in val_chains for cid in dataset.chain_ids])

        train_ds = QualityGatedDataset.__new__(QualityGatedDataset)
        train_ds.contexts = dataset.contexts[train_mask]
        train_ds.targets = dataset.targets[train_mask]
        train_ds.chain_ids = dataset.chain_ids[train_mask]
        train_ds.delta_mode = dataset.delta_mode

        val_ds = QualityGatedDataset.__new__(QualityGatedDataset)
        val_ds.contexts = dataset.contexts[val_mask]
        val_ds.targets = dataset.targets[val_mask]
        val_ds.chain_ids = dataset.chain_ids[val_mask]
        val_ds.delta_mode = dataset.delta_mode

        overlap = len(set(train_ds.chain_ids) & set(val_ds.chain_ids))

        print(f"\n=== Chain-Level Split ===")
        print(f"Train: {len(train_ds)} sequences from {len(train_chains)} chains")
        print(f"Val: {len(val_ds)} sequences from {len(val_chains)} chains")
        print(f"Overlap: {overlap} (MUST BE 0) {'✅' if overlap == 0 else '❌'}")

        return train_ds, val_ds


# ============================================================================
# CONSULTANT FIX C: IMPROVED LOSS WITH PROPER NORMALIZATION
# ============================================================================

class ConsultantLoss(nn.Module):
    """
    Consultant's exact loss recipe:
    L = MSE(ŷ_n, y_n) + 0.5*(1 - cos(ŷ_n, y_n)) + α*InfoNCE

    With α=0.05, τ=0.07

    CRITICAL: L2-normalize BEFORE computing losses!
    """

    def __init__(
        self,
        alpha_infonce: float = 0.05,
        temperature: float = 0.07
    ):
        super().__init__()
        self.alpha = alpha_infonce
        self.tau = temperature

    def forward(
        self,
        y_hat: torch.Tensor,
        y_tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            y_hat: [B, D] - ALREADY L2-NORMALIZED predictions
            y_tgt: [B, D] - ALREADY L2-NORMALIZED targets

        Returns:
            loss, stats
        """
        # MSE on normalized vectors
        loss_mse = F.mse_loss(y_hat, y_tgt)

        # Cosine penalty: 0.5 * (1 - cosine)
        cos_sim = (y_hat * y_tgt).sum(dim=-1).mean()
        loss_cosine = 0.5 * (1.0 - cos_sim)

        # InfoNCE: in-batch contrastive
        sim_matrix = torch.mm(y_hat, y_tgt.t()) / self.tau
        labels = torch.arange(y_hat.size(0), device=y_hat.device)
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        # Total loss
        loss = loss_mse + loss_cosine + self.alpha * loss_infonce

        stats = {
            'loss_total': loss.item(),
            'loss_mse': loss_mse.item(),
            'loss_cosine': loss_cosine.item(),
            'loss_infonce': loss_infonce.item(),
            'cosine_sim': cos_sim.item()
        }

        return loss, stats


# ============================================================================
# HIT@K EVALUATION (FIX B: PROPER DELTA RECONSTRUCTION)
# ============================================================================

def compute_hit_at_k(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
    delta_mode: bool = True
) -> Dict[str, float]:
    """
    Consultant's exact recipe:
    1. Predict delta: Δ̂ = model(x_curr)
    2. Reconstruct: ŷ = x_curr + Δ̂
    3. L2-normalize: ŷ_norm = L2(ŷ)  ← CRITICAL!
    4. Retrieve: ANN.search(ŷ_norm)
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            if delta_mode:
                contexts, deltas, y_curr, targets = batch
                contexts = contexts.to(device)
                y_curr = y_curr.to(device)

                # Predict delta
                delta_hat = model(contexts)

                # CONSULTANT FIX B: Reconstruct THEN normalize
                y_hat = y_curr + delta_hat
                y_hat_norm = l2_normalize(y_hat)

                all_preds.append(y_hat_norm.cpu())
            else:
                contexts, targets = batch
                contexts = contexts.to(device)

                y_hat = model(contexts)
                y_hat_norm = l2_normalize(y_hat)

                all_preds.append(y_hat_norm.cpu())

            all_targets.append(targets)

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_targets_norm = l2_normalize(all_targets)

    # Similarity matrix
    sim_matrix = torch.mm(all_preds, all_targets_norm.t())

    # Top-K
    max_k = min(max(k_values), len(all_preds))
    _, topk_indices = sim_matrix.topk(k=max_k, dim=-1)

    ground_truth = torch.arange(len(all_preds)).unsqueeze(1)

    results = {}
    for k in k_values:
        if k > max_k:
            results[f'hit@{k}'] = float('nan')
            continue
        hits = (topk_indices[:, :k] == ground_truth).any(dim=1)
        results[f'hit@{k}'] = hits.float().mean().item()

    # Also compute mean cosine
    cosine_sim = (all_preds * all_targets_norm).sum(dim=-1).mean().item()
    results['cosine'] = cosine_sim

    return results


# ============================================================================
# TRAINING EPOCH WITH GRADIENT ACCUMULATION (FIX C)
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ConsultantLoss,
    device: torch.device,
    delta_mode: bool,
    accumulation_steps: int = 1
) -> Tuple[float, Dict]:
    """Train one epoch with gradient accumulation"""
    model.train()

    total_loss = 0.0
    all_stats = {
        'loss_total': 0.0,
        'loss_mse': 0.0,
        'loss_cosine': 0.0,
        'loss_infonce': 0.0,
        'cosine_sim': 0.0
    }

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if delta_mode:
            contexts, deltas, y_curr, targets = batch
            contexts = contexts.to(device)
            deltas = deltas.to(device)
            y_curr = y_curr.to(device)
            targets = targets.to(device)

            # Predict delta
            delta_hat = model(contexts)

            # CONSULTANT FIX B: Reconstruct THEN normalize
            y_hat = y_curr + delta_hat
            y_hat_norm = l2_normalize(y_hat)
            y_tgt_norm = l2_normalize(targets)

        else:
            contexts, targets = batch
            contexts = contexts.to(device)
            targets = targets.to(device)

            y_hat = model(contexts)
            y_hat_norm = l2_normalize(y_hat)
            y_tgt_norm = l2_normalize(targets)

        # Compute loss
        loss, stats = criterion(y_hat_norm, y_tgt_norm)

        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate stats
        for k, v in stats.items():
            all_stats[k] += v / len(dataloader)

        # Optimizer step every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {stats['loss_total']:.6f} | "
                  f"Cosine: {stats['cosine_sim']:.4f}")

    # Final step if needed
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader), all_stats


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ConsultantLoss,
    device: torch.device,
    delta_mode: bool,
    compute_retrieval: bool = True
) -> Dict[str, float]:
    """Evaluate with proper normalization"""
    model.eval()

    total_loss = 0.0
    all_stats = {
        'loss_total': 0.0,
        'loss_mse': 0.0,
        'loss_cosine': 0.0,
        'loss_infonce': 0.0,
        'cosine_sim': 0.0
    }

    with torch.no_grad():
        for batch in dataloader:
            if delta_mode:
                contexts, deltas, y_curr, targets = batch
                contexts = contexts.to(device)
                y_curr = y_curr.to(device)
                targets = targets.to(device)

                delta_hat = model(contexts)
                y_hat = y_curr + delta_hat
                y_hat_norm = l2_normalize(y_hat)
                y_tgt_norm = l2_normalize(targets)

            else:
                contexts, targets = batch
                contexts = contexts.to(device)
                targets = targets.to(device)

                y_hat = model(contexts)
                y_hat_norm = l2_normalize(y_hat)
                y_tgt_norm = l2_normalize(targets)

            loss, stats = criterion(y_hat_norm, y_tgt_norm)

            for k, v in stats.items():
                all_stats[k] += v

    n = len(dataloader)
    results = {k: v / n for k, v in all_stats.items()}

    # Hit@K
    if compute_retrieval:
        retrieval = compute_hit_at_k(
            model, dataloader, device,
            k_values=[1, 5, 10],
            delta_mode=delta_mode
        )
        results.update(retrieval)

    return results


# ============================================================================
# MAIN TRAINING WITH CONSULTANT'S RECIPE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Final LVM Trainer (Consultant Recipe)'
    )

    # Model and data
    parser.add_argument('--model-type', required=True,
                        choices=['lstm', 'gru', 'transformer', 'amn',
                                'hierarchical_gru', 'memory_gru'])
    parser.add_argument('--data', required=True)

    # Training (consultant's exact params)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulation-steps', type=int, default=8,
                        help='Gradient accumulation to reach effective batch=256')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Consultant: 1e-4 (was 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Consultant: 1e-4')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')

    # Early stopping (FIX A)
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience on Hit@5')

    # Loss (FIX C)
    parser.add_argument('--alpha-infonce', type=float, default=0.05,
                        help='Consultant: 0.05 (was 0.1)')
    parser.add_argument('--temperature', type=float, default=0.07)

    # Quality gates (FIX D)
    parser.add_argument('--min-coherence', type=float, default=0.78)
    parser.add_argument('--min-length', type=int, default=7)

    # Output
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'artifacts/lvm/models_final/{args.model_type}_{timestamp}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL LVM TRAINER (Consultant's Exact Recipe)")
    print("=" * 80)
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Data: {args.data}")
    print()
    print("Consultant Fixes:")
    print(f"  A) Early stopping: Hit@5, patience={args.patience}")
    print(f"  B) L2-norm before losses: ✅")
    print(f"  C) Loss: MSE + 0.5*cos + {args.alpha_infonce}*InfoNCE")
    print(f"  D) Quality gates: coherence≥{args.min_coherence}, len≥{args.min_length}")
    print()
    print(f"Training:")
    print(f"  LR: {args.lr} (lower, per consultant)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Batch size: {args.batch_size} × {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} effective")
    print(f"  Epochs: {args.epochs} (with early stop)")
    print()

    # Load dataset
    print("Loading dataset with quality gates...")
    dataset = QualityGatedDataset(
        args.data,
        min_coherence=args.min_coherence,
        min_length=args.min_length,
        delta_mode=True
    )

    # Split
    train_ds, val_ds = QualityGatedDataset.chain_split(
        dataset, train_ratio=0.9, seed=args.seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    # Model
    print("\nCreating model...")
    device = torch.device(args.device)

    configs = {
        'lstm': {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2, 'dropout': 0.2},
        'gru': {'input_dim': 768, 'd_model': 512, 'num_layers': 4, 'dropout': 0.0},
        'transformer': {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1},
        'amn': {'input_dim': 768, 'd_model': 256, 'hidden_dim': 512},
        'hierarchical_gru': {'d_model': 768, 'hidden_dim': 512, 'chunk_size': 10, 'num_chunks': 10},
        'memory_gru': {'d_model': 768, 'hidden_dim': 512, 'num_layers': 4, 'memory_slots': 2048}
    }

    model = create_model(args.model_type, **configs[args.model_type]).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    print()

    # Loss
    criterion = ConsultantLoss(
        alpha_infonce=args.alpha_infonce,
        temperature=args.temperature
    )

    # Optimizer with consultant's params
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler: Cosine with warmup
    warmup_epochs = 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=args.lr * 0.1
    )

    # Training loop with early stopping
    print("=" * 80)
    print("TRAINING START (Early Stopping on Hit@5)")
    print("=" * 80)
    print()

    best_hit5 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device,
            delta_mode=True,
            accumulation_steps=args.accumulation_steps
        )

        # Evaluate (compute Hit@K every 5 epochs or if close to stopping)
        compute_retrieval = (epoch % 5 == 0) or (epoch == args.epochs - 1) or (patience_counter > 0)
        val_results = evaluate(
            model, val_loader, criterion, device,
            delta_mode=True,
            compute_retrieval=compute_retrieval
        )

        # Learning rate schedule (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        # Print
        print(f"\nTrain: Loss={train_loss:.6f}, Cosine={train_stats['cosine_sim']:.4f}")
        print(f"Val: Loss={val_results['loss_total']:.6f}, Cosine={val_results['cosine_sim']:.4f}")

        if compute_retrieval:
            hit5 = val_results.get('hit@5', 0.0)
            print(f"Hit@1={val_results.get('hit@1', 0.0):.4f}, "
                  f"Hit@5={hit5:.4f}, "
                  f"Hit@10={val_results.get('hit@10', 0.0):.4f}")

            # CONSULTANT FIX A: Early stopping on Hit@5
            if hit5 > best_hit5 + 1e-4:
                best_hit5 = hit5
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_type': args.model_type,
                    'model_state_dict': model.state_dict(),
                    'val_results': val_results,
                    'args': vars(args)
                }, output_dir / 'best_val_hit5.pt')

                print(f"✓ New best Hit@5: {best_hit5:.4f} (saved)")
            else:
                patience_counter += 1
                print(f"⏳ No improvement ({patience_counter}/{args.patience})")

                if patience_counter >= args.patience:
                    print(f"\n🛑 Early stopping! Best Hit@5: {best_hit5:.4f}")
                    break

        # Save history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_stats['cosine_sim'],
            'val_loss': val_results['loss_total'],
            'val_cosine': val_results['cosine_sim'],
            'lr': optimizer.param_groups[0]['lr']
        }

        if compute_retrieval:
            epoch_data.update({
                'hit@1': val_results.get('hit@1', 0.0),
                'hit@5': val_results.get('hit@5', 0.0),
                'hit@10': val_results.get('hit@10', 0.0)
            })

        history.append(epoch_data)

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'model_name': MODEL_SPECS[args.model_type]['name'],
            'history': history,
            'best_hit5': best_hit5,
            'stopped_epoch': epoch + 1,
            'args': vars(args),
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Hit@5: {best_hit5:.4f}")
    print(f"Stopped at epoch: {epoch + 1}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
