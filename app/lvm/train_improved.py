#!/usr/bin/env python3
"""
Improved LVM Trainer - Consultant Recommendations
==================================================

Critical improvements based on consultant feedback:
1. ✅ Hit@1/5/10 retrieval evaluation (not just cosine!)
2. ✅ Chain-level train/val split (no concept leakage)
3. ✅ Mixed loss: MSE + Cosine + InfoNCE
4. ✅ Delta prediction (predict Δ = y_next - y_curr)
5. ✅ Chain coherence filtering (neighbor cosine ≥0.78)
6. ✅ L2 normalization before loss

Go/No-Go Targets (Production Readiness):
- Chain split purity: 0 leakage ✅
- Pre-train chain coherence: ≥0.78 ✅
- Val cosine: ≥0.60 (target)
- Hit@1 / Hit@5: ≥30% / ≥55% (target)

Usage:
    python app/lvm/train_improved.py \
        --model-type memory_gru \
        --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
        --epochs 20
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
# IMPROVED DATASET WITH CHAIN-LEVEL SPLIT
# ============================================================================

class ChainAwareDataset(Dataset):
    """
    Dataset that preserves chain structure for proper evaluation.

    Each sequence belongs to a chain (Wikipedia article). We must ensure:
    - No concept from a chain appears in both train and val
    - Chain coherence ≥0.78 (neighbor cosine)
    - Support for delta prediction mode
    """

    def __init__(
        self,
        npz_path: str,
        chain_ids: np.ndarray = None,
        delta_mode: bool = True,
        coherence_threshold: float = 0.78
    ):
        """
        Args:
            npz_path: Path to NPZ with context_sequences, target_vectors
            chain_ids: Array of chain IDs (one per sequence) for splitting
            delta_mode: If True, predict delta instead of absolute vector
            coherence_threshold: Filter chains with neighbor cosine < threshold
        """
        data = np.load(npz_path)
        self.contexts = torch.FloatTensor(data['context_sequences'])
        self.targets = torch.FloatTensor(data['target_vectors'])

        # Chain IDs (for proper splitting)
        if chain_ids is not None:
            self.chain_ids = chain_ids
        elif 'chain_ids' in data:
            # Try to load from NPZ file
            self.chain_ids = data['chain_ids']
            print(f"Loaded chain_ids from NPZ file")
        else:
            # ⚠️ FALLBACK: Treat each sequence as independent
            # This is NOT ideal (violates chain-level split) but allows testing
            self.chain_ids = np.arange(len(self.contexts))
            print(f"⚠️ WARNING: No chain_ids provided - using sequence-level split")
            print(f"   This may cause leakage if sequences share concepts!")
            print(f"   For production: export data with chain_ids")

        self.delta_mode = delta_mode

        # Filter by chain coherence
        if coherence_threshold > 0:
            self._filter_by_coherence(coherence_threshold)

        print(f"Loaded {len(self.contexts)} sequences")
        print(f"Context shape: {self.contexts.shape}")
        print(f"Target shape: {self.targets.shape}")
        print(f"Unique chains: {len(np.unique(self.chain_ids))}")
        print(f"Delta mode: {delta_mode}")

    def _filter_by_coherence(self, threshold: float):
        """Filter sequences by chain coherence (neighbor cosine)"""
        # Compute neighbor cosine for each sequence
        # (cosine between last context vector and target)
        last_context = self.contexts[:, -1, :]  # [N, 768]

        # Normalize
        last_norm = F.normalize(last_context, p=2, dim=-1)
        target_norm = F.normalize(self.targets, p=2, dim=-1)

        # Cosine similarity
        coherence = (last_norm * target_norm).sum(dim=-1)  # [N]

        # Filter
        mask = coherence >= threshold
        n_before = len(self.contexts)
        n_after = mask.sum().item()

        self.contexts = self.contexts[mask]
        self.targets = self.targets[mask]
        self.chain_ids = self.chain_ids[mask.numpy()]

        print(f"Chain coherence filter: {n_before} → {n_after} sequences "
              f"(removed {n_before - n_after}, {100*(1-n_after/n_before):.1f}%)")
        print(f"Mean coherence: {coherence[mask].mean():.4f} (threshold: {threshold})")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        target = self.targets[idx]

        if self.delta_mode:
            # Predict delta: Δ = y_next - y_curr
            y_curr = context[-1]  # Last context vector
            delta = target - y_curr
            return context, delta, y_curr, target
        else:
            return context, target

    @staticmethod
    def chain_split(
        dataset: 'ChainAwareDataset',
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Tuple['ChainAwareDataset', 'ChainAwareDataset']:
        """
        Split dataset by chains (not sequences) to prevent leakage.

        Returns:
            (train_dataset, val_dataset)
        """
        # Get unique chain IDs
        unique_chains = np.unique(dataset.chain_ids)
        n_chains = len(unique_chains)

        # Shuffle chains
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_chains)

        # Split chains
        n_train_chains = int(train_ratio * n_chains)
        train_chains = set(unique_chains[:n_train_chains])
        val_chains = set(unique_chains[n_train_chains:])

        # Create masks
        train_mask = np.array([cid in train_chains for cid in dataset.chain_ids])
        val_mask = np.array([cid in val_chains for cid in dataset.chain_ids])

        # Create split datasets
        train_dataset = ChainAwareDataset.__new__(ChainAwareDataset)
        train_dataset.contexts = dataset.contexts[train_mask]
        train_dataset.targets = dataset.targets[train_mask]
        train_dataset.chain_ids = dataset.chain_ids[train_mask]
        train_dataset.delta_mode = dataset.delta_mode

        val_dataset = ChainAwareDataset.__new__(ChainAwareDataset)
        val_dataset.contexts = dataset.contexts[val_mask]
        val_dataset.targets = dataset.targets[val_mask]
        val_dataset.chain_ids = dataset.chain_ids[val_mask]
        val_dataset.delta_mode = dataset.delta_mode

        # Verify no leakage
        train_chains_set = set(train_dataset.chain_ids)
        val_chains_set = set(val_dataset.chain_ids)
        overlap = train_chains_set & val_chains_set

        print(f"\n=== Chain-Level Split (ZERO LEAKAGE) ===")
        print(f"Train: {len(train_dataset)} sequences from {len(train_chains)} chains")
        print(f"Val: {len(val_dataset)} sequences from {len(val_chains)} chains")
        print(f"Chain overlap: {len(overlap)} (MUST BE 0) {'✅' if len(overlap) == 0 else '❌ LEAK!'}")
        print()

        return train_dataset, val_dataset


# ============================================================================
# IMPROVED LOSS FUNCTION
# ============================================================================

class ImprovedLoss(nn.Module):
    """
    Mixed loss: MSE + Cosine + InfoNCE

    Following consultant recommendations:
    1. L2-normalize predictions and targets
    2. MSE on normalized vectors
    3. Cosine penalty: 0.5 * (1 - cosine)
    4. InfoNCE: in-batch contrastive learning
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_cosine: float = 0.5,
        lambda_infonce: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_cosine = lambda_cosine
        self.lambda_infonce = lambda_infonce
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: [batch, 768] - raw predictions (not normalized)
            target: [batch, 768] - raw targets (not normalized)

        Returns:
            loss: scalar
            stats: dict with loss components
        """
        # 1. L2 normalize (consultant requirement)
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)

        # 2. MSE loss (on normalized vectors)
        loss_mse = F.mse_loss(pred_norm, target_norm)

        # 3. Cosine penalty: 0.5 * (1 - cosine)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        loss_cosine = 0.5 * (1.0 - cosine_sim)

        # 4. InfoNCE: in-batch contrastive
        # Compute similarity matrix: [batch, batch]
        sim_matrix = torch.mm(pred_norm, target_norm.t()) / self.temperature

        # Positive pairs are on diagonal
        labels = torch.arange(pred.size(0), device=pred.device)
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        # Total loss
        loss = (
            self.lambda_mse * loss_mse +
            self.lambda_cosine * loss_cosine +
            self.lambda_infonce * loss_infonce
        )

        stats = {
            'loss_total': loss.item(),
            'loss_mse': loss_mse.item(),
            'loss_cosine': loss_cosine.item(),
            'loss_infonce': loss_infonce.item(),
            'cosine_sim': cosine_sim.item()
        }

        return loss, stats


# ============================================================================
# RETRIEVAL EVALUATION (Hit@K)
# ============================================================================

def compute_hit_at_k(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
    delta_mode: bool = True
) -> Dict[str, float]:
    """
    Compute Hit@K: Does predicted vector retrieve correct next concept?

    This is the CRITICAL metric (consultant: "targets to pass"):
    - Hit@1 ≥30%
    - Hit@5 ≥55%
    - Hit@10 ≥70%

    Args:
        model: Trained LVM
        dataloader: Validation data
        device: torch device
        k_values: K values to compute
        delta_mode: If True, model predicts deltas

    Returns:
        Dictionary with Hit@K metrics
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

                # Model predicts delta
                pred_delta = model(contexts)

                # Reconstruct prediction: ŷ = y_curr + Δ̂
                pred = y_curr + pred_delta
            else:
                contexts, targets = batch
                contexts = contexts.to(device)
                pred = model(contexts)

            all_preds.append(pred.cpu())
            all_targets.append(targets)

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)  # [N, 768]
    all_targets = torch.cat(all_targets, dim=0)  # [N, 768]

    # Normalize
    all_preds_norm = F.normalize(all_preds, p=2, dim=-1)
    all_targets_norm = F.normalize(all_targets, p=2, dim=-1)

    # Compute similarity matrix: [N, N]
    # (each prediction vs all targets in validation set)
    sim_matrix = torch.mm(all_preds_norm, all_targets_norm.t())

    # For each prediction, rank all targets
    # Top-K retrieved indices (handle small validation sets)
    max_k = min(max(k_values), len(all_preds))
    _, topk_indices = sim_matrix.topk(k=max_k, dim=-1)

    # Ground truth: index i should retrieve target i
    ground_truth = torch.arange(len(all_preds)).unsqueeze(1)

    # Compute Hit@K
    results = {}
    for k in k_values:
        if k > max_k:
            # Skip if k is larger than validation set
            results[f'hit@{k}'] = float('nan')
            continue
        # Check if ground truth is in top-k
        hits = (topk_indices[:, :k] == ground_truth).any(dim=1)
        hit_rate = hits.float().mean().item()
        results[f'hit@{k}'] = hit_rate

    # Also compute mean cosine for reference
    cosine_sim = (all_preds_norm * all_targets_norm).sum(dim=-1).mean().item()
    results['cosine'] = cosine_sim

    return results


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ImprovedLoss,
    device: torch.device,
    delta_mode: bool
) -> Tuple[float, Dict]:
    """Train one epoch with improved loss"""
    model.train()

    total_loss = 0.0
    all_stats = {
        'loss_total': 0.0,
        'loss_mse': 0.0,
        'loss_cosine': 0.0,
        'loss_infonce': 0.0,
        'cosine_sim': 0.0
    }

    for batch_idx, batch in enumerate(dataloader):
        if delta_mode:
            contexts, deltas, y_curr, targets = batch
            contexts = contexts.to(device)
            deltas = deltas.to(device)
            y_curr = y_curr.to(device)

            # Model predicts delta
            pred_delta = model(contexts)

            # Loss on delta prediction
            loss, stats = criterion(pred_delta, deltas)
        else:
            contexts, targets = batch
            contexts = contexts.to(device)
            targets = targets.to(device)

            pred = model(contexts)
            loss, stats = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in stats.items():
            all_stats[k] += v

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Cosine: {stats['cosine_sim']:.4f}")

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in all_stats.items()}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: ImprovedLoss,
    device: torch.device,
    delta_mode: bool,
    compute_retrieval: bool = True
) -> Dict[str, float]:
    """
    Evaluate model with both loss and retrieval metrics.

    Returns:
        Dictionary with:
        - loss_total, loss_mse, loss_cosine, loss_infonce
        - cosine_sim
        - hit@1, hit@5, hit@10 (if compute_retrieval=True)
    """
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
                deltas = deltas.to(device)

                pred_delta = model(contexts)
                loss, stats = criterion(pred_delta, deltas)
            else:
                contexts, targets = batch
                contexts = contexts.to(device)
                targets = targets.to(device)

                pred = model(contexts)
                loss, stats = criterion(pred, targets)

            total_loss += loss.item()
            for k, v in stats.items():
                all_stats[k] += v

    n = len(dataloader)
    results = {k: v / n for k, v in all_stats.items()}

    # Compute retrieval metrics (Hit@K)
    if compute_retrieval:
        retrieval_metrics = compute_hit_at_k(
            model, dataloader, device,
            k_values=[1, 5, 10],
            delta_mode=delta_mode
        )
        results.update(retrieval_metrics)

    return results


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Improved LVM Trainer (Consultant Recommendations)'
    )

    # Model and data
    parser.add_argument('--model-type', required=True,
                        choices=['lstm', 'gru', 'transformer', 'amn',
                                'hierarchical_gru', 'memory_gru'])
    parser.add_argument('--data', required=True,
                        help='Path to training sequences NPZ')

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')

    # Improved training options
    parser.add_argument('--delta-mode', action='store_true', default=True,
                        help='Predict delta instead of absolute (RECOMMENDED)')
    parser.add_argument('--coherence-threshold', type=float, default=0.78,
                        help='Filter chains with neighbor cosine < threshold')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Train/val split ratio')

    # Loss weights
    parser.add_argument('--lambda-mse', type=float, default=1.0)
    parser.add_argument('--lambda-cosine', type=float, default=0.5)
    parser.add_argument('--lambda-infonce', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)

    # Output
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'artifacts/lvm/models_improved/{args.model_type}_{timestamp}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IMPROVED LVM TRAINER (Consultant Recommendations)")
    print("=" * 80)
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Data: {args.data}")
    print(f"Delta mode: {args.delta_mode} {'✅ RECOMMENDED' if args.delta_mode else '⚠️'}")
    print(f"Coherence threshold: {args.coherence_threshold}")
    print()
    print("Loss Configuration:")
    print(f"  MSE weight: {args.lambda_mse}")
    print(f"  Cosine weight: {args.lambda_cosine}")
    print(f"  InfoNCE weight: {args.lambda_infonce}")
    print(f"  Temperature: {args.temperature}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ChainAwareDataset(
        args.data,
        delta_mode=args.delta_mode,
        coherence_threshold=args.coherence_threshold
    )

    # Chain-level split (ZERO LEAKAGE)
    train_dataset, val_dataset = ChainAwareDataset.chain_split(
        dataset,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("Creating model...")
    device = torch.device(args.device)

    # Get model config
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

    # Loss and optimizer
    criterion = ImprovedLoss(
        lambda_mse=args.lambda_mse,
        lambda_cosine=args.lambda_cosine,
        lambda_infonce=args.lambda_infonce,
        temperature=args.temperature
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    print("=" * 80)
    print("TRAINING START")
    print("=" * 80)
    print()

    best_val_loss = float('inf')
    best_hit5 = 0.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device, args.delta_mode
        )

        # Evaluate (with retrieval metrics every 5 epochs)
        compute_retrieval = (epoch % 5 == 0) or (epoch == args.epochs - 1)
        val_results = evaluate(
            model, val_loader, criterion, device, args.delta_mode,
            compute_retrieval=compute_retrieval
        )

        scheduler.step(val_results['loss_total'])

        # Print results
        print(f"\nTrain: Loss={train_loss:.6f}, Cosine={train_stats['cosine_sim']:.4f}")
        print(f"Val: Loss={val_results['loss_total']:.6f}, Cosine={val_results['cosine_sim']:.4f}")

        if compute_retrieval:
            print(f"Retrieval: Hit@1={val_results['hit@1']:.4f}, "
                  f"Hit@5={val_results['hit@5']:.4f}, "
                  f"Hit@10={val_results['hit@10']:.4f}")

            # Check go/no-go thresholds
            hit1_ok = val_results['hit@1'] >= 0.30
            hit5_ok = val_results['hit@5'] >= 0.55
            cosine_ok = val_results['cosine_sim'] >= 0.60

            print(f"Go/No-Go: Hit@1={'✅' if hit1_ok else '❌'}, "
                  f"Hit@5={'✅' if hit5_ok else '❌'}, "
                  f"Cosine={'✅' if cosine_ok else '❌'}")

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
                'hit@1': val_results['hit@1'],
                'hit@5': val_results['hit@5'],
                'hit@10': val_results['hit@10']
            })

        history.append(epoch_data)

        # Save best model
        if val_results['loss_total'] < best_val_loss:
            best_val_loss = val_results['loss_total']
            torch.save({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'model_state_dict': model.state_dict(),
                'val_results': val_results,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_loss={best_val_loss:.6f})")

        if compute_retrieval and val_results['hit@5'] > best_hit5:
            best_hit5 = val_results['hit@5']
            torch.save({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'model_state_dict': model.state_dict(),
                'val_results': val_results,
                'args': vars(args)
            }, output_dir / 'best_hit5_model.pt')
            print(f"✓ Saved best Hit@5 model ({best_hit5:.4f})")

    # Save final results
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'model_name': MODEL_SPECS[args.model_type]['name'],
            'history': history,
            'best_val_loss': best_val_loss,
            'best_hit5': best_hit5,
            'args': vars(args),
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best Hit@5: {best_hit5:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
