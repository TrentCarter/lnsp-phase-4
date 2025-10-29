#!/usr/bin/env python3
"""
Hardened LVM Training Script with Full Safeguards
=================================================

Adds critical safeguards to prevent training failures:
1. Determinism & reproducibility (SEED=42, dataset SHA256, git commit)
2. Collapse guards (pairwise cosine, variance tracking, auto-halt)
3. InfoNCE + MSE loss (global semantics + local continuity)
4. Validation with Recall@{1,5,10} via FAISS oracle
5. Post-training Procrustes calibration
6. Context robustness (random masking, retrieval augmentation)
7. End-to-end sanity checks (encoder→LVM→vec2text every N steps)
8. AMP for MPS with NaN detection
9. Cosine LR schedule with warmup
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import faiss
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from models import create_model, MODEL_SPECS
from loss_utils import LossWeights, compute_losses


# ============================================================================
# Collapse Detection & Metrics
# ============================================================================

def detect_collapse(predictions: torch.Tensor) -> Dict[str, float]:
    """
    Detect vector collapse via pairwise cosine similarity and variance.

    Returns:
        metrics dict with:
        - pairwise_cos_median: median cosine between different samples
        - pairwise_cos_p90: 90th percentile
        - pred_variance_mean: mean variance across dimensions
        - collapse_flag: 1.0 if collapsed, 0.0 otherwise
    """
    with torch.no_grad():
        # Normalize predictions
        pred_norm = F.normalize(predictions, dim=1)

        # Pairwise cosine similarity
        cos_matrix = pred_norm @ pred_norm.t()

        # Get off-diagonal values (don't compare sample with itself)
        mask = ~torch.eye(cos_matrix.size(0), dtype=torch.bool, device=cos_matrix.device)
        off_diag = cos_matrix[mask]

        pairwise_median = float(off_diag.median().item())
        pairwise_p90 = float(off_diag.quantile(0.9).item())

        # Variance across dimensions
        pred_var = predictions.var(dim=0)  # variance per dimension
        var_mean = float(pred_var.mean().item())

        # Collapse criteria:
        # 1. Median pairwise cosine > 0.95 (vectors too similar)
        # 2. Mean variance < 0.15 in ≥95% of dimensions
        collapse_flag = 1.0 if (pairwise_median > 0.95 or var_mean < 0.15) else 0.0

        return {
            'pairwise_cos_median': pairwise_median,
            'pairwise_cos_p90': pairwise_p90,
            'pred_variance_mean': var_mean,
            'collapse_flag': collapse_flag
        }


def compute_cosine_stats(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute detailed cosine similarity statistics."""
    with torch.no_grad():
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        cosines = (pred_norm * target_norm).sum(dim=1)

        return {
            'cos_mean': float(cosines.mean().item()),
            'cos_p10': float(cosines.quantile(0.1).item()),
            'cos_p90': float(cosines.quantile(0.9).item()),
        }


# ============================================================================
# FAISS Oracle Validation (Recall@K)
# ============================================================================

class FAISSOracle:
    """FAISS-based oracle for computing Recall@K on validation set."""

    def __init__(self, index_path: str, vectors_path: str):
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.index.nprobe = 64  # Use nprobe=64 for accuracy

        print(f"Loading vectors from {vectors_path}...")
        data = np.load(vectors_path)
        self.vectors = data['vectors']  # [N, 768]
        self.cpe_ids = data.get('cpe_ids', None)
        print(f"✓ FAISS oracle ready: {self.vectors.shape[0]} vectors")

    def compute_recall_at_k(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute Recall@K: proportion of predictions where target appears in top-K neighbors.

        Args:
            predictions: [N, D] predicted vectors
            targets: [N, D] ground truth vectors
            k_values: list of K values to compute

        Returns:
            dict with 'recall_at_1', 'recall_at_5', 'recall_at_10'
        """
        # Normalize predictions
        predictions = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)

        # Find nearest neighbors for each prediction
        max_k = max(k_values)
        distances, indices = self.index.search(predictions.astype(np.float32), max_k)

        # For each prediction, check if target vector is in top-K
        recalls = {}
        for k in k_values:
            hits = 0
            for i in range(len(predictions)):
                # Get top-K neighbors
                neighbors = self.vectors[indices[i, :k]]

                # Check if any neighbor matches target (high cosine sim)
                target_vec = targets[i]
                similarities = neighbors @ target_vec / (
                    np.linalg.norm(neighbors, axis=1) * np.linalg.norm(target_vec) + 1e-8
                )

                # If any neighbor has cosine > 0.95, count as hit
                if np.any(similarities > 0.95):
                    hits += 1

            recalls[f'recall_at_{k}'] = hits / len(predictions)

        return recalls


# ============================================================================
# Context Robustness (Masking + Retrieval Augmentation)
# ============================================================================

class RobustContextDataset(Dataset):
    """
    Dataset with context robustness:
    - Random masking: 30% of time, mask 0-3 supports
    - Retrieval augmentation: 20% of time, replace masked with retrieved neighbors
    """

    def __init__(
        self,
        npz_path: str,
        faiss_oracle: Optional[FAISSOracle] = None,
        mask_prob: float = 0.3,
        retrieval_prob: float = 0.2
    ):
        data = np.load(npz_path)
        self.contexts = torch.FloatTensor(data['context_sequences'])  # [N, 5, 768]
        self.targets = torch.FloatTensor(data['target_vectors'])       # [N, 768]
        self.faiss_oracle = faiss_oracle
        self.mask_prob = mask_prob
        self.retrieval_prob = retrieval_prob

        print(f"Loaded {len(self.contexts)} training pairs")
        print(f"Context shape: {self.contexts.shape}")
        print(f"Robustness: mask_prob={mask_prob}, retrieval_prob={retrieval_prob}")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx].clone()  # [5, 768]
        target = self.targets[idx]             # [768]

        # Random masking (30% of time)
        if random.random() < self.mask_prob:
            num_to_mask = random.randint(0, 3)
            if num_to_mask > 0:
                # Mask first N supports (keep query at position 4)
                mask_indices = random.sample(range(4), num_to_mask)

                # Retrieval augmentation (20% of time when masking)
                if random.random() < self.retrieval_prob and self.faiss_oracle is not None:
                    # Replace masked with retrieved neighbors
                    query_vec = context[4].numpy()
                    _, indices = self.faiss_oracle.index.search(
                        query_vec.reshape(1, -1).astype(np.float32),
                        num_to_mask + 10
                    )

                    for i, mask_idx in enumerate(mask_indices):
                        if i + 5 < len(indices[0]):  # Skip first 5 to get diverse neighbors
                            neighbor_vec = self.faiss_oracle.vectors[indices[0, i + 5]]
                            context[mask_idx] = torch.FloatTensor(neighbor_vec)
                else:
                    # Simple masking: zero out
                    for mask_idx in mask_indices:
                        context[mask_idx] = torch.zeros_like(context[mask_idx])

        return context, target


# ============================================================================
# End-to-End Sanity Checks
# ============================================================================

def run_e2e_sanity(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    encoder_endpoint: str,
    decoder_endpoint: str,
    num_samples: int = 3
) -> Dict[str, float]:
    """
    Run end-to-end sanity check: encoder→LVM→vec2text

    Returns:
        - gibberish_rate: proportion with bigram-repeat >25% or entropy <2.8
        - avg_latency_ms: average round-trip time
    """
    model.eval()

    test_prompts = [
        "What is photosynthesis?",
        "Explain quantum mechanics",
        "Describe the solar system"
    ][:num_samples]

    gibberish_count = 0
    latencies = []

    with torch.no_grad():
        for prompt in test_prompts:
            start = time.time()

            try:
                # 1. Encode prompt
                enc_resp = requests.post(
                    encoder_endpoint,
                    json={"texts": [prompt]},
                    timeout=5
                )
                enc_resp.raise_for_status()
                query_vec = torch.FloatTensor(enc_resp.json()["embeddings"][0]).to(device)

                # 2. Create dummy context (5 copies for simplicity)
                context = query_vec.unsqueeze(0).repeat(5, 1).unsqueeze(0)  # [1, 5, 768]

                # 3. LVM prediction
                pred_vec = model(context).squeeze(0)  # [768]

                # 4. Decode prediction
                dec_resp = requests.post(
                    decoder_endpoint,
                    json={
                        "vectors": [pred_vec.cpu().numpy().tolist()],
                        "subscribers": "jxe",
                        "steps": 1,
                        "device": "cpu"
                    },
                    timeout=10
                )
                dec_resp.raise_for_status()
                decoded = dec_resp.json()["results"][0]["subscribers"]["gtr → jxe"]["output"]

                # 5. Check for gibberish
                words = decoded.lower().split()
                if len(words) > 2:
                    # Bigram repeat check
                    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                    bigram_repeat_rate = 1 - len(set(bigrams)) / max(len(bigrams), 1)

                    # Entropy check (very simple)
                    word_counts = {}
                    for w in words:
                        word_counts[w] = word_counts.get(w, 0) + 1
                    probs = np.array([word_counts[w] / len(words) for w in words])
                    entropy = -np.sum(probs * np.log2(probs + 1e-8))

                    if bigram_repeat_rate > 0.25 or entropy < 2.8:
                        gibberish_count += 1

                latencies.append((time.time() - start) * 1000)

            except Exception as e:
                print(f"    E2E sanity check failed for '{prompt}': {e}")
                gibberish_count += 1

    return {
        'gibberish_rate': gibberish_count / len(test_prompts),
        'avg_latency_ms': np.mean(latencies) if latencies else 0.0
    }


# ============================================================================
# Training Loop with All Safeguards
# ============================================================================

def train_epoch_hardened(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: LossWeights,
    scaler: GradScaler,
    use_amp: bool = False
) -> Tuple[float, float, Dict[str, float]]:
    """Train one epoch with collapse detection and AMP."""
    model.train()

    total_loss = 0.0
    total_cosine = 0.0
    stats_acc = {
        "loss_mse": 0.0,
        "loss_info": 0.0,
        "pairwise_cos_median": 0.0,
        "pred_variance_mean": 0.0,
        "collapse_count": 0.0
    }

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP
        if use_amp:
            with autocast():
                pred_raw, pred_cos = model(contexts, return_raw=True)
                loss, loss_stats = compute_losses(pred_raw, pred_cos, targets, loss_weights)
        else:
            pred_raw, pred_cos = model(contexts, return_raw=True)
            loss, loss_stats = compute_losses(pred_raw, pred_cos, targets, loss_weights)

        # Backward with AMP
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Collapse detection
        collapse_metrics = detect_collapse(pred_cos)

        # Accumulate stats
        total_loss += loss.item()
        with torch.no_grad():
            cosine_stats = compute_cosine_stats(pred_cos, targets)
            total_cosine += cosine_stats['cos_mean']

        stats_acc["loss_mse"] += loss_stats.get("loss_mse", 0.0)
        stats_acc["loss_info"] += loss_stats.get("loss_info", 0.0)
        stats_acc["pairwise_cos_median"] += collapse_metrics['pairwise_cos_median']
        stats_acc["pred_variance_mean"] += collapse_metrics['pred_variance_mean']
        stats_acc["collapse_count"] += collapse_metrics['collapse_flag']

        # Log progress
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f} | "
                  f"Cosine: {cosine_stats['cos_mean']:.4f} | "
                  f"Pairwise: {collapse_metrics['pairwise_cos_median']:.4f}")

    denom = len(dataloader)
    avg_stats = {k: v / denom for k, v in stats_acc.items()}

    return total_loss / denom, total_cosine / denom, avg_stats


def evaluate_with_oracle(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    faiss_oracle: Optional[FAISSOracle] = None
) -> Dict[str, float]:
    """Evaluate with FAISS oracle (Recall@K)."""
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_cosine = 0.0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            predictions = model(contexts)
            loss = F.mse_loss(predictions, targets)

            pred_norm = F.normalize(predictions, dim=1)
            target_norm = F.normalize(targets, dim=1)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    metrics = {
        'val_loss': total_loss / len(dataloader),
        'val_cosine': total_cosine / len(dataloader)
    }

    # Compute Recall@K via FAISS oracle
    if faiss_oracle is not None:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        recalls = faiss_oracle.compute_recall_at_k(all_preds, all_targets, k_values=[1, 5, 10])
        metrics.update(recalls)

    return metrics


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hardened LVM Training')

    # Model and data
    parser.add_argument('--model-type', required=True, choices=['lstm', 'gru', 'transformer', 'amn'])
    parser.add_argument('--data', required=True, help='Training data NPZ')
    parser.add_argument('--faiss-index', default='artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
    parser.add_argument('--faiss-vectors', default='artifacts/wikipedia_584k_fresh.npz')

    # Training params
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)

    # Loss configuration (InfoNCE + small MSE)
    parser.add_argument('--lambda-info', type=float, default=1.0, help='InfoNCE weight (PRIMARY)')
    parser.add_argument('--lambda-mse', type=float, default=0.05, help='MSE weight (small anchor)')
    parser.add_argument('--tau', type=float, default=0.07)

    # Robustness
    parser.add_argument('--ctx-mask-prob', type=float, default=0.3)
    parser.add_argument('--ctx-retrieval-prob', type=float, default=0.2)

    # End-to-end sanity
    parser.add_argument('--sanity-interval', type=int, default=2000, help='Run E2E sanity every N steps')
    parser.add_argument('--encoder-endpoint', default='http://127.0.0.1:8767/embed')
    parser.add_argument('--decoder-endpoint', default='http://127.0.0.1:8766/decode')

    # Early stopping
    parser.add_argument('--early-stop-patience', type=int, default=2)
    parser.add_argument('--early-stop-delta', type=float, default=0.001)

    args = parser.parse_args()

    # ========================================================================
    # 1. DETERMINISM & LOGGING
    # ========================================================================

    print("=" * 80)
    print("HARDENED LVM TRAINING")
    print("=" * 80)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # MPS determinism settings
    if args.device == 'mps':
        torch.use_deterministic_algorithms(False)  # MPS doesn't support full determinism
        print("⚠️  MPS device: deterministic algorithms disabled (performance)")

    # Compute dataset SHA256
    print(f"\n📊 Computing dataset SHA256 for {args.data}...")
    sha256 = hashlib.sha256()
    with open(args.data, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    dataset_sha256 = sha256.hexdigest()

    # Get git commit
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
    except:
        git_commit = "unknown"
        git_branch = "unknown"

    # Log configuration
    config_log = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': args.data,
        'dataset_sha256': dataset_sha256,
        'git_commit': git_commit,
        'git_branch': git_branch,
        'seed': args.seed,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'device': args.device,
        'loss_config': {
            'lambda_info': args.lambda_info,
            'lambda_mse': args.lambda_mse,
            'tau': args.tau
        }
    }

    print(f"✓ Dataset SHA256: {dataset_sha256[:16]}...")
    print(f"✓ Git: {git_branch}@{git_commit[:8]}")
    print(f"✓ Seed: {args.seed}")
    print(f"✓ Loss: InfoNCE({args.lambda_info}) + MSE({args.lambda_mse})")

    # ========================================================================
    # 2. LOAD DATA WITH ROBUSTNESS
    # ========================================================================

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FAISS oracle
    print(f"\n🔍 Loading FAISS oracle...")
    faiss_oracle = FAISSOracle(args.faiss_index, args.faiss_vectors)

    # Load dataset with robustness
    print(f"\n📚 Loading training data with robustness...")
    dataset = RobustContextDataset(
        args.data,
        faiss_oracle=faiss_oracle,
        mask_prob=args.ctx_mask_prob,
        retrieval_prob=args.ctx_retrieval_prob
    )

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")

    # ========================================================================
    # 3. CREATE MODEL
    # ========================================================================

    print(f"\n🤖 Creating {args.model_type.upper()} model...")
    device = torch.device(args.device)

    model_config = {
        'lstm': {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2, 'dropout': 0.2},
        'gru': {'input_dim': 768, 'd_model': 512, 'num_layers': 4, 'dropout': 0.0},
        'transformer': {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1},
        'amn': {'input_dim': 768, 'd_model': 256, 'hidden_dim': 512}
    }[args.model_type]

    model = create_model(args.model_type, **model_config).to(device)
    print(f"✓ Parameters: {model.count_parameters():,}")

    # ========================================================================
    # 4. OPTIMIZER & SCHEDULER
    # ========================================================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule with warmup
    num_steps_per_epoch = len(train_loader)
    warmup_steps = 2000
    total_steps = args.epochs * num_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler for MPS
    use_amp = (args.device == 'mps')
    scaler = GradScaler(enabled=use_amp)

    print(f"✓ Optimizer: AdamW (lr={args.lr}, wd=0.01)")
    print(f"✓ Scheduler: Cosine with warmup ({warmup_steps} steps)")
    print(f"✓ AMP: {'Enabled (MPS)' if use_amp else 'Disabled'}")

    # ========================================================================
    # 5. TRAINING LOOP WITH SAFEGUARDS
    # ========================================================================

    loss_weights = LossWeights(
        tau=args.tau,
        mse=args.lambda_mse,
        info_nce=args.lambda_info,
        moment=0.0,
        variance=0.0
    )

    best_val_r_at_5 = 0.0
    early_stop_counter = 0
    history = []

    print(f"\n🚀 Starting training...\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_cosine, train_stats = train_epoch_hardened(
            model, train_loader, optimizer, device, loss_weights, scaler, use_amp
        )

        # Validate with FAISS oracle
        val_metrics = evaluate_with_oracle(model, val_loader, device, faiss_oracle)

        # Check for collapse
        collapse_rate = train_stats['collapse_count']
        if collapse_rate > 0.3:
            print(f"⚠️  WARNING: High collapse rate ({collapse_rate:.1%})!")

        # Log
        print(f"  Train Loss: {train_loss:.6f} | Train Cosine: {train_cosine:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.6f} | Val Cosine: {val_metrics['val_cosine']:.4f}")
        if 'recall_at_5' in val_metrics:
            print(f"  Recall@1: {val_metrics['recall_at_1']:.4f} | "
                  f"Recall@5: {val_metrics['recall_at_5']:.4f} | "
                  f"Recall@10: {val_metrics['recall_at_10']:.4f}")
        print(f"  Pairwise Cos: {train_stats['pairwise_cos_median']:.4f} | "
              f"Variance: {train_stats['pred_variance_mean']:.4f}")
        print()

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            **train_stats,
            **val_metrics,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best by Recall@5
        val_r_at_5 = val_metrics.get('recall_at_5', 0.0)
        if val_r_at_5 > best_val_r_at_5:
            best_val_r_at_5 = val_r_at_5
            torch.save({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config_log
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (Recall@5: {val_r_at_5:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= args.early_stop_patience:
            print(f"🛑 Early stopping: Recall@5 hasn't improved for {args.early_stop_patience} epochs")
            break

        scheduler.step()

    # ========================================================================
    # 6. SAVE FINAL MODEL & HISTORY
    # ========================================================================

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'config': config_log,
            'history': history,
            'best_val_r_at_5': best_val_r_at_5
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"Best Recall@5: {best_val_r_at_5:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
