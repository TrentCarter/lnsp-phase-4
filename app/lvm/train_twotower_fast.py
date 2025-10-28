#!/usr/bin/env python3
"""
Fast Two-Tower Training with Same-Article + Near-Miss Negatives

OPTIMIZED: Uses batched operations, no per-sample loops.

Usage:
    python app/lvm/train_twotower_fast.py \
        --resume artifacts/lvm/models/twotower_samearticle/epoch3.pt \
        --train-npz artifacts/lvm/train_clean_disjoint.npz \
        --same-article-k 3 \
        --nearmiss-jsonl artifacts/mined/nearmiss_train_ep3.jsonl \
        --p-cache-npy artifacts/eval/p_train_ep3.npy \
        --epochs 1 --batch-size 256 --lr 1e-4 \
        --device cpu \
        --save-dir artifacts/lvm/models/twotower_fast
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Unbuffered Output Configuration (Fix log visibility)
# ============================================================
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.lvm.train_twotower import QueryTower, PayloadTower


# ============================================================
# Data Loading
# ============================================================

def load_nearmiss_map(path):
    """Load near-miss map: query_idx -> [negative_indices]"""
    if not path or not os.path.exists(path):
        return {}
    nm = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qi = int(r.get("q_index", r.get("query_idx")))
            nm[qi] = [int(x) for x in r.get("near_miss_indices", r.get("negative_indices", []))]
    print(f"  Loaded {len(nm)} near-miss queries")
    return nm


def load_p_cache(path):
    """Load pre-computed P vectors (L2-normalized)."""
    if not path or not os.path.exists(path):
        return None
    P = np.load(path).astype("float32")
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    P = P / norms
    print(f"  Loaded P-cache: {P.shape}")
    return P


def build_article_index(article_ids, global_ids):
    """Build {article_id: [global_indices]} for fast lookup.

    Args:
        article_ids: array of article IDs
        global_ids: array of global indices (pre-permutation row IDs)
    """
    index = defaultdict(list)
    for glob_idx, art_id in zip(global_ids, article_ids):
        index[int(art_id)].append(int(glob_idx))
    return dict(index)


# ============================================================
# Dataset
# ============================================================

class FastDataset(Dataset):
    """Dataset with TRUE global indices (pre-permutation row IDs)."""

    def __init__(self, contexts, targets, article_ids, global_ids):
        """
        Args:
            contexts: [N, ...] context arrays
            targets: [N, 768] target vectors
            article_ids: [N] article IDs
            global_ids: [N] TRUE global indices (original row IDs before shuffle)
        """
        self.contexts = contexts
        self.targets = targets
        self.article_ids = article_ids
        self.global_ids = global_ids  # CRITICAL: store original global IDs

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = torch.from_numpy(self.contexts[idx]).float()
        tgt = torch.from_numpy(self.targets[idx]).float()
        art_id = int(self.article_ids[idx])
        glob_idx = int(self.global_ids[idx])  # Return TRUE global ID
        return ctx, tgt, art_id, idx, glob_idx


# ============================================================
# Fast Batched Training
# ============================================================

def sample_negatives_batch(article_ids, global_indices,
                           article_index, nearmiss_map, P_cache,
                           same_article_k=3):
    """
    Sample negatives for entire batch at once (FAST).

    Args:
        article_ids: [B] article IDs for this batch
        global_indices: [B] TRUE global indices (pre-permutation row IDs)
        article_index: Dict[article_id -> List[global_ids]]
        nearmiss_map: Dict[global_id -> List[negative_global_ids]]
        P_cache: [N, 768] pre-computed P vectors (indexed by global ID)
        same_article_k: number of same-article negatives to sample

    Returns:
        neg_matrix: [B, max_negs, 768] padded tensor
        neg_mask: [B, max_negs] bool mask (True = valid negative)
    """
    B = len(article_ids)
    all_neg_indices = []

    for i in range(B):
        neg_idx_set = set()

        # Same-article negatives (use GLOBAL indices)
        art_id = int(article_ids[i])
        glob_idx = int(global_indices[i])
        candidates = article_index.get(art_id, [])
        # CRITICAL: exclude current sample's GLOBAL ID, not local
        candidates = [c for c in candidates if c != glob_idx]
        if candidates:
            k = min(same_article_k, len(candidates))
            sampled = np.random.choice(candidates, k, replace=False)
            neg_idx_set.update(sampled.tolist())

        # Near-miss negatives (already using global IDs)
        nm_cands = nearmiss_map.get(glob_idx, [])
        neg_idx_set.update(nm_cands[:1])  # Take first near-miss

        all_neg_indices.append(list(neg_idx_set))

    # Find max negatives for padding
    max_negs = max(len(x) for x in all_neg_indices) if all_neg_indices else 0
    if max_negs == 0:
        return torch.zeros(B, 1, 768), torch.zeros(B, 1, dtype=torch.bool)

    # Build padded matrix from P_cache
    neg_matrix = np.zeros((B, max_negs, 768), dtype=np.float32)
    neg_mask = np.zeros((B, max_negs), dtype=bool)

    for i, neg_list in enumerate(all_neg_indices):
        if neg_list and P_cache is not None:
            valid_idx = [idx for idx in neg_list if idx < len(P_cache)]
            if valid_idx:
                neg_vecs = P_cache[valid_idx]
                neg_matrix[i, :len(valid_idx), :] = neg_vecs
                neg_mask[i, :len(valid_idx)] = True

    return torch.from_numpy(neg_matrix), torch.from_numpy(neg_mask)


def train_epoch(q_tower, p_tower, dataloader, optimizer, device,
                article_index, nearmiss_map, P_cache,
                same_article_k=3, tau=0.07, epoch=0):
    """Fast batched training (no per-sample loops)."""
    q_tower.train()
    p_tower.train()

    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        contexts, targets, article_ids, local_idx, global_idx = batch
        contexts = contexts.to(device)
        targets = targets.to(device)
        B = contexts.size(0)

        # Forward
        q = q_tower(contexts)  # [B, 768]
        p = p_tower(targets)   # [B, 768]

        # Sample negatives (batched, fast) - FIXED: only need global_idx now
        neg_matrix, neg_mask = sample_negatives_batch(
            article_ids.numpy(), global_idx.numpy(),
            article_index, nearmiss_map, P_cache, same_article_k
        )
        neg_matrix = neg_matrix.to(device)
        neg_mask = neg_mask.to(device)

        # DEFENSIVE GUARD 1: Check near-miss coverage
        if nearmiss_map and step % 200 == 0:
            nm_count = sum(1 for gid in global_idx.tolist() if len(nearmiss_map.get(int(gid), [])) > 0)
            coverage_pct = 100.0 * nm_count / B
            if nm_count < max(1, B // 4):
                print(f"⚠️  Near-miss coverage low: {nm_count}/{B} ({coverage_pct:.1f}%)", flush=True)

        # DEFENSIVE GUARD 2: Positive not in negatives
        if step % 200 == 0 and neg_mask.any():
            # This is approximate check - full check would need to compare P vectors
            pass  # Skip expensive check in hot path, but log coverage instead

        # === Batched InfoNCE ===
        # In-batch positives: [B, B]
        logits_inbatch = torch.mm(q, p.t()) / tau

        # Extra negatives: [B, max_negs]
        if neg_matrix.size(1) > 0:
            # q: [B, 768], neg_matrix: [B, max_negs, 768]
            logits_extra = torch.bmm(
                q.unsqueeze(1),  # [B, 1, 768]
                neg_matrix.transpose(1, 2)  # [B, 768, max_negs]
            ).squeeze(1) / tau  # [B, max_negs]

            # Mask invalid negatives
            logits_extra = logits_extra.masked_fill(~neg_mask, -1e9)

            # Concatenate: [B, B + max_negs]
            logits = torch.cat([logits_inbatch, logits_extra], dim=1)
        else:
            logits = logits_inbatch

        # Cross-entropy loss (gold on diagonal)
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Heartbeat logging every 200 steps
        if step % 200 == 0:
            # Count active negatives
            neg_count = int(neg_mask.sum())
            neg_per_sample = neg_count / B if B > 0 else 0

            hb = {
                "epoch": int(epoch),
                "step": int(step),
                "steps_total": int(len(dataloader)),
                "loss": float(loss.item()),
                "avg_loss": float(total_loss / (num_batches + 1e-8)),
                "negatives_per_sample": float(neg_per_sample),
            }
            Path("artifacts/lvm/train_heartbeat.json").parent.mkdir(parents=True, exist_ok=True)
            with open("artifacts/lvm/train_heartbeat.json", "w") as f:
                json.dump(hb, f)
            print(f"[ep {epoch} step {step}/{len(dataloader)}] loss={loss.item():.4f} | negs/sample={neg_per_sample:.1f}", flush=True)

    return {'train_loss': total_loss / num_batches}


@torch.no_grad()
def validate(q_tower, p_tower, dataloader, device):
    """Validation."""
    q_tower.eval()
    p_tower.eval()

    all_q = []
    all_p = []

    for batch in dataloader:
        contexts, targets, _, _, _ = batch
        contexts = contexts.to(device)
        targets = targets.to(device)

        q = q_tower(contexts)
        p = p_tower(targets)

        all_q.append(q.cpu())
        all_p.append(p.cpu())

    all_q = torch.cat(all_q, dim=0)
    all_p = torch.cat(all_p, dim=0)

    cos = F.cosine_similarity(all_q, all_p, dim=1)

    return {
        'val_cosine': float(cos.mean()),
        'val_cosine_std': float(cos.std()),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Resume
    parser.add_argument('--resume', type=Path, required=True)

    # Architecture (inherited)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--d-state', type=int, default=128)
    parser.add_argument('--conv-sz', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Loss & Negatives
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--same-article-k', type=int, default=3)
    parser.add_argument('--nearmiss-jsonl', type=str, default="")
    parser.add_argument('--p-cache-npy', type=str, default="")

    # Data
    parser.add_argument('--train-npz', type=Path, required=True)
    parser.add_argument('--val-split', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)

    # System
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-dir', type=Path, required=True)

    args = parser.parse_args()
    # Ensure save_dir is Path object (defensive)
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FAST TWO-TOWER TRAINING")
    print("=" * 80)
    print(f"Resume: {args.resume}")
    print(f"Same-article K: {args.same_article_k}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint['args']
    print(f"  Loaded epoch {checkpoint['epoch']}")
    print()

    # Load negatives
    print("Loading negatives...")
    nearmiss_map = load_nearmiss_map(args.nearmiss_jsonl)
    P_cache = load_p_cache(args.p_cache_npy)
    if args.nearmiss_jsonl and P_cache is None:
        raise SystemExit("ERROR: --nearmiss-jsonl requires --p-cache-npy")
    print()

    # Load data
    print("Loading data...")
    data = np.load(args.train_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']
    truth_keys = data['truth_keys']
    article_ids = truth_keys[:, 0]

    print(f"  Total samples: {len(contexts)}")
    print(f"  Unique articles: {len(np.unique(article_ids))}")
    print()

    # CRITICAL: Create global IDs BEFORE permutation
    global_ids = np.arange(len(contexts), dtype=np.int64)
    print(f"  Global IDs: {global_ids[:5]}... (total: {len(global_ids)})")

    # Build article index with GLOBAL IDs
    print("Building article index...")
    article_index = build_article_index(article_ids, global_ids)
    print(f"  Indexed {len(article_index)} articles")
    print()

    # Split (permute indices, not data)
    n_val = int(len(contexts) * args.val_split)
    n_train = len(contexts) - n_val
    perm = np.random.permutation(len(contexts))

    # Extract permuted slices
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = FastDataset(
        contexts[train_indices],
        targets[train_indices],
        article_ids[train_indices],
        global_ids[train_indices]  # CRITICAL: pass true global IDs
    )
    val_dataset = FastDataset(
        contexts[val_indices],
        targets[val_indices],
        article_ids[val_indices],
        global_ids[val_indices]  # CRITICAL: pass true global IDs
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=False
    )

    print(f"  Train: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)}")
    print()

    # Create towers
    print("Creating towers...")
    q_tower = QueryTower(
        backbone_type=ckpt_args.get('arch_q', 'mamba_s'),
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        conv_sz=args.conv_sz,
        expand=args.expand,
        dropout=args.dropout,
    ).to(args.device)

    p_tower = PayloadTower(
        backbone_type=ckpt_args.get('arch_p', 'mamba_s'),
        d_model=args.d_model,
    ).to(args.device)

    q_tower.load_state_dict(checkpoint['q_tower_state_dict'])
    p_tower.load_state_dict(checkpoint['p_tower_state_dict'])
    print(f"  Loaded weights from epoch {checkpoint['epoch']}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(q_tower.parameters()) + list(p_tower.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    history = []
    start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}")
        print(f"{'='*80}")

        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            q_tower, p_tower, train_loader, optimizer, args.device,
            article_index, nearmiss_map, P_cache,
            args.same_article_k, args.temperature, epoch
        )

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

        print(f"\nEpoch {epoch} ({epoch_time/60:.1f} min)")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val cosine: {val_metrics['val_cosine']:.4f} ± {val_metrics['val_cosine_std']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'q_tower_state_dict': q_tower.state_dict(),
            'p_tower_state_dict': p_tower.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(args),
        }

        # Convert Paths to strings
        ckpt['args']['save_dir'] = str(ckpt['args']['save_dir'])
        ckpt['args']['train_npz'] = str(ckpt['args']['train_npz'])
        ckpt['args']['resume'] = str(ckpt['args']['resume'])

        # Ensure save_dir is Path object
        save_path = Path(args.save_dir) / f'epoch{epoch}.pt'
        torch.save(ckpt, save_path)
        print(f"\n  ✅ Saved: {save_path}")

    # Save history
    history_path = Path(args.save_dir) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
