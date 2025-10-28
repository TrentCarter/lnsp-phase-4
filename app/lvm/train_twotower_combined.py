#!/usr/bin/env python3
"""
Two-Tower Retriever with Same-Article + Near-Miss Negatives

Combines:
1. In-batch negatives (standard InfoNCE)
2. Same-article negatives (K=3 from same article)
3. Near-miss negatives (K=1 from mined top-5 FAISS results)

Usage:
    python app/lvm/train_twotower_combined.py \
        --resume artifacts/lvm/models/twotower_samearticle/epoch3.pt \
        --train-npz artifacts/lvm/train_clean_disjoint.npz \
        --same-article-k 3 \
        --nearmiss-jsonl artifacts/mined/nearmiss_train_ep3.jsonl \
        --nearmiss-per-query 1 \
        --p-cache-npy artifacts/eval/p_train_ep3.npy \
        --epochs 1 --eval-every 1 \
        --batch-size 64 --lr 1e-4 \
        --device cpu \
        --save-dir artifacts/lvm/models/twotower_combined
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

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.lvm.train_twotower import QueryTower, PayloadTower


# ============================================================
# Helper Functions for Near-Miss Support
# ============================================================

def load_nearmiss_map(path):
    """Load near-miss negatives map from JSONL."""
    if not path or not os.path.exists(path):
        return {}
    nm = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Support both key names from different miners
            qi = int(r.get("q_index", r.get("query_idx")))
            nm[qi] = [int(x) for x in r.get("near_miss_indices", r.get("negative_indices", []))]
    print(f"  Loaded {len(nm)} near-miss queries")
    return nm


def load_p_cache(path):
    """Load pre-computed P-tower vectors (L2-normalized)."""
    if not path or not os.path.exists(path):
        return None
    P = np.load(path).astype("float32")
    # Ensure L2-normalized
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    P = P / norms
    print(f"  Loaded P-cache: {P.shape}")
    return P


def sample_same_article_negatives_for_batch(article_cache, gold_idx_batch, article_id_batch, k):
    """Sample same-article negatives for a batch."""
    neg_lists = []
    for gold_idx, aid in zip(gold_idx_batch, article_id_batch):
        pool = article_cache.get(int(aid), [])
        pool = [pi for pi in pool if pi != int(gold_idx)]
        if not pool:
            neg_lists.append([])
            continue
        if len(pool) <= k:
            neg_lists.append(pool)
        else:
            neg_lists.append(np.random.choice(pool, size=k, replace=False).tolist())
    return neg_lists


def sample_nearmiss_negatives_for_batch(nearmiss_map, q_index_batch, per_query):
    """Sample near-miss negatives for a batch."""
    neg_lists = []
    for qi in q_index_batch:
        cand = nearmiss_map.get(int(qi), [])
        if not cand:
            neg_lists.append([])
            continue
        if len(cand) <= per_query:
            neg_lists.append(cand)
        else:
            neg_lists.append(np.random.choice(cand, size=per_query, replace=False).tolist())
    return neg_lists


def info_nce_per_sample(q_vec, p_pos, neg_mats, tau=0.07):
    """
    Per-sample InfoNCE with ragged negatives.

    Args:
        q_vec: [B, 768] L2-normalized query vectors
        p_pos: [B, 768] L2-normalized positive payload vectors
        neg_mats: list of B tensors, each [Ni, 768] (may be empty)
        tau: temperature

    Returns:
        loss: scalar tensor
    """
    B = q_vec.size(0)
    losses = []

    for i in range(B):
        qi = q_vec[i:i+1]  # [1, 768]
        pos = p_pos[i:i+1]  # [1, 768]
        pos_logit = (qi * pos).sum(dim=-1) / tau  # [1]

        if neg_mats[i].size(0) == 0:
            # No extra negatives, only positive
            denom_logits = pos_logit  # [1]
        else:
            neg = neg_mats[i]  # [Ni, 768]
            neg_logits = (qi @ neg.t()).squeeze(0) / tau  # [Ni]
            denom_logits = torch.cat([pos_logit, neg_logits], dim=0)  # [1+Ni]

        log_prob = pos_logit - torch.logsumexp(denom_logits, dim=0)
        losses.append(-log_prob)

    return torch.stack(losses).mean()


# ============================================================
# Dataset with Query Index
# ============================================================

class TwoTowerDatasetWithIndex(Dataset):
    """Dataset with article IDs and query indices for combined negatives."""

    def __init__(self, contexts, targets, article_ids, start_idx=0):
        self.contexts = contexts
        self.targets = targets
        self.article_ids = article_ids
        self.start_idx = start_idx  # Global index offset

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = torch.from_numpy(self.contexts[idx]).float()
        tgt = torch.from_numpy(self.targets[idx]).float()
        art_id = int(self.article_ids[idx])
        q_idx = self.start_idx + idx
        return ctx, tgt, art_id, idx, q_idx  # local idx and global q_idx


# ============================================================
# Training Function with Combined Negatives
# ============================================================

def train_epoch(q_tower, p_tower, dataloader, optimizer, device,
                article_index, nearmiss_map, P_cache,
                same_article_k=3, nearmiss_per_query=1, tau=0.07):
    """Train one epoch with in-batch + same-article + near-miss negatives."""
    q_tower.train()
    p_tower.train()

    total_loss_inbatch = 0.0
    total_loss_extra = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        contexts, targets, article_ids, local_indices, q_indices = batch
        contexts = contexts.to(device)
        targets = targets.to(device)

        B = contexts.size(0)

        # Forward through towers
        q = q_tower(contexts)  # [B, 768]
        p = p_tower(targets)   # [B, 768]

        # === Part 1: In-batch InfoNCE ===
        logits_inbatch = torch.mm(q, p.t()) / tau  # [B, B]
        labels = torch.arange(B, device=device)
        loss_inbatch = F.cross_entropy(logits_inbatch, labels)

        # === Part 2: Per-sample InfoNCE with extra negatives ===
        # Sample same-article negatives
        sa_lists = sample_same_article_negatives_for_batch(
            article_index, local_indices.numpy(), article_ids.numpy(), same_article_k
        )

        # Sample near-miss negatives
        nm_lists = sample_nearmiss_negatives_for_batch(
            nearmiss_map, q_indices.numpy(), nearmiss_per_query
        )

        # Materialize extra negatives from P_cache (stop-grad)
        extra_negs = []
        if P_cache is None:
            extra_negs = [torch.empty(0, q.size(1), device=device) for _ in range(B)]
        else:
            for sa_idx, nm_idx in zip(sa_lists, nm_lists):
                idxs = np.unique(np.array(sa_idx + nm_idx, dtype=np.int64))
                if idxs.size == 0:
                    extra_negs.append(torch.empty(0, q.size(1), device=device))
                else:
                    neg_np = P_cache[idxs]  # [N, 768] float32
                    neg_t = torch.from_numpy(neg_np).to(device)
                    extra_negs.append(F.normalize(neg_t, dim=-1))

        loss_extra = info_nce_per_sample(q, p, extra_negs, tau=tau)

        # === Combined Loss ===
        loss = 0.5 * loss_inbatch + 0.5 * loss_extra

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_inbatch += loss_inbatch.item()
        total_loss_extra += loss_extra.item()
        total_loss += loss.item()
        num_batches += 1

    return {
        'train_loss': total_loss / num_batches,
        'train_loss_inbatch': total_loss_inbatch / num_batches,
        'train_loss_extra': total_loss_extra / num_batches,
    }


@torch.no_grad()
def validate(q_tower, p_tower, dataloader, device):
    """Validate on held-out data."""
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
# Utility Functions
# ============================================================

def build_article_index(article_ids):
    """Build {article_id: [indices]} for fast same-article sampling."""
    index = defaultdict(list)
    for idx, art_id in enumerate(article_ids):
        index[int(art_id)].append(idx)
    return dict(index)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Resume
    parser.add_argument('--resume', type=Path, required=True,
                        help='Checkpoint to resume from')

    # Architecture (inherited from checkpoint)
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-layers', type=int, default=8)
    parser.add_argument('--d-state', type=int, default=128)
    parser.add_argument('--conv-sz', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Loss & Negatives
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--same-article-k', type=int, default=3,
                        help='Number of same-article negatives per query')
    parser.add_argument('--nearmiss-jsonl', type=str, default="",
                        help='JSONL with {q_index: int, near_miss_indices: [int, ...]}')
    parser.add_argument('--nearmiss-per-query', type=int, default=1,
                        help='Number of near-miss negatives per query')
    parser.add_argument('--p-cache-npy', type=str, default="",
                        help='Precomputed P tower outputs (float32 [N, 768], L2-normed)')

    # Data
    parser.add_argument('--train-npz', type=Path, required=True)
    parser.add_argument('--val-split', type=float, default=0.2)

    # Training
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)

    # System
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--eval-every', type=int, default=1)

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TWO-TOWER TRAINING (Combined Negatives)")
    print("=" * 80)
    print(f"Resume from: {args.resume}")
    print(f"Same-article K: {args.same_article_k}")
    print(f"Near-miss per query: {args.nearmiss_per_query}")
    print(f"Loss: InfoNCE (τ={args.temperature})")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint['args']
    print(f"  Loaded epoch {checkpoint['epoch']}")
    print()

    # Load near-miss map and P-cache
    print("Loading negatives...")
    nearmiss_map = load_nearmiss_map(args.nearmiss_jsonl)
    P_cache = load_p_cache(args.p_cache_npy)

    if args.nearmiss_jsonl and P_cache is None:
        raise SystemExit("[ABORT] --nearmiss-jsonl provided but --p-cache-npy missing or invalid")
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

    # Build article index
    print("Building article index...")
    article_index = build_article_index(article_ids)
    print(f"  Indexed {len(article_index)} articles")
    print()

    # Split
    n_val = int(len(contexts) * args.val_split)
    n_train = len(contexts) - n_val
    indices = np.random.permutation(len(contexts))

    train_dataset = TwoTowerDatasetWithIndex(
        contexts[indices[:n_train]],
        targets[indices[:n_train]],
        article_ids[indices[:n_train]],
        start_idx=0
    )
    val_dataset = TwoTowerDatasetWithIndex(
        contexts[indices[n_train:]],
        targets[indices[n_train:]],
        article_ids[indices[n_train:]],
        start_idx=n_train
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

    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print()

    # Create towers (inherit from checkpoint)
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

    # Load weights
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
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            q_tower, p_tower, train_loader, optimizer, args.device,
            article_index, nearmiss_map, P_cache,
            args.same_article_k, args.nearmiss_per_query, args.temperature
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

        print(f"Epoch {epoch} ({epoch_time:.1f}s)")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        print(f"    - In-batch: {train_metrics['train_loss_inbatch']:.4f}")
        print(f"    - Extra (SA+NM): {train_metrics['train_loss_extra']:.4f}")
        print(f"  Val cosine: {val_metrics['val_cosine']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print()

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

        torch.save(ckpt, Path(args.save_dir) / f'epoch{epoch}.pt')
        print(f"  ✅ Saved epoch{epoch}.pt")
        print()

    # Save history
    with open(Path(args.save_dir) / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Checkpoints: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
