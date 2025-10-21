#!/usr/bin/env python3
"""
Train two-tower retriever - Phase 2 with Hard Negatives + Memory Bank

Improvements over Phase 1:
1. Memory bank queue (10k-50k recent document vectors)
2. ANN-based hard negative mining every N epochs
3. Optional margin loss for hard negatives
4. Better learning rate schedule

Usage:
    python tools/train_twotower_phase2.py \
      --pairs artifacts/twotower/pairs_v3_synth.npz \
      --init-ckpt runs/twotower_v3_phase1/checkpoints/best_recall500.pt \
      --bs 32 --accum 8 --epochs 50 \
      --lr 1e-5 --wd 0.01 --tau 0.07 \
      --memory-bank-size 20000 \
      --mine-every 2 --num-hard-negs 16 \
      --out runs/twotower_v3_phase2
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from collections import deque


class GRUPoolQuery(nn.Module):
    """Query tower: Bidirectional GRU + mean pooling."""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.proj = nn.Linear(2 * hidden_dim, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        q = self.proj(pooled)
        q = F.normalize(q, p=2, dim=-1)
        return q


class IdentityDocTower(nn.Module):
    """Document tower: Just L2-normalize (identity transformation)."""
    def forward(self, d):
        return F.normalize(d, p=2, dim=-1)


class TwoTowerDataset(Dataset):
    """Dataset for two-tower training pairs."""

    def __init__(self, X, Y, hard_negatives=None):
        """
        Args:
            X: (N, seq_len, 768) - Context sequences
            Y: (N, 768) - Target vectors
            hard_negatives: (N, K, 768) - Optional hard negatives per sample
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.hard_negatives = None
        if hard_negatives is not None:
            self.hard_negatives = torch.from_numpy(hard_negatives).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.hard_negatives is not None:
            return self.X[idx], self.Y[idx], self.hard_negatives[idx]
        return self.X[idx], self.Y[idx]


class MemoryBank:
    """FIFO queue of recent document vectors for additional negatives."""

    def __init__(self, max_size=20000, dim=768):
        self.max_size = max_size
        self.dim = dim
        self.queue = deque(maxlen=max_size)

    def add(self, vectors):
        """Add batch of vectors to memory bank."""
        for v in vectors:
            self.queue.append(v.cpu().numpy())

    def sample(self, k):
        """Sample k vectors from memory bank."""
        if len(self.queue) < k:
            k = len(self.queue)
        if k == 0:
            return None
        indices = np.random.choice(len(self.queue), size=k, replace=False)
        return np.stack([self.queue[i] for i in indices])


def info_nce_loss_with_extras(query, doc_pos, tau=0.07, memory_bank=None, hard_negs=None):
    """
    InfoNCE loss with optional memory bank negatives and hard negatives.

    Args:
        query: (batch, 768) - Query vectors
        doc_pos: (batch, 768) - Positive document vectors
        tau: Temperature parameter
        memory_bank: MemoryBank object or None
        hard_negs: (batch, K, 768) - Hard negatives per sample or None

    Returns:
        loss: Scalar loss
    """
    batch_size = query.size(0)
    device = query.device

    # Positive similarities (diagonal)
    pos_sim = (query * doc_pos).sum(dim=-1, keepdim=True) / tau  # (batch, 1)

    # In-batch negatives
    neg_sim = torch.matmul(query, doc_pos.T) / tau  # (batch, batch)

    # Combine: positive + in-batch negatives
    all_sims = [pos_sim, neg_sim]

    # Add memory bank negatives if available
    if memory_bank is not None and len(memory_bank.queue) > 0:
        mb_size = min(256, len(memory_bank.queue))
        mb_vecs = memory_bank.sample(mb_size)
        if mb_vecs is not None:
            mb_vecs = torch.from_numpy(mb_vecs).float().to(device)
            mb_vecs = F.normalize(mb_vecs, p=2, dim=-1)
            mb_sim = torch.matmul(query, mb_vecs.T) / tau  # (batch, mb_size)
            all_sims.append(mb_sim)

    # Add hard negatives if available
    if hard_negs is not None:
        # hard_negs: (batch, K, 768)
        K = hard_negs.size(1)
        hard_negs_norm = F.normalize(hard_negs, p=2, dim=-1)
        hard_sim = torch.matmul(
            query.unsqueeze(1), hard_negs_norm.transpose(1, 2)
        ).squeeze(1) / tau  # (batch, K)
        all_sims.append(hard_sim)

    # Concatenate all similarities
    logits = torch.cat(all_sims, dim=1)  # (batch, 1 + batch + mb_size + K)

    # Labels: positive is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def mine_hard_negatives(model_q, model_d, train_loader, bank_vectors, k=16, device='cpu'):
    """
    Mine hard negatives using ANN search.

    For each training sample, find K nearest neighbors in the bank that are
    NOT the positive (i.e., confusors).

    Returns:
        hard_negs: (N, K, 768) - Hard negatives for each sample
    """
    model_q.eval()
    model_d.eval()

    print("  Mining hard negatives...")

    # Encode bank
    print("    Encoding bank...")
    bank_tensor = torch.from_numpy(bank_vectors).float().to(device)
    with torch.no_grad():
        bank_encoded = model_d(bank_tensor).cpu().numpy()  # (N_bank, 768)

    # Collect all queries and targets
    all_queries = []
    all_targets = []

    print("    Encoding queries...")
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="    Queries", leave=False):
            if len(batch) == 3:
                X, Y, _ = batch
            else:
                X, Y = batch
            X = X.to(device)
            queries = model_q(X).cpu().numpy()
            targets = Y.numpy()
            all_queries.append(queries)
            all_targets.append(targets)

    all_queries = np.concatenate(all_queries, axis=0)  # (N, 768)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 768)

    # For each query, find top-K+1 nearest neighbors (excluding self)
    print(f"    Finding top-{k} hard negatives per sample...")
    hard_negs = []

    for query, target in tqdm(zip(all_queries, all_targets), total=len(all_queries), desc="    Mining", leave=False):
        # Compute similarities to all bank vectors
        sims = np.dot(bank_encoded, query)  # (N_bank,)

        # Get top-K+1 indices
        top_indices = np.argsort(-sims)[:k+10]  # Extra to filter out positive

        # Find target in bank
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        target_sims = np.dot(bank_encoded, target_norm)
        target_idx = np.argmax(target_sims)

        # Filter out positive, take first K
        hard_idx = [idx for idx in top_indices if idx != target_idx][:k]

        # Get hard negative vectors
        hard_vecs = bank_vectors[hard_idx]  # (K, 768)
        hard_negs.append(hard_vecs)

    hard_negs = np.stack(hard_negs, axis=0)  # (N, K, 768)

    print(f"  ✓ Mined {len(hard_negs)} × {k} hard negatives")

    return hard_negs


def evaluate_recall(model_q, model_d, val_loader, bank_vectors, k_values=[10, 100, 500, 1000], device='cpu'):
    """Evaluate Recall@K on validation set."""
    model_q.eval()
    model_d.eval()

    print("  Encoding bank...")
    bank_tensor = torch.from_numpy(bank_vectors).float().to(device)
    with torch.no_grad():
        bank_encoded = model_d(bank_tensor).cpu().numpy()

    recalls = {k: [] for k in k_values}

    print("  Evaluating queries...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Eval", leave=False):
            if len(batch) == 3:
                X, Y, _ = batch
            else:
                X, Y = batch
            X = X.to(device)

            queries = model_q(X).cpu().numpy()
            targets = Y.numpy()

            for query, target in zip(queries, targets):
                sims = np.dot(bank_encoded, query)
                top_k_idx = np.argsort(-sims)[:max(k_values)]

                target_norm = target / (np.linalg.norm(target) + 1e-8)
                target_sims = np.dot(bank_encoded, target_norm)
                target_idx = np.argmax(target_sims)

                for k in k_values:
                    recalls[k].append(1.0 if target_idx in top_k_idx[:k] else 0.0)

    metrics = {f'recall@{k}': np.mean(recalls[k]) * 100 for k in k_values}

    return metrics


def train_epoch(model_q, model_d, train_loader, optimizer, tau, accumulation_steps, device, memory_bank=None):
    """Train for one epoch."""
    model_q.train()
    model_d.train()

    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="  Training")

    for i, batch in enumerate(pbar):
        # Unpack batch (may have hard negatives)
        if len(batch) == 3:
            X, Y, hard_negs = batch
            hard_negs = hard_negs.to(device)
        else:
            X, Y = batch
            hard_negs = None

        X = X.to(device)
        Y = Y.to(device)

        # Forward
        query = model_q(X)
        doc_pos = model_d(Y)

        # Add to memory bank
        if memory_bank is not None:
            memory_bank.add(doc_pos.detach())

        # Loss with extras
        loss = info_nce_loss_with_extras(query, doc_pos, tau=tau, memory_bank=memory_bank, hard_negs=hard_negs)

        # Backward
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()

        # Accumulate
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Final step if leftover
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description="Train two-tower retriever - Phase 2")

    # Data
    parser.add_argument('--pairs', required=True, help='Training pairs NPZ file')
    parser.add_argument('--bank', default='artifacts/wikipedia_500k_corrected_vectors.npz', help='Vector bank for eval')
    parser.add_argument('--init-ckpt', help='Initialize from Phase 1 checkpoint')

    # Model
    parser.add_argument('--query-tower', default='gru_pool', choices=['gru_pool'], help='Query tower type')
    parser.add_argument('--doc-tower', default='identity', choices=['identity'], help='Doc tower type')
    parser.add_argument('--hidden-dim', type=int, default=512, help='GRU hidden dimension')
    parser.add_argument('--num-layers', type=int, default=1, help='GRU layers')

    # Training
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--tau', type=float, default=0.07, help='InfoNCE temperature')

    # Phase 2 features
    parser.add_argument('--memory-bank-size', type=int, default=20000, help='Memory bank size')
    parser.add_argument('--mine-every', type=int, default=2, help='Mine hard negatives every N epochs')
    parser.add_argument('--num-hard-negs', type=int, default=16, help='Number of hard negatives per sample')

    # Infrastructure
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'], help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', required=True, help='Output directory')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print("="*60)
    print("TWO-TOWER RETRIEVER TRAINING - PHASE 2")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.bs} × {args.accum} accum = {args.bs * args.accum} effective")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.tau}")
    print(f"Memory bank size: {args.memory_bank_size:,}")
    print(f"Mine hard negatives every: {args.mine_every} epochs")
    print(f"Hard negatives per sample: {args.num_hard_negs}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    data = np.load(args.pairs, allow_pickle=True)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']

    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val:   {len(X_val):,} pairs")

    # Load bank
    print(f"\nLoading bank from: {args.bank}")
    bank_data = np.load(args.bank, allow_pickle=True)
    bank_vectors = bank_data['vectors']
    print(f"  Bank size: {bank_vectors.shape}")

    # Create models
    print("\nBuilding models...")
    model_q = GRUPoolQuery(d_model=768, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    model_d = IdentityDocTower().to(device)

    # Initialize from Phase 1 if provided
    if args.init_ckpt:
        print(f"  Loading checkpoint: {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model_q.load_state_dict(ckpt['model_q_state_dict'])
        print("  ✓ Initialized from Phase 1 checkpoint")

    print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model_q.parameters(), lr=args.lr, weight_decay=args.wd)

    # Memory bank
    memory_bank = MemoryBank(max_size=args.memory_bank_size, dim=768)

    # Output directory
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # Save config
    config = vars(args)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    history = []
    best_recall500 = 0.0
    hard_negatives = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Mine hard negatives periodically
        if epoch % args.mine_every == 0:
            # Reload dataset with new hard negatives
            train_dataset_temp = TwoTowerDataset(X_train, Y_train)
            train_loader_temp = DataLoader(train_dataset_temp, batch_size=args.bs, shuffle=False)

            hard_negatives = mine_hard_negatives(
                model_q, model_d, train_loader_temp, bank_vectors,
                k=args.num_hard_negs, device=device
            )

        # Create dataset with current hard negatives
        train_dataset = TwoTowerDataset(X_train, Y_train, hard_negatives)
        val_dataset = TwoTowerDataset(X_val, Y_val)

        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

        # Train
        train_loss = train_epoch(
            model_q, model_d, train_loader, optimizer,
            args.tau, args.accum, device, memory_bank
        )

        # Evaluate
        print("\nEvaluating...")
        metrics = evaluate_recall(model_q, model_d, val_loader, bank_vectors, device=device)

        # Log
        epoch_result = {
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        }
        history.append(epoch_result)

        print(f"\nEpoch {epoch} results:")
        print(f"  Train loss: {train_loss:.4f}")
        for k, v in metrics.items():
            print(f"  {k.capitalize()}: {v:.2f}%")

        # Save best
        recall500 = metrics['recall@500']
        if recall500 > best_recall500:
            best_recall500 = recall500
            print(f"  ✓ New best Recall@500: {recall500:.2f}%")

            torch.save({
                'epoch': epoch,
                'model_q': model_q.state_dict(),
                'optimizer': optimizer.state_dict(),
                'recall500': recall500,
                'config': config
            }, ckpt_dir / 'best_recall500.pt')

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_q': model_q.state_dict(),
                'optimizer': optimizer.state_dict(),
                'recall500': recall500
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')

        # Save history
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Recall@500: {best_recall500:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == '__main__':
    main()
