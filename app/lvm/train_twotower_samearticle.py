#!/usr/bin/env python3
"""
Two-Tower Retriever with Same-Article Negatives

Adds 1-3 same-article negatives per query to tighten local ranking.

Usage:
    python app/lvm/train_twotower_samearticle.py \
        --resume artifacts/lvm/models/twotower_mamba_s/epoch2.pt \
        --train-npz artifacts/lvm/train_clean_disjoint.npz \
        --same-article-k 3 \
        --epochs 1 --eval-every 1 \
        --batch-size 256 --lr 1e-4 \
        --device mps \
        --save-dir artifacts/lvm/models/twotower_samearticle
"""
import argparse
import json
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


class TwoTowerDatasetWithArticleID(Dataset):
    """Dataset with article IDs for same-article negative sampling."""

    def __init__(self, contexts, targets, article_ids):
        self.contexts = contexts
        self.targets = targets
        self.article_ids = article_ids

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = torch.from_numpy(self.contexts[idx]).float()
        tgt = torch.from_numpy(self.targets[idx]).float()
        art_id = int(self.article_ids[idx])
        return ctx, tgt, art_id


class TwoTowerInfoNCEWithSameArticle(nn.Module):
    """InfoNCE with same-article negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, p_pos, p_same_article=None):
        """
        Args:
            q: [B, 768] query vectors
            p_pos: [B, 768] positive payload vectors
            p_same_article: [B, K_sa, 768] same-article negatives (optional)

        Returns:
            loss, metrics
        """
        B = q.shape[0]

        # Compute similarity with positives: [B, B]
        logits = torch.mm(q, p_pos.t()) / self.temperature

        # Add same-article negatives if provided
        if p_same_article is not None:
            # p_same_article: [B, K_sa, 768]
            # q: [B, 768] → [B, 1, 768]
            # Compute: [B, K_sa]
            same_article_sim = torch.bmm(
                q.unsqueeze(1),  # [B, 1, 768]
                p_same_article.transpose(1, 2)  # [B, 768, K_sa]
            ).squeeze(1) / self.temperature  # [B, K_sa]

            # Concatenate: [B, B+K_sa]
            logits = torch.cat([logits, same_article_sim], dim=1)

        # Labels: positives are on diagonal (first B columns)
        labels = torch.arange(B, device=q.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            # Positive cosines (diagonal of first B×B block)
            pos_cos = torch.diag(logits[:, :B]).mean() * self.temperature

            # Negative cosines
            mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
            neg_cos_inbatch = logits[:, :B][mask].mean() * self.temperature

            if p_same_article is not None:
                neg_cos_samearticle = same_article_sim.mean() * self.temperature
                neg_cos = (neg_cos_inbatch + neg_cos_samearticle) / 2
            else:
                neg_cos = neg_cos_inbatch

            separation = pos_cos - neg_cos

        metrics = {
            'loss': loss.item(),
            'pos_cos': pos_cos.item(),
            'neg_cos': neg_cos.item(),
            'separation': separation.item(),
        }

        return loss, metrics


def build_article_index(article_ids):
    """Build {article_id: [indices]} for fast same-article sampling."""
    index = defaultdict(list)
    for idx, art_id in enumerate(article_ids):
        index[int(art_id)].append(idx)
    return dict(index)


def sample_same_article_negatives(article_ids, targets, article_index, k=3):
    """
    Sample k same-article negatives for each item.

    Returns:
        negatives: [B, k, 768] tensor
    """
    B = len(article_ids)
    negatives = []

    for i, art_id in enumerate(article_ids):
        art_id = int(art_id)
        candidates = article_index.get(art_id, [i])

        # Remove current index
        candidates = [c for c in candidates if c != i]

        if len(candidates) == 0:
            # Fallback: use random from batch
            candidates = list(range(B))
            candidates.remove(i)

        # Sample k (with replacement if needed)
        if len(candidates) >= k:
            sampled = np.random.choice(candidates, k, replace=False)
        else:
            sampled = np.random.choice(candidates, k, replace=True)

        neg_vecs = targets[sampled]
        negatives.append(neg_vecs)

    negatives = np.stack(negatives, axis=0)  # [B, k, 768]
    return torch.from_numpy(negatives).float()


def train_epoch(q_tower, p_tower, dataloader, loss_fn, optimizer, device,
                targets_full, article_index, same_article_k=3):
    """Train one epoch with same-article negatives."""
    q_tower.train()
    p_tower.train()

    total_loss = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_sep = 0.0
    num_batches = 0

    for contexts, targets, article_ids in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Sample same-article negatives
        p_same_article = sample_same_article_negatives(
            article_ids.numpy(),
            targets_full,
            article_index,
            k=same_article_k
        ).to(device)

        # Forward
        q = q_tower(contexts)
        p = p_tower(targets)

        # Loss
        loss, metrics = loss_fn(q, p, p_same_article)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += metrics['loss']
        total_pos += metrics['pos_cos']
        total_neg += metrics['neg_cos']
        total_sep += metrics['separation']
        num_batches += 1

    return {
        'train_loss': total_loss / num_batches,
        'train_pos_cos': total_pos / num_batches,
        'train_neg_cos': total_neg / num_batches,
        'train_separation': total_sep / num_batches,
    }


@torch.no_grad()
def validate(q_tower, p_tower, dataloader, device):
    """Validate on held-out data."""
    q_tower.eval()
    p_tower.eval()

    all_q = []
    all_p = []

    for contexts, targets, _ in dataloader:
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

    # Loss
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--same-article-k', type=int, default=3,
                        help='Number of same-article negatives per query')

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
    parser.add_argument('--eval-every', type=int, default=1)

    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TWO-TOWER TRAINING (Same-Article Negatives)")
    print("=" * 80)
    print(f"Resume from: {args.resume}")
    print(f"Same-article K: {args.same_article_k}")
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

    train_dataset = TwoTowerDatasetWithArticleID(
        contexts[indices[:n_train]],
        targets[indices[:n_train]],
        article_ids[indices[:n_train]]
    )
    val_dataset = TwoTowerDatasetWithArticleID(
        contexts[indices[n_train:]],
        targets[indices[n_train:]],
        article_ids[indices[n_train:]]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
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

    # Loss
    loss_fn = TwoTowerInfoNCEWithSameArticle(temperature=args.temperature)

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
            q_tower, p_tower, train_loader, loss_fn, optimizer, args.device,
            targets, article_index, args.same_article_k
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
        print(f"  Train Δ (pos-neg): {train_metrics['train_separation']:.4f}")
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
