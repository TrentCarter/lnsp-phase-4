#!/usr/bin/env python3
"""
Two-Tower Retriever with Same-Article + Near-Miss Hard Negatives

Combines:
- Same-article negatives (K=3): local discrimination within article
- Near-miss negatives (K=1): global discrimination from retrieval errors

Usage:
    python app/lvm/train_twotower_hardneg.py \
        --resume artifacts/lvm/models/twotower_samearticle/epoch3.pt \
        --train-npz artifacts/lvm/train_clean_disjoint.npz \
        --nearmiss-jsonl artifacts/mined/nearmiss_ep3.jsonl \
        --same-article-k 3 \
        --epochs 1 --eval-every 1 \
        --batch-size 256 --lr 1e-4 \
        --device mps \
        --save-dir artifacts/lvm/models/twotower_hardneg
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


class HardNegativeDataset(Dataset):
    """Dataset with same-article + near-miss negatives."""

    def __init__(self, contexts, targets, article_ids, nearmiss_map, corpus_targets):
        self.contexts = contexts
        self.targets = targets
        self.article_ids = article_ids
        self.nearmiss_map = nearmiss_map  # query_idx -> [neg_indices]
        self.corpus_targets = corpus_targets  # All target vectors for near-miss lookup

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = torch.from_numpy(self.contexts[idx]).float()
        tgt = torch.from_numpy(self.targets[idx]).float()
        art_id = int(self.article_ids[idx])

        # Get near-miss negatives for this query
        nearmiss_indices = self.nearmiss_map.get(idx, [])
        if len(nearmiss_indices) > 0:
            nearmiss_vecs = self.corpus_targets[nearmiss_indices]
            nearmiss = torch.from_numpy(nearmiss_vecs).float()
        else:
            nearmiss = torch.empty(0, 768)

        return ctx, tgt, art_id, nearmiss


class TwoTowerInfoNCEWithHardNeg(nn.Module):
    """InfoNCE with same-article + near-miss negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, p_pos, p_same_article=None, p_nearmiss=None):
        """
        Args:
            q: [B, 768] query vectors
            p_pos: [B, 768] positive payload vectors
            p_same_article: [B, K_sa, 768] same-article negatives (optional)
            p_nearmiss: [B, K_nm, 768] near-miss negatives (optional)

        Returns:
            loss, metrics
        """
        B = q.shape[0]

        # Compute similarity with positives: [B, B]
        logits = torch.mm(q, p_pos.t()) / self.temperature

        # Add same-article negatives
        if p_same_article is not None and p_same_article.shape[1] > 0:
            same_article_sim = torch.bmm(
                q.unsqueeze(1),  # [B, 1, 768]
                p_same_article.transpose(1, 2)  # [B, 768, K_sa]
            ).squeeze(1) / self.temperature  # [B, K_sa]
            logits = torch.cat([logits, same_article_sim], dim=1)

        # Add near-miss negatives
        if p_nearmiss is not None and p_nearmiss.shape[1] > 0:
            nearmiss_sim = torch.bmm(
                q.unsqueeze(1),  # [B, 1, 768]
                p_nearmiss.transpose(1, 2)  # [B, 768, K_nm]
            ).squeeze(1) / self.temperature  # [B, K_nm]
            logits = torch.cat([logits, nearmiss_sim], dim=1)

        # Labels: positives are on diagonal (first B columns)
        labels = torch.arange(B, device=q.device)

        # InfoNCE loss
        loss = F.cross_entropy(logits, labels)

        # Metrics
        with torch.no_grad():
            # Positive cosines (diagonal of first B×B block)
            pos_cos = torch.diag(logits[:, :B]).mean() * self.temperature

            # Negative cosines (all non-diagonal)
            mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
            neg_cos_batch = logits[:, :B][mask].mean() * self.temperature

            # Same-article negatives
            if p_same_article is not None and p_same_article.shape[1] > 0:
                neg_cos_sa = logits[:, B:B+p_same_article.shape[1]].mean() * self.temperature
            else:
                neg_cos_sa = neg_cos_batch

            # Near-miss negatives
            if p_nearmiss is not None and p_nearmiss.shape[1] > 0:
                start_nm = B + (p_same_article.shape[1] if p_same_article is not None else 0)
                neg_cos_nm = logits[:, start_nm:].mean() * self.temperature
            else:
                neg_cos_nm = neg_cos_batch

            metrics = {
                'pos_cos': pos_cos.item(),
                'neg_cos_batch': neg_cos_batch.item(),
                'neg_cos_sa': neg_cos_sa.item(),
                'neg_cos_nm': neg_cos_nm.item(),
                'separation': (pos_cos - neg_cos_batch).item(),
            }

        return loss, metrics


def sample_same_article_negatives(article_ids, targets, article_index, k=3, device='cpu'):
    """Sample k same-article negatives for each item."""
    B = len(article_ids)
    negatives = []

    for i, art_id in enumerate(article_ids):
        candidates = [c for c in article_index[int(art_id)] if c != i]
        if len(candidates) == 0:
            # No same-article candidates, use zeros
            negatives.append(np.zeros((k, 768), dtype='float32'))
            continue

        if len(candidates) >= k:
            sampled = np.random.choice(candidates, k, replace=False)
        else:
            sampled = np.random.choice(candidates, k, replace=True)

        neg_vecs = targets[sampled]
        negatives.append(neg_vecs)

    negatives = np.stack(negatives, axis=0)  # [B, k, 768]
    return torch.from_numpy(negatives).float().to(device)


def collate_with_hardneg(batch, targets, article_index, same_article_k=3, device='cpu'):
    """Collate with same-article + near-miss negatives."""
    contexts, pos_targets, art_ids, nearmiss = zip(*batch)

    # Stack batch
    contexts = torch.stack(contexts).to(device)
    pos_targets = torch.stack(pos_targets).to(device)

    # Sample same-article negatives
    p_same_article = sample_same_article_negatives(
        art_ids, targets, article_index, k=same_article_k, device=device
    )

    # Stack near-miss negatives (pad to same length)
    max_nm = max(nm.shape[0] for nm in nearmiss)
    if max_nm > 0:
        B = len(nearmiss)
        p_nearmiss = torch.zeros(B, max_nm, 768, device=device)
        for i, nm in enumerate(nearmiss):
            if nm.shape[0] > 0:
                p_nearmiss[i, :nm.shape[0], :] = nm.to(device)
    else:
        p_nearmiss = None

    return contexts, pos_targets, p_same_article, p_nearmiss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--train-npz', type=Path, required=True)
    parser.add_argument('--nearmiss-jsonl', type=Path, required=True)
    parser.add_argument('--same-article-k', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=Path, required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("TWO-TOWER TRAINING: SAME-ARTICLE + NEAR-MISS HARD NEGATIVES")
    print("=" * 80)
    print(f"Resume: {args.resume}")
    print(f"Train NPZ: {args.train_npz}")
    print(f"Near-miss: {args.nearmiss_jsonl}")
    print(f"Same-article K: {args.same_article_k}")
    print(f"Device: {args.device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    ckpt_args = checkpoint['args']

    q_tower = QueryTower(
        backbone_type=ckpt_args.get('arch_q', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
    ).to(args.device)
    q_tower.load_state_dict(checkpoint['q_tower_state_dict'])

    p_tower = PayloadTower(
        backbone_type=ckpt_args.get('arch_p', 'mamba_s'),
        d_model=ckpt_args.get('d_model', 768),
    ).to(args.device)
    p_tower.load_state_dict(checkpoint['p_tower_state_dict'])

    print(f"  Loaded epoch {checkpoint['epoch']}")
    print()

    # Load training data
    print("Loading training data...")
    data = np.load(args.train_npz, allow_pickle=True)
    contexts = data['context_sequences']  # [N, 5, 768]
    targets = data['target_vectors']  # [N, 768]
    truth_keys = data['truth_keys']  # [N, 2] -> [article_id, chunk_id]
    article_ids = truth_keys[:, 0]
    print(f"  Training size: {len(contexts)}")
    print()

    # Build article index (article_id -> list of training indices)
    print("Building article index...")
    article_index = defaultdict(list)
    for i, art_id in enumerate(article_ids):
        article_index[int(art_id)].append(i)
    print(f"  Articles: {len(article_index)}")
    print()

    # Load near-miss negatives
    print("Loading near-miss negatives...")
    nearmiss_map = {}
    with open(args.nearmiss_jsonl) as f:
        for line in f:
            record = json.loads(line)
            # Near-miss was mined on eval set, need to map to training indices
            # For now, skip if not in training set
            q_idx = record['query_idx']
            if q_idx < len(contexts):
                nearmiss_map[q_idx] = record['negative_indices']
    print(f"  Near-miss records: {len(nearmiss_map)}")
    print()

    # Create dataset
    dataset = HardNegativeDataset(contexts, targets, article_ids, nearmiss_map, targets)

    # Create dataloader with custom collate
    def collate_fn(batch):
        return collate_with_hardneg(
            batch, targets, article_index,
            same_article_k=args.same_article_k,
            device=args.device
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Create loss
    criterion = TwoTowerInfoNCEWithHardNeg(temperature=0.07)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(q_tower.parameters()) + list(p_tower.parameters()),
        lr=args.lr,
    )

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, start_epoch + args.epochs):
        q_tower.train()
        p_tower.train()

        epoch_metrics = defaultdict(list)
        epoch_start = time.time()

        for batch_idx, (contexts, pos_targets, p_sa, p_nm) in enumerate(dataloader):
            # Forward
            q = q_tower(contexts)
            p = p_tower(pos_targets)

            loss, metrics = criterion(q, p, p_sa, p_nm)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            epoch_metrics['loss'].append(loss.item())
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Δ: {metrics['separation']:.4f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        print()
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.1f} min")
        print(f"  Loss: {np.mean(epoch_metrics['loss']):.4f}")
        print(f"  Pos cos: {np.mean(epoch_metrics['pos_cos']):.4f}")
        print(f"  Neg cos (batch): {np.mean(epoch_metrics['neg_cos_batch']):.4f}")
        print(f"  Neg cos (same-article): {np.mean(epoch_metrics['neg_cos_sa']):.4f}")
        print(f"  Neg cos (near-miss): {np.mean(epoch_metrics['neg_cos_nm']):.4f}")
        print(f"  Separation: {np.mean(epoch_metrics['separation']):.4f}")
        print()

        # Save checkpoint
        args.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'q_tower_state_dict': q_tower.state_dict(),
            'p_tower_state_dict': p_tower.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
        }
        torch.save(checkpoint, Path(args.save_dir) / f'epoch{epoch+1}.pt')
        print(f"✅ Saved: {args.save_dir}/epoch{epoch+1}.pt")
        print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
