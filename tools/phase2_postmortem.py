#!/usr/bin/env python3
"""
Phase 2 Post-Mortem Diagnostics

Confirms failure mode by analyzing:
1. Cosine separation: E[cos(q,d_pos)] vs E[cos(q,d_neg_hard)]
2. Near-duplicate rate in mined negatives
3. Bank alignment (distribution check)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


class GRUPoolQuery(torch.nn.Module):
    """Query tower from Phase 2"""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(d_model, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        # x: (B, L, D)
        out, _ = self.gru(x)  # (B, L, 2*hidden)
        pooled = out.mean(dim=1)  # mean pooling
        proj = self.proj(pooled)  # (B, D)
        return F.normalize(proj, dim=-1)


class IdentityDocTower(torch.nn.Module):
    """Document tower (just L2 norm)"""
    def forward(self, x):
        return F.normalize(x, dim=-1)


def load_model(ckpt_path, device):
    """Load Phase 2 checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_q = GRUPoolQuery(d_model=768, hidden_dim=512, num_layers=1).to(device)
    model_d = IdentityDocTower().to(device)

    # Handle both checkpoint formats
    if 'model_q_state_dict' in ckpt:
        model_q.load_state_dict(ckpt['model_q_state_dict'])
    else:
        model_q.load_state_dict(ckpt['model_q'])

    model_q.eval()
    model_d.eval()

    return model_q, model_d


def compute_cosine_separation(model_q, model_d, val_loader, bank_vectors, device, num_hard_negs=16):
    """
    Compute margin: E[cos(q, d_pos)] - E[cos(q, d_neg_hard)]
    """
    print("\n=== Cosine Separation Analysis ===")

    pos_sims = []
    hard_neg_sims = []

    # Sample bank if too large (avoid memory issues)
    if len(bank_vectors) > 100000:
        print(f"  Sampling bank: {len(bank_vectors):,} → 100,000 vectors")
        sample_idxs = np.random.choice(len(bank_vectors), size=100000, replace=False)
        bank_sampled = bank_vectors[sample_idxs]
    else:
        bank_sampled = bank_vectors

    # Build FAISS index for hard negative mining
    import faiss
    bank_norm = bank_sampled / (np.linalg.norm(bank_sampled, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(768)
    index.add(bank_norm.astype(np.float32))

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(val_loader, desc="Computing separations"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Encode
            q = model_q(X_batch)  # (B, D)
            d_pos = model_d(Y_batch)  # (B, D)

            # Positive similarity
            pos_sim = (q * d_pos).sum(dim=-1)  # (B,)
            pos_sims.extend(pos_sim.cpu().numpy().tolist())

            # Mine hard negatives
            q_np = q.cpu().numpy().astype(np.float32)
            D, I = index.search(q_np, num_hard_negs + 1)  # +1 to exclude positive

            # Get hard negative similarities
            for i in range(len(q)):
                # Exclude the positive (if it appears in top-K)
                hard_idxs = I[i][:num_hard_negs]
                hard_vecs = bank_norm[hard_idxs]  # (K, D)
                hard_sims = (q_np[i] @ hard_vecs.T).tolist()  # (K,)
                hard_neg_sims.extend(hard_sims)

    pos_mean = np.mean(pos_sims)
    hard_neg_mean = np.mean(hard_neg_sims)
    margin = pos_mean - hard_neg_mean

    print(f"  E[cos(q, d_pos)]: {pos_mean:.4f}")
    print(f"  E[cos(q, d_neg_hard)]: {hard_neg_mean:.4f}")
    print(f"  Margin Δ: {margin:.4f}")

    if margin < 0.05:
        print(f"  ⚠️  MARGIN COLLAPSE! Margin < 0.05 means hard negs are too hard/impure.")
    elif margin < 0.10:
        print(f"  ⚠️  Low margin. Hard negatives may be too challenging.")
    else:
        print(f"  ✓ Healthy margin.")

    return {
        'pos_mean': float(pos_mean),
        'hard_neg_mean': float(hard_neg_mean),
        'margin': float(margin),
        'pos_std': float(np.std(pos_sims)),
        'hard_neg_std': float(np.std(hard_neg_sims))
    }


def compute_near_duplicate_rate(model_q, val_loader, bank_vectors, device, threshold=0.98):
    """
    Count hard negatives with cos(query)>0.98 AND cos(positive)>0.98
    """
    print("\n=== Near-Duplicate Rate in Hard Negatives ===")

    # Sample bank if too large
    if len(bank_vectors) > 100000:
        print(f"  Sampling bank: {len(bank_vectors):,} → 100,000 vectors")
        sample_idxs = np.random.choice(len(bank_vectors), size=100000, replace=False)
        bank_sampled = bank_vectors[sample_idxs]
    else:
        bank_sampled = bank_vectors

    import faiss
    bank_norm = bank_sampled / (np.linalg.norm(bank_sampled, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(768)
    index.add(bank_norm.astype(np.float32))

    total_negs = 0
    near_dup_count = 0

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(val_loader, desc="Checking duplicates"):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            q = model_q(X_batch)
            d_pos = Y_batch  # Already normalized

            q_np = q.cpu().numpy().astype(np.float32)
            d_pos_np = d_pos.cpu().numpy().astype(np.float32)

            # Mine top-16 hard negs
            D, I = index.search(q_np, 16)

            for i in range(len(q)):
                hard_idxs = I[i]
                hard_vecs = bank_norm[hard_idxs]

                # Check each hard neg
                for j, hard_vec in enumerate(hard_vecs):
                    cos_q_neg = float(q_np[i] @ hard_vec)
                    cos_pos_neg = float(d_pos_np[i] @ hard_vec)

                    total_negs += 1
                    if cos_q_neg > threshold and cos_pos_neg > threshold:
                        near_dup_count += 1

    dup_rate = near_dup_count / total_negs if total_negs > 0 else 0.0

    print(f"  Total hard negatives checked: {total_negs}")
    print(f"  Near-duplicates (cos>0.98 to both q and d_pos): {near_dup_count}")
    print(f"  Duplicate rate: {dup_rate*100:.2f}%")

    if dup_rate > 0.02:
        print(f"  ⚠️  HIGH DUPLICATE RATE! Miner is feeding duplicates/neighbors.")
    else:
        print(f"  ✓ Low duplicate rate.")

    return {
        'total_negs': total_negs,
        'near_duplicates': near_dup_count,
        'duplicate_rate': float(dup_rate)
    }


def compute_bank_alignment(bank_vectors, sample_size=10000):
    """
    Plot distribution of cos(random query, random doc) to check if centered at 0
    """
    print("\n=== Bank Alignment Sanity Check ===")

    # Random sample
    n = min(sample_size, len(bank_vectors))
    idxs = np.random.choice(len(bank_vectors), size=n, replace=False)
    sample = bank_vectors[idxs]

    # Normalize
    sample_norm = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)

    # Compute pairwise cosines (sample vs sample)
    cos_matrix = sample_norm @ sample_norm.T

    # Get off-diagonal elements (exclude self-similarity)
    mask = ~np.eye(n, dtype=bool)
    off_diag = cos_matrix[mask]

    mean_cos = np.mean(off_diag)
    std_cos = np.std(off_diag)

    print(f"  Random doc-doc cosines:")
    print(f"    Mean: {mean_cos:.4f}")
    print(f"    Std: {std_cos:.4f}")
    print(f"    Min: {np.min(off_diag):.4f}")
    print(f"    Max: {np.max(off_diag):.4f}")

    if abs(mean_cos) > 0.10:
        print(f"  ⚠️  Mean far from 0.0! Consider whitening docs (mean-center + unit-variance).")
    else:
        print(f"  ✓ Bank is reasonably centered.")

    return {
        'mean': float(mean_cos),
        'std': float(std_cos),
        'min': float(np.min(off_diag)),
        'max': float(np.max(off_diag))
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Post-Mortem Diagnostics")
    parser.add_argument('--ckpt', type=str, required=True, help='Phase 2 checkpoint path')
    parser.add_argument('--pairs', type=str, required=True, help='Validation pairs NPZ')
    parser.add_argument('--bank', type=str, required=True, help='Vector bank NPZ')
    parser.add_argument('--out', type=str, default='runs/twotower_v3_phase2/postmortem.json',
                        help='Output JSON with diagnostics')
    parser.add_argument('--device', type=str, default='mps', help='Device (mps/cpu)')
    args = parser.parse_args()

    print("============================================================")
    print("PHASE 2 POST-MORTEM DIAGNOSTICS")
    print("============================================================")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Pairs: {args.pairs}")
    print(f"Bank: {args.bank}")
    print(f"Device: {args.device}")
    print()

    device = torch.device(args.device)

    # Load model
    print("Loading model...")
    model_q, model_d = load_model(args.ckpt, device)
    print(f"  ✓ Model loaded")

    # Load data
    print("Loading data...")
    data = np.load(args.pairs)
    X_val = torch.from_numpy(data['X_val']).float()
    Y_val = torch.from_numpy(data['Y_val']).float()
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"  Val pairs: {len(val_dataset)}")

    bank_data = np.load(args.bank)
    bank_vectors = bank_data['vectors']
    print(f"  Bank size: {bank_vectors.shape}")

    # Run diagnostics
    results = {}

    # 1. Cosine separation
    results['cosine_separation'] = compute_cosine_separation(
        model_q, model_d, val_loader, bank_vectors, device, num_hard_negs=16
    )

    # 2. Near-duplicate rate
    results['near_duplicate_rate'] = compute_near_duplicate_rate(
        model_q, val_loader, bank_vectors, device, threshold=0.98
    )

    # 3. Bank alignment
    results['bank_alignment'] = compute_bank_alignment(bank_vectors, sample_size=10000)

    # Save results
    print(f"\n=== Saving Results ===")
    print(f"  Output: {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n============================================================")
    print("POST-MORTEM COMPLETE")
    print("============================================================")

    # Summary
    print("\nSummary:")
    print(f"  Margin: {results['cosine_separation']['margin']:.4f}")
    print(f"  Duplicate rate: {results['near_duplicate_rate']['duplicate_rate']*100:.2f}%")
    print(f"  Bank alignment: {results['bank_alignment']['mean']:.4f} ± {results['bank_alignment']['std']:.4f}")


if __name__ == '__main__':
    main()
