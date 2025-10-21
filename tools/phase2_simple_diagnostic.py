#!/usr/bin/env python3
"""
Simplified Phase 2 Diagnostic (avoids tqdm/FAISS deadlock)
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


class GRUPoolQuery(torch.nn.Module):
    """Query tower from Phase 2"""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(d_model, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        proj = self.proj(pooled)
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

    if 'model_q_state_dict' in ckpt:
        model_q.load_state_dict(ckpt['model_q_state_dict'])
    else:
        model_q.load_state_dict(ckpt['model_q'])

    model_q.eval()
    model_d.eval()

    return model_q, model_d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pairs', type=str, required=True)
    parser.add_argument('--bank', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sample-size', type=int, default=500, help='Pairs to sample for diagnostics')
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2 SIMPLIFIED DIAGNOSTIC")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print()

    device = torch.device(args.device)

    # Load model
    print("Loading model...")
    model_q, model_d = load_model(args.ckpt, device)
    print("  ✓ Model loaded")

    # Load data
    print("Loading data...")
    data = np.load(args.pairs)
    X_val = torch.from_numpy(data['X_val']).float()
    Y_val = torch.from_numpy(data['Y_val']).float()
    print(f"  Val pairs: {len(X_val)}")

    bank_data = np.load(args.bank)
    bank_vectors = bank_data['vectors']
    print(f"  Bank size: {bank_vectors.shape}")

    # Sample pairs for faster diagnostics
    n_sample = min(args.sample_size, len(X_val))
    sample_idxs = np.random.choice(len(X_val), size=n_sample, replace=False)
    X_sample = X_val[sample_idxs].to(device)
    Y_sample = Y_val[sample_idxs].to(device)
    print(f"  Using {n_sample} pairs for diagnostics")

    # Sample bank
    bank_sample_size = min(50000, len(bank_vectors))
    bank_idxs = np.random.choice(len(bank_vectors), size=bank_sample_size, replace=False)
    bank_sample = bank_vectors[bank_idxs]
    bank_norm = bank_sample / (np.linalg.norm(bank_sample, axis=1, keepdims=True) + 1e-9)
    print(f"  Sampled bank: {bank_sample_size}")

    # Build FAISS index
    print("\nBuilding FAISS index...")
    import faiss
    index = faiss.IndexFlatIP(768)
    index.add(bank_norm.astype(np.float32))
    print("  ✓ Index built")

    # Compute diagnostics
    print("\n=== Computing Metrics ===")

    with torch.no_grad():
        # Encode all samples at once
        print("  Encoding queries and positives...")
        q = model_q(X_sample)  # (N, D)
        d_pos = model_d(Y_sample)  # (N, D)

        # Positive similarities
        pos_sims = (q * d_pos).sum(dim=-1).cpu().numpy()  # (N,)

        # Mine hard negatives
        print("  Mining hard negatives via FAISS...")
        q_np = q.cpu().numpy().astype(np.float32)
        D, I = index.search(q_np, 16)  # Get top-16 hard negs

        # Compute hard negative similarities
        print("  Computing hard neg similarities...")
        hard_neg_sims = []
        near_dups = 0
        total_negs = 0

        d_pos_np = d_pos.cpu().numpy().astype(np.float32)

        for i in range(len(q)):
            hard_idxs = I[i]
            hard_vecs = bank_norm[hard_idxs]  # (16, D)

            # Cosines between query and hard negs
            hard_sims = (q_np[i] @ hard_vecs.T).tolist()
            hard_neg_sims.extend(hard_sims)

            # Check for near-duplicates (cos>0.98 to both q and pos)
            for hard_vec in hard_vecs:
                cos_q = float(q_np[i] @ hard_vec)
                cos_pos = float(d_pos_np[i] @ hard_vec)
                total_negs += 1
                if cos_q > 0.98 and cos_pos > 0.98:
                    near_dups += 1

    # Compute statistics
    pos_mean = float(np.mean(pos_sims))
    hard_neg_mean = float(np.mean(hard_neg_sims))
    margin = pos_mean - hard_neg_mean
    dup_rate = near_dups / total_negs if total_negs > 0 else 0.0

    # Bank alignment
    print("  Computing bank alignment...")
    align_sample_size = 5000
    align_idxs = np.random.choice(len(bank_sample), size=min(align_sample_size, len(bank_sample)), replace=False)
    align_sample = bank_norm[align_idxs]
    cos_matrix = align_sample @ align_sample.T
    mask = ~np.eye(len(align_sample), dtype=bool)
    off_diag = cos_matrix[mask]
    bank_mean = float(np.mean(off_diag))
    bank_std = float(np.std(off_diag))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n1. Cosine Separation:")
    print(f"   E[cos(q, d_pos)]: {pos_mean:.4f}")
    print(f"   E[cos(q, d_neg_hard)]: {hard_neg_mean:.4f}")
    print(f"   Margin Δ: {margin:.4f}")
    if margin < 0.05:
        print(f"   ⚠️  MARGIN COLLAPSE! Hard negs too hard/impure")
    elif margin < 0.10:
        print(f"   ⚠️  Low margin")
    else:
        print(f"   ✓ Healthy margin")

    print("\n2. Near-Duplicate Rate:")
    print(f"   Total hard negs checked: {total_negs}")
    print(f"   Near-duplicates (cos>0.98 to both): {near_dups}")
    print(f"   Duplicate rate: {dup_rate*100:.2f}%")
    if dup_rate > 0.02:
        print(f"   ⚠️  HIGH DUPLICATE RATE!")
    else:
        print(f"   ✓ Low duplicate rate")

    print("\n3. Bank Alignment:")
    print(f"   Random doc-doc cosines: {bank_mean:.4f} ± {bank_std:.4f}")
    if abs(bank_mean) > 0.10:
        print(f"   ⚠️  Mean far from 0.0!")
    else:
        print(f"   ✓ Bank reasonably centered")

    # Save results
    results = {
        'cosine_separation': {
            'pos_mean': pos_mean,
            'hard_neg_mean': hard_neg_mean,
            'margin': margin,
            'pos_std': float(np.std(pos_sims)),
            'hard_neg_std': float(np.std(hard_neg_sims))
        },
        'near_duplicate_rate': {
            'total_negs': total_negs,
            'near_duplicates': near_dups,
            'duplicate_rate': dup_rate
        },
        'bank_alignment': {
            'mean': bank_mean,
            'std': bank_std
        }
    }

    out_path = 'runs/twotower_v3_phase2/postmortem.json'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {out_path}")
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
