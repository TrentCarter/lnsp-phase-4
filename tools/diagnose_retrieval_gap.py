#!/usr/bin/env python3
"""
Diagnostic Tool for Retrieval Gap Analysis
==========================================

Analyzes why models with good validation cosine (0.51-0.58) achieve 0% retrieval.

Key metrics:
1. Rank-of-truth CDF: Where does ground truth rank in full 584k index?
2. Cosine margin: gap between cos(pred,true) vs max impostor cosine
3. Distribution analysis: pred vector geometry vs GTR-T5 space

Usage:
    python tools/diagnose_retrieval_gap.py \
        --model artifacts/lvm/models/mamba_sandwich/best.pt \
        --eval-npz artifacts/lvm/eval_v2_ready_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
        --out artifacts/lvm/diagnosis_mamba_sandwich.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model


def load_model(checkpoint_path: Path, device: str):
    """Load Mamba model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    model_type = args['model_type']

    # Build kwargs based on model type
    model_kwargs = {
        'd_model': args['d_model'],
        'd_state': args['d_state'],
        'conv_sz': args['conv_sz'],
        'expand': args['expand'],
        'dropout': args['dropout'],
    }

    if 'sandwich' in model_type:
        model_kwargs.update({
            'n_layers_mamba': args.get('n_layers_mamba', 8),
            'n_layers_local': args.get('n_layers_local', 4),
            'local_attn_win': args.get('local_attn_win', 8),
            'n_heads': args.get('n_heads', 4),
        })
    elif 'hybrid' in model_type:
        model_kwargs.update({
            'n_layers': args['n_layers'],
            'local_attn_win': args.get('local_attn_win', 8),
            'local_attn_every': args.get('local_attn_every', 4),
            'n_heads': args.get('n_heads', 4),
        })
    else:
        model_kwargs['n_layers'] = args['n_layers']

    if 'gr' in model_type:
        model_kwargs['gru_hidden'] = args.get('gru_hidden', 256)

    model_kwargs.update({
        'use_alignment_head': args.get('use_alignment_head', False),
        'alignment_alpha': args.get('alignment_alpha', 0.25),
    })

    model = create_model(model_type=model_type, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, args


@torch.no_grad()
def generate_predictions(model, contexts, device, batch_size=128):
    """Generate predictions for all contexts."""
    contexts_tensor = torch.from_numpy(contexts).float()
    all_preds = []

    for i in range(0, len(contexts), batch_size):
        batch = contexts_tensor[i:i+batch_size].to(device)
        preds = model(batch)

        if len(preds.shape) == 3:
            preds = preds[:, -1, :]

        preds = F.normalize(preds, p=2, dim=1)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def compute_rank_of_truth(pred_vecs, truth_keys, payload, index, article_chunk_to_id, nprobe=256):
    """
    Compute rank of ground truth in full index for each prediction.

    Returns:
        ranks: Array of ranks (0-indexed, -1 if not found in top-10000)
        cosines_true: Cosine similarity between pred and true target
        cosines_rank1: Cosine similarity between pred and rank-1 result
    """
    print(f"\nComputing rank-of-truth for {len(pred_vecs)} predictions...")
    print(f"  Using nprobe={nprobe} for exhaustive search")

    # Set high nprobe for better recall
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe

    ranks = []
    cosines_true = []
    cosines_rank1 = []

    for i in range(len(pred_vecs)):
        pred = pred_vecs[i]
        art_idx, chunk_idx = truth_keys[i]
        truth_id = article_chunk_to_id[(int(art_idx), int(chunk_idx))]

        # Get ground truth vector
        _, _, true_vec = payload[truth_id]
        cos_true = float(np.dot(pred, true_vec))
        cosines_true.append(cos_true)

        # Search index for top-10000 (to find rank)
        query = pred.reshape(1, -1).astype(np.float32)
        D, I = index.search(query, min(10000, index.ntotal))

        # Find rank of ground truth
        if truth_id in I[0]:
            rank = int(np.where(I[0] == truth_id)[0][0])
            ranks.append(rank)
        else:
            ranks.append(-1)  # Not in top-10000

        # Get rank-1 cosine
        rank1_id = int(I[0][0])
        _, _, rank1_vec = payload[rank1_id]
        cos_rank1 = float(np.dot(pred, rank1_vec))
        cosines_rank1.append(cos_rank1)

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(pred_vecs)}")

    return np.array(ranks), np.array(cosines_true), np.array(cosines_rank1)


def main():
    ap = argparse.ArgumentParser(description="Diagnose retrieval gap for LVM models")
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to model checkpoint")
    ap.add_argument("--eval-npz", type=Path, required=True,
                    help="Path to evaluation NPZ")
    ap.add_argument("--payload", type=Path, required=True,
                    help="Path to payload NPY")
    ap.add_argument("--faiss", type=Path, required=True,
                    help="Path to FAISS index")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"],
                    help="Device for inference")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Number of samples to diagnose")
    ap.add_argument("--nprobe", type=int, default=256,
                    help="FAISS nprobe for exhaustive search")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON file")

    args = ap.parse_args()

    print("=" * 80)
    print("Retrieval Gap Diagnostic")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit} samples")
    print("=" * 80)

    # Load components
    print("\nLoading model...")
    model, model_args = load_model(args.model, args.device)
    print(f"  Model type: {model_args['model_type']}")

    print("\nLoading data...")
    eval_data = np.load(args.eval_npz, allow_pickle=True)
    contexts = eval_data['contexts'][:args.limit]
    targets = eval_data['targets'][:args.limit]
    truth_keys = eval_data['truth_keys'][:args.limit]

    payload = np.load(args.payload, allow_pickle=True).item()
    index = faiss.read_index(str(args.faiss))

    # Build reverse index
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    # Generate predictions
    print(f"\nGenerating predictions...")
    pred_vecs = generate_predictions(model, contexts, args.device)

    # Compute validation-style cosines
    val_cosines = [np.dot(pred_vecs[i], targets[i]) for i in range(len(pred_vecs))]

    # Compute ranks and margins
    ranks, cosines_true, cosines_rank1 = compute_rank_of_truth(
        pred_vecs, truth_keys, payload, index, article_chunk_to_id, args.nprobe
    )

    # Compute margins
    margins = cosines_true - cosines_rank1  # How much better is truth vs rank-1?

    # Analyze results
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    print("\n1. Validation Cosine (pred vs target):")
    print(f"  Mean: {np.mean(val_cosines):.4f}")
    print(f"  Median: {np.median(val_cosines):.4f}")
    print(f"  Range: [{np.min(val_cosines):.4f}, {np.max(val_cosines):.4f}]")

    print("\n2. Rank-of-Truth Distribution:")
    in_top50 = (ranks >= 0) & (ranks < 50)
    in_top500 = (ranks >= 0) & (ranks < 500)
    in_top5000 = (ranks >= 0) & (ranks < 5000)
    not_found = ranks < 0

    print(f"  In top-50: {np.sum(in_top50)}/{len(ranks)} ({100*np.mean(in_top50):.1f}%)")
    print(f"  In top-500: {np.sum(in_top500)}/{len(ranks)} ({100*np.mean(in_top500):.1f}%)")
    print(f"  In top-5000: {np.sum(in_top5000)}/{len(ranks)} ({100*np.mean(in_top5000):.1f}%)")
    print(f"  Not in top-10000: {np.sum(not_found)}/{len(ranks)} ({100*np.mean(not_found):.1f}%)")

    valid_ranks = ranks[ranks >= 0]
    if len(valid_ranks) > 0:
        print(f"\n  Rank percentiles (when found):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    {p}th: {int(np.percentile(valid_ranks, p))}")

    print("\n3. Cosine Margins (true vs rank-1 impostor):")
    print(f"  Mean margin: {np.mean(margins):.4f}")
    print(f"  Median margin: {np.median(margins):.4f}")
    print(f"  Positive margins: {np.sum(margins > 0)}/{len(margins)} ({100*np.mean(margins > 0):.1f}%)")
    print(f"  Margin > 0.05: {np.sum(margins > 0.05)}/{len(margins)} ({100*np.mean(margins > 0.05):.1f}%)")

    print("\n4. Rank-1 Impostor Analysis:")
    print(f"  Mean cos(pred, rank-1): {np.mean(cosines_rank1):.4f}")
    print(f"  Mean cos(pred, true): {np.mean(cosines_true):.4f}")
    print(f"  Gap: {np.mean(cosines_rank1) - np.mean(cosines_true):.4f}")

    # Save results
    results = {
        "model": str(args.model),
        "n_samples": int(len(pred_vecs)),
        "validation_cosine": {
            "mean": float(np.mean(val_cosines)),
            "median": float(np.median(val_cosines)),
            "min": float(np.min(val_cosines)),
            "max": float(np.max(val_cosines)),
        },
        "rank_of_truth": {
            "in_top_50_pct": float(100 * np.mean(in_top50)),
            "in_top_500_pct": float(100 * np.mean(in_top500)),
            "in_top_5000_pct": float(100 * np.mean(in_top5000)),
            "not_in_top_10000_pct": float(100 * np.mean(not_found)),
            "median_rank": int(np.median(valid_ranks)) if len(valid_ranks) > 0 else None,
            "percentiles": {
                str(p): int(np.percentile(valid_ranks, p))
                for p in [10, 25, 50, 75, 90, 95, 99]
            } if len(valid_ranks) > 0 else {},
        },
        "cosine_margins": {
            "mean": float(np.mean(margins)),
            "median": float(np.median(margins)),
            "positive_pct": float(100 * np.mean(margins > 0)),
            "gt_0.05_pct": float(100 * np.mean(margins > 0.05)),
        },
        "impostor_analysis": {
            "mean_cos_rank1": float(np.mean(cosines_rank1)),
            "mean_cos_true": float(np.mean(cosines_true)),
            "gap": float(np.mean(cosines_rank1) - np.mean(cosines_true)),
        },
    }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
