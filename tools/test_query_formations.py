#!/usr/bin/env python3
"""
Quick experiment: Test different query formation strategies.

Strategies to test:
1. Last vector (baseline from hybrid test): 4.55% Recall@5
2. Mean of all context
3. Mean of last 100 vectors
4. Exponentially weighted average (recent = higher weight)
5. LVM prediction (already computed in hybrid test)

This should take ~5-10 minutes total.
"""

import numpy as np
import faiss
from pathlib import Path
import json
from tqdm import tqdm

def load_data():
    """Load validation data and FAISS index."""
    print("Loading data...")
    
    # Validation data
    val_data = np.load('artifacts/lvm/data_phase3_tmd/validation_sequences_ctx100.npz', allow_pickle=True)
    context_seqs = val_data['context_sequences']
    target_indices = val_data['target_indices']
    
    # FAISS index
    index = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
    index.nprobe = 32  # Higher nprobe for better recall
    
    print(f"  Validation samples: {len(context_seqs):,}")
    print(f"  Context length: {context_seqs.shape[1]}")
    print(f"  FAISS vectors: {index.ntotal:,}")
    print(f"  nprobe: {index.nprobe}")
    
    return context_seqs, target_indices, index


def normalize(vectors):
    """L2 normalize vectors."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-8)


def query_last_vector(context):
    """Strategy 1: Use last vector as query (baseline)."""
    return context[-1:, :]


def query_mean_all(context):
    """Strategy 2: Mean of all context vectors."""
    return np.mean(context, axis=0, keepdims=True)


def query_mean_recent(context, n=100):
    """Strategy 3: Mean of last N vectors."""
    return np.mean(context[-n:], axis=0, keepdims=True)


def query_exponential_weighted(context, alpha=0.1):
    """
    Strategy 4: Exponentially weighted average.
    
    More recent vectors get higher weight: w_t = alpha * (1-alpha)^(T-t)
    """
    T = len(context)
    weights = np.array([alpha * (1 - alpha) ** (T - t - 1) for t in range(T)])
    weights = weights / weights.sum()  # Normalize
    
    weighted = np.sum(context * weights[:, np.newaxis], axis=0, keepdims=True)
    return weighted


def evaluate_query_strategy(context_seqs, target_indices, index, query_fn, name):
    """Evaluate a query formation strategy."""
    k_values = [1, 5, 10, 100, 500, 1000]
    recalls = {k: [] for k in k_values}
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    for context, target_idx in tqdm(zip(context_seqs, target_indices), total=len(context_seqs), desc=name):
        # Form query
        query = query_fn(context)
        
        # Normalize
        query_norm = normalize(query).astype(np.float32)
        
        # Search
        distances, indices = index.search(query_norm, max(k_values))
        
        # Compute recall@k
        for k in k_values:
            recalls[k].append(1.0 if target_idx in indices[0, :k] else 0.0)
    
    # Aggregate
    results = {f'recall@{k}': np.mean(recalls[k]) * 100 for k in k_values}
    
    print(f"\nResults:")
    for metric, value in results.items():
        print(f"  {metric:15s}: {value:6.2f}%")
    
    return results


def main():
    # Load data
    context_seqs, target_indices, index = load_data()
    
    # Define strategies
    strategies = [
        ("Last vector (baseline)", query_last_vector),
        ("Mean of all context", query_mean_all),
        ("Mean of last 100", lambda ctx: query_mean_recent(ctx, n=100)),
        ("Mean of last 200", lambda ctx: query_mean_recent(ctx, n=200)),
        ("Exp weighted (α=0.1)", lambda ctx: query_exponential_weighted(ctx, alpha=0.1)),
        ("Exp weighted (α=0.05)", lambda ctx: query_exponential_weighted(ctx, alpha=0.05)),
    ]
    
    # Run experiments
    all_results = {}
    for name, query_fn in strategies:
        results = evaluate_query_strategy(context_seqs, target_indices, index, query_fn, name)
        all_results[name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Query Formation Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<30} | {'R@5':>8} | {'R@100':>8} | {'R@500':>8} | {'R@1000':>8}")
    print(f"{'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    for name, results in all_results.items():
        print(f"{name:<30} | {results['recall@5']:>7.2f}% | {results['recall@100']:>7.2f}% | {results['recall@500']:>7.2f}% | {results['recall@1000']:>7.2f}%")
    
    # Find best
    best_strategy = max(all_results.items(), key=lambda x: x[1]['recall@500'])
    print(f"\n✓ Best strategy: {best_strategy[0]}")
    print(f"  Recall@500: {best_strategy[1]['recall@500']:.2f}%")
    
    # Save results
    output_path = Path('artifacts/evals/query_formation_results.json')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
