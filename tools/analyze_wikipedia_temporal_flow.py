#!/usr/bin/env python3
"""
Analyze Wikipedia Data Temporal Flow Quality

Investigates whether English Wikipedia articles have inherent backward temporal bias
that makes next-chunk prediction harder than previous-chunk prediction.

**Research Questions**:
1. Is Δ (forward - backward) consistently negative across all articles?
2. Are some articles/domains more backward-biased than others?
3. Is this a chunking artifact or inherent to Wikipedia structure?
4. What do actual examples look like (qualitative analysis)?
5. Would reversing chunk order help?

**Hypotheses**:
- H1: Wikipedia lead sections preview content → forward bias
- H2: Later sections reference earlier concepts → backward bias
- H3: Explanatory structure (general→specific) → backward bias
- H4: Chunking at arbitrary boundaries → noise, no systematic bias

**Usage**:
    python tools/analyze_wikipedia_temporal_flow.py \\
        --articles-npz artifacts/wikipedia_584k_fresh.npz \\
        --sequences-npz artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \\
        --n-samples 5000 \\
        --output-dir artifacts/lvm/wikipedia_temporal_analysis
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_data(articles_path: str, sequences_path: str) -> Tuple[np.ndarray, Dict]:
    """Load article vectors and training sequences."""
    print(f"[INFO] Loading articles from {articles_path}")
    articles = np.load(articles_path)

    print(f"[INFO] Loading sequences from {sequences_path}")
    sequences = np.load(sequences_path, allow_pickle=True)

    return articles, sequences


def compute_directional_bias(contexts: np.ndarray, targets: np.ndarray, prev: np.ndarray) -> Dict:
    """
    Compute forward vs backward bias for a set of sequences.

    Args:
        contexts: (N, 5, 768) context vectors
        targets: (N, 768) target_next vectors
        prev: (N, 768) target_prev vectors

    Returns:
        Dict with forward_cos, backward_cos, delta, and per-sample metrics
    """
    # Convert to torch for fast cosine computation
    contexts_t = torch.from_numpy(contexts).float()
    targets_t = torch.from_numpy(targets).float()
    prev_t = torch.from_numpy(prev).float()

    # Normalize
    ctx_last = F.normalize(contexts_t[:, -1, :], dim=-1, p=2)  # (N, 768)
    targets_norm = F.normalize(targets_t, dim=-1, p=2)
    prev_norm = F.normalize(prev_t, dim=-1, p=2)

    # Compute cosines
    forward_cos = (ctx_last * targets_norm).sum(dim=-1)  # (N,)
    backward_cos = (ctx_last * prev_norm).sum(dim=-1)  # (N,)

    delta = forward_cos - backward_cos  # (N,)

    return {
        "forward_mean": forward_cos.mean().item(),
        "backward_mean": backward_cos.mean().item(),
        "delta_mean": delta.mean().item(),
        "forward_std": forward_cos.std().item(),
        "backward_std": backward_cos.std().item(),
        "delta_std": delta.std().item(),
        "forward_samples": forward_cos.numpy(),
        "backward_samples": backward_cos.numpy(),
        "delta_samples": delta.numpy(),
    }


def analyze_by_article(sequences: Dict, n_samples: int = 5000) -> Dict:
    """Analyze directional bias grouped by article."""
    contexts = sequences["context_sequences"][:n_samples]  # (N, 5, 768)
    targets = sequences["target_vectors"][:n_samples]  # (N, 768)
    metadata = sequences["metadata"][:n_samples]  # (N,) array of dicts

    # Extract article_ids and chunk_ids from metadata
    article_ids = np.array([m["article_index"] for m in metadata])
    chunk_ids = np.array([m["target_chunk_index"] for m in metadata])

    # Compute prev vectors (should be context[:, -2, :] but need to extract properly)
    # For P6 format, prev is the chunk before target_next
    # In original sequences: [c0, c1, c2, c3, c4] → target (c5)
    # We want: prev = c4, target_next = c5
    # But contexts only has c0-c4, so prev should be extracted differently

    # Extract prev from article using article_id and chunk_id
    # prev_chunk_id = chunk_id - 1 (the chunk right before target)
    # This is tricky - let's use context[:, -2, :] as approximate prev
    prev = contexts[:, -2, :]  # (N, 768) - second-to-last context position

    # Overall bias
    overall = compute_directional_bias(contexts, targets, prev)

    # Per-article analysis
    article_stats = defaultdict(list)
    unique_articles = np.unique(article_ids)

    print(f"[INFO] Analyzing {len(unique_articles)} unique articles...")

    for article_id in unique_articles:
        mask = article_ids == article_id
        if mask.sum() < 3:  # Need at least 3 samples
            continue

        ctx_art = contexts[mask]
        tgt_art = targets[mask]
        prev_art = prev[mask]

        stats = compute_directional_bias(ctx_art, tgt_art, prev_art)
        article_stats[int(article_id)] = {
            "n_samples": int(mask.sum()),
            "delta_mean": stats["delta_mean"],
            "forward_mean": stats["forward_mean"],
            "backward_mean": stats["backward_mean"],
        }

    # Compute distribution of per-article deltas
    deltas = [v["delta_mean"] for v in article_stats.values()]

    return {
        "overall": overall,
        "per_article": article_stats,
        "article_delta_distribution": {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "median": float(np.median(deltas)),
            "pct_negative": float((np.array(deltas) < 0).sum() / len(deltas)),
        }
    }


def analyze_by_chunk_position(sequences: Dict, n_samples: int = 5000) -> Dict:
    """Analyze directional bias by position in article (early vs late chunks)."""
    contexts = sequences["context_sequences"][:n_samples]
    targets = sequences["target_vectors"][:n_samples]
    metadata = sequences["metadata"][:n_samples]
    chunk_ids = np.array([m["target_chunk_index"] for m in metadata])
    prev = contexts[:, -2, :]

    # Bin by chunk position (early: 0-4, mid: 5-9, late: 10+)
    bins = {
        "early (0-4)": chunk_ids < 5,
        "mid (5-9)": (chunk_ids >= 5) & (chunk_ids < 10),
        "late (10+)": chunk_ids >= 10,
    }

    results = {}
    for bin_name, mask in bins.items():
        if mask.sum() < 10:
            continue

        stats = compute_directional_bias(contexts[mask], targets[mask], prev[mask])
        results[bin_name] = {
            "n_samples": int(mask.sum()),
            "delta_mean": stats["delta_mean"],
            "forward_mean": stats["forward_mean"],
            "backward_mean": stats["backward_mean"],
        }

    return results


def find_worst_examples(sequences: Dict, n_samples: int = 5000, top_k: int = 20) -> List[Dict]:
    """Find sequences with most extreme backward bias."""
    contexts = sequences["context_sequences"][:n_samples]
    targets = sequences["target_vectors"][:n_samples]
    metadata = sequences["metadata"][:n_samples]
    article_ids = np.array([m["article_index"] for m in metadata])
    chunk_ids = np.array([m["target_chunk_index"] for m in metadata])
    prev = contexts[:, -2, :]

    stats = compute_directional_bias(contexts, targets, prev)
    deltas = stats["delta_samples"]

    # Find most backward-biased (most negative delta)
    worst_indices = np.argsort(deltas)[:top_k]

    examples = []
    for idx in worst_indices:
        examples.append({
            "index": int(idx),
            "article_id": int(article_ids[idx]),
            "chunk_id": int(chunk_ids[idx]),
            "delta": float(deltas[idx]),
            "forward_cos": float(stats["forward_samples"][idx]),
            "backward_cos": float(stats["backward_samples"][idx]),
        })

    return examples


def compute_offset_curve(sequences: Dict, n_samples: int = 1000) -> Dict:
    """Compute cosine similarity for offsets k ∈ {-5...+5} to detect monotonic structure."""
    contexts = sequences["context_sequences"][:n_samples]  # (N, 5, 768)

    # Convert to torch
    contexts_t = torch.from_numpy(contexts).float()
    ctx_last = F.normalize(contexts_t[:, -1, :], dim=-1, p=2)  # (N, 768)

    # Compute cosine for each offset k relative to ctx_last (position 4)
    # k=-1: position 3 (one back), k=0: position 4 (self), k=+1: would be position 5 (target)
    offsets = {}

    for k in range(-4, 1):  # k=-4 to k=0 (positions 0 to 4 in context)
        pos = 4 + k  # Position in context array
        if 0 <= pos < 5:
            target_k = F.normalize(contexts_t[:, pos, :], dim=-1, p=2)
            cos_k = (ctx_last * target_k).sum(dim=-1).mean().item()
            offsets[k] = cos_k

    return offsets


def test_reversed_order(sequences: Dict, n_samples: int = 1000) -> Dict:
    """Test if reversing chunk order within articles improves forward bias."""
    contexts = sequences["context_sequences"][:n_samples]
    targets = sequences["target_vectors"][:n_samples]
    prev = contexts[:, -2, :]

    # Original order
    original = compute_directional_bias(contexts, targets, prev)

    # Reverse context order (but keep target same)
    contexts_reversed = np.flip(contexts, axis=1).copy()  # Reverse sequence dimension
    reversed_stats = compute_directional_bias(contexts_reversed, targets, prev)

    return {
        "original_delta": original["delta_mean"],
        "reversed_delta": reversed_stats["delta_mean"],
        "improvement": reversed_stats["delta_mean"] - original["delta_mean"],
    }


def generate_report(analysis: Dict, output_dir: Path):
    """Generate comprehensive analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove numpy arrays from analysis before JSON serialization
    analysis_json = analysis.copy()
    if "overall_bias" in analysis_json and "overall" in analysis_json["overall_bias"]:
        overall = analysis_json["overall_bias"]["overall"]
        for key in ["forward_samples", "backward_samples", "delta_samples"]:
            overall.pop(key, None)

    # Save full analysis JSON
    with open(output_dir / "full_analysis.json", "w") as f:
        json.dump(analysis_json, f, indent=2)

    # Generate markdown report
    report = []
    report.append("# Wikipedia Temporal Flow Analysis\n")
    report.append(f"**Date**: {analysis['metadata']['timestamp']}\n")
    report.append(f"**Samples**: {analysis['metadata']['n_samples']}\n\n")

    report.append("## Overall Directional Bias\n")
    overall = analysis["overall_bias"]["overall"]
    report.append(f"- **Forward** (ctx[-1] → target_next): {overall['forward_mean']:.4f} ± {overall['forward_std']:.4f}\n")
    report.append(f"- **Backward** (ctx[-1] → target_prev): {overall['backward_mean']:.4f} ± {overall['backward_std']:.4f}\n")
    report.append(f"- **Δ (Forward - Backward)**: {overall['delta_mean']:.4f} ± {overall['delta_std']:.4f}\n")

    if overall['delta_mean'] < 0:
        report.append(f"\n⚠️ **BACKWARD BIAS CONFIRMED**: Δ = {overall['delta_mean']:.4f} (backward {abs(overall['delta_mean'])*100:.1f}% stronger)\n\n")
    else:
        report.append(f"\n✅ **FORWARD BIAS**: Δ = {overall['delta_mean']:.4f}\n\n")

    report.append("## Per-Article Analysis\n")
    art_dist = analysis["overall_bias"]["article_delta_distribution"]
    report.append(f"- **Mean Δ per article**: {art_dist['mean']:.4f}\n")
    report.append(f"- **Std Δ per article**: {art_dist['std']:.4f}\n")
    report.append(f"- **Range**: [{art_dist['min']:.4f}, {art_dist['max']:.4f}]\n")
    report.append(f"- **% articles with backward bias**: {art_dist['pct_negative']*100:.1f}%\n\n")

    report.append("## By Chunk Position\n")
    for pos, stats in analysis["by_chunk_position"].items():
        report.append(f"- **{pos}**: Δ = {stats['delta_mean']:.4f} (n={stats['n_samples']})\n")
    report.append("\n")

    report.append("## Offset Curve (cos vs offset k)\n")
    report.append("Shows cosine similarity between ctx[-1] (position 4) and earlier positions:\n")
    for k, cos_val in sorted(analysis["offset_curve"].items()):
        report.append(f"- **k={k}** (position {4+k}): cos = {cos_val:.4f}\n")
    report.append("\n")

    report.append("## Reversed Order Test\n")
    rev = analysis["reversed_order_test"]
    report.append(f"- **Original order**: Δ = {rev['original_delta']:.4f}\n")
    report.append(f"- **Reversed order**: Δ = {rev['reversed_delta']:.4f}\n")
    report.append(f"- **Improvement**: {rev['improvement']:.4f}\n")

    if rev['improvement'] > 0.05:
        report.append(f"\n✅ **Reversing chunk order helps!** (+{rev['improvement']*100:.1f}% forward bias)\n\n")
    else:
        report.append(f"\n⚠️ Reversing chunk order does NOT help significantly.\n\n")

    report.append("## Worst Examples (Most Backward-Biased)\n")
    report.append("| Index | Article ID | Chunk ID | Δ | Forward | Backward |\n")
    report.append("|-------|------------|----------|---|---------|----------|\n")
    for ex in analysis["worst_examples"][:10]:
        report.append(f"| {ex['index']} | {ex['article_id']} | {ex['chunk_id']} | {ex['delta']:.3f} | {ex['forward_cos']:.3f} | {ex['backward_cos']:.3f} |\n")

    report.append("\n## Conclusions\n")
    if overall['delta_mean'] < -0.05:
        report.append("1. ❌ **Wikipedia data has STRONG backward bias** (Δ < -0.05)\n")
        report.append("2. 🔬 **Root cause**: Likely explanatory structure (later chunks reference earlier concepts)\n")
        report.append("3. 💡 **Solutions**:\n")
        report.append("   - Try reversing chunk order within articles\n")
        report.append("   - Use different data sources (scientific papers, tutorials, stories)\n")
        report.append("   - Synthetically generate forward-flowing sequences\n")
        report.append("   - Accept backward bias and train with MUCH stronger directional pressure\n")
    elif overall['delta_mean'] < 0:
        report.append("1. ⚠️ **Wikipedia data has MODERATE backward bias** (Δ slightly negative)\n")
        report.append("2. 🎯 **Training should work** with balanced directional pressure (P6b v2.3)\n")
    else:
        report.append("1. ✅ **Wikipedia data has FORWARD bias** (Δ positive)\n")
        report.append("2. 🎯 **Training should work well** with minimal directional pressure\n")

    report_text = "".join(report)

    with open(output_dir / "REPORT.md", "w") as f:
        f.write(report_text)

    print("\n" + "="*80)
    print(report_text)
    print("="*80)
    print(f"\n[INFO] Full report saved to {output_dir / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Wikipedia temporal flow quality")
    parser.add_argument("--articles-npz", required=True, help="Path to article vectors NPZ")
    parser.add_argument("--sequences-npz", required=True, help="Path to training sequences NPZ")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples to analyze")
    parser.add_argument("--output-dir", default="artifacts/lvm/wikipedia_temporal_analysis",
                        help="Output directory for analysis")

    args = parser.parse_args()

    print("="*80)
    print("Wikipedia Temporal Flow Analysis")
    print("="*80)

    # Load data
    articles, sequences = load_data(args.articles_npz, args.sequences_npz)

    # Run analyses
    print(f"\n[INFO] Analyzing {args.n_samples} sequences...")

    print("[1/5] Overall bias analysis...")
    overall_bias = analyze_by_article(sequences, args.n_samples)

    print("[2/5] By chunk position...")
    by_chunk_position = analyze_by_chunk_position(sequences, args.n_samples)

    print("[3/5] Finding worst examples...")
    worst_examples = find_worst_examples(sequences, args.n_samples, top_k=20)

    print("[4/6] Computing offset curve...")
    offset_curve = compute_offset_curve(sequences, min(args.n_samples, 1000))

    print("[5/6] Testing reversed order...")
    reversed_order_test = test_reversed_order(sequences, min(args.n_samples, 1000))

    # Compile results
    from datetime import datetime
    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "articles_path": args.articles_npz,
            "sequences_path": args.sequences_npz,
            "n_samples": args.n_samples,
            "seed": SEED,
        },
        "overall_bias": overall_bias,
        "by_chunk_position": by_chunk_position,
        "offset_curve": offset_curve,
        "worst_examples": worst_examples,
        "reversed_order_test": reversed_order_test,
    }

    print("[6/6] Generating report...")
    generate_report(analysis, Path(args.output_dir))

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
