"""
Train/Val Split Parity Checker

Verify that training and validation distributions match.
This can diagnose train/val divergence like P7's failure
(positive train margin, negative val margin).

Checks:
1. Temporal bias (Δ): cos(c4, target) - cos(c0, target)
2. Forward signal: cos(c4, target) vs cos(c4, c3)
3. Coherence: similarity between adjacent context positions
4. Distribution statistics: mean, std, quantiles

Usage:
    python tools/check_split_parity.py \
        --train artifacts/lvm/arxiv_train_sequences.npz \
        --val artifacts/lvm/arxiv_val_sequences.npz \
        --n-samples 5000
"""

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path


def compute_temporal_metrics(contexts, targets, n_samples=5000):
    """
    Compute temporal bias and forward signal

    Args:
        contexts: (N, K, D) - context sequences
        targets: (N, D) - target next chunks
        n_samples: number of samples to analyze

    Returns:
        metrics: dictionary with temporal statistics
    """
    N = min(n_samples, len(contexts))

    cos_c0_target = []
    cos_c4_target = []
    cos_c4_c3 = []

    for i in range(N):
        c0 = contexts[i, 0]   # First context
        c3 = contexts[i, -2]  # Second-to-last context
        c4 = contexts[i, -1]  # Last context
        target = targets[i]

        # Temporal bias: does target correlate more with recent context?
        sim_c0 = cosine_similarity([c0], [target])[0][0]
        sim_c4 = cosine_similarity([c4], [target])[0][0]

        # Forward signal: is target more similar to c4 than c4 is to c3?
        sim_c4_c3 = cosine_similarity([c4], [c3])[0][0]

        cos_c0_target.append(sim_c0)
        cos_c4_target.append(sim_c4)
        cos_c4_c3.append(sim_c4_c3)

    # Convert to arrays
    cos_c0_target = np.array(cos_c0_target)
    cos_c4_target = np.array(cos_c4_target)
    cos_c4_c3 = np.array(cos_c4_c3)

    # Temporal bias
    delta = cos_c4_target - cos_c0_target

    # Forward signal: target is "forward" if it's less similar to c4 than c3 is
    # (because target is NEW, not just continuation)
    forward_signal = cos_c4_c3 - cos_c4_target

    metrics = {
        'cos_c0_target_mean': np.mean(cos_c0_target),
        'cos_c0_target_std': np.std(cos_c0_target),
        'cos_c4_target_mean': np.mean(cos_c4_target),
        'cos_c4_target_std': np.std(cos_c4_target),
        'delta_mean': np.mean(delta),
        'delta_std': np.std(delta),
        'delta_positive_pct': (delta > 0).mean() * 100,
        'forward_signal_mean': np.mean(forward_signal),
        'forward_signal_std': np.std(forward_signal),
        'forward_signal_positive_pct': (forward_signal > 0).mean() * 100,
        # Raw arrays for plotting
        'cos_c0_target': cos_c0_target,
        'cos_c4_target': cos_c4_target,
        'delta': delta,
        'forward_signal': forward_signal
    }

    return metrics


def compute_coherence_metrics(contexts, n_samples=5000):
    """
    Compute coherence between adjacent context positions

    Args:
        contexts: (N, K, D) - context sequences
        n_samples: number of samples to analyze

    Returns:
        coherence_matrix: (K-1,) - mean cosine similarity between adjacent positions
    """
    N = min(n_samples, len(contexts))
    K = contexts.shape[1]

    coherence = []

    for k in range(K - 1):
        sims = []
        for i in range(N):
            c_k = contexts[i, k]
            c_k1 = contexts[i, k+1]
            sim = cosine_similarity([c_k], [c_k1])[0][0]
            sims.append(sim)
        coherence.append(np.mean(sims))

    return np.array(coherence)


def compare_distributions(train_metrics, val_metrics, split_name_train="Train", split_name_val="Val"):
    """
    Compare train and val distributions

    Args:
        train_metrics: metrics from training set
        val_metrics: metrics from validation set
        split_name_train: name for train split
        split_name_val: name for val split

    Returns:
        comparison: dictionary with comparison results
    """
    print(f"\n{'='*70}")
    print("TRAIN/VAL DISTRIBUTION COMPARISON")
    print(f"{'='*70}\n")

    # Compare key metrics
    metrics_to_compare = [
        ('cos_c4_target_mean', 'cos(c4, target) mean', 0.02),
        ('delta_mean', 'Temporal bias (Δ)', 0.01),
        ('forward_signal_mean', 'Forward signal', 0.01),
        ('delta_positive_pct', '% sequences with Δ > 0', 5.0)
    ]

    mismatches = []

    for key, name, threshold in metrics_to_compare:
        train_val = train_metrics[key]
        val_val = val_metrics[key]
        diff = abs(train_val - val_val)

        status = "✅ MATCH" if diff < threshold else "❌ MISMATCH"

        print(f"{name}:")
        print(f"  {split_name_train}: {train_val:.4f}")
        print(f"  {split_name_val}:   {val_val:.4f}")
        print(f"  Diff:     {diff:.4f}  ({status}, threshold={threshold})")
        print()

        if diff >= threshold:
            mismatches.append((name, diff, threshold))

    # Overall verdict
    if len(mismatches) == 0:
        print(f"✅ {split_name_train}/{split_name_val} distributions MATCH")
        print("   Train/val split appears balanced.\n")
        return True
    else:
        print(f"❌ {split_name_train}/{split_name_val} distributions MISMATCH")
        print(f"   Found {len(mismatches)} significant differences:")
        for name, diff, threshold in mismatches:
            print(f"     - {name}: diff={diff:.4f} (threshold={threshold})")
        print()
        print("⚠️  This mismatch could explain train/val divergence!")
        print("   Consider re-splitting data with stratification.\n")
        return False


def plot_distributions(train_metrics, val_metrics, output_dir):
    """
    Plot distribution comparisons

    Args:
        train_metrics: metrics from training set
        val_metrics: metrics from validation set
        output_dir: directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Temporal bias (Δ) distributions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(train_metrics['delta'], bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(val_metrics['delta'], bins=50, alpha=0.5, label='Val', density=True)
    plt.axvline(train_metrics['delta_mean'], color='blue', linestyle='--', label=f"Train mean: {train_metrics['delta_mean']:.3f}")
    plt.axvline(val_metrics['delta_mean'], color='orange', linestyle='--', label=f"Val mean: {val_metrics['delta_mean']:.3f}")
    plt.xlabel('Temporal Bias (Δ)')
    plt.ylabel('Density')
    plt.title('Temporal Bias Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(train_metrics['forward_signal'], bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(val_metrics['forward_signal'], bins=50, alpha=0.5, label='Val', density=True)
    plt.axvline(train_metrics['forward_signal_mean'], color='blue', linestyle='--')
    plt.axvline(val_metrics['forward_signal_mean'], color='orange', linestyle='--')
    plt.xlabel('Forward Signal')
    plt.ylabel('Density')
    plt.title('Forward Signal Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'split_parity_distributions.png', dpi=150)
    print(f"Saved plot: {output_dir / 'split_parity_distributions.png'}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Check train/val split parity')
    parser.add_argument('--train', required=True, help='Training sequences NPZ')
    parser.add_argument('--val', required=True, help='Validation sequences NPZ')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples to analyze')
    parser.add_argument('--output-dir', default='artifacts/lvm/split_parity', help='Output directory for plots')

    args = parser.parse_args()

    print(f"Loading datasets...")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")

    # Load data
    train_data = np.load(args.train)
    val_data = np.load(args.val)

    train_contexts = train_data['contexts']
    train_targets = train_data['targets']
    val_contexts = val_data['contexts']
    val_targets = val_data['targets']

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_contexts)} sequences")
    print(f"  Val:   {len(val_contexts)} sequences")
    print(f"  Analyzing {args.n_samples} samples from each split")

    # Compute metrics
    print("\nComputing temporal metrics...")
    train_metrics = compute_temporal_metrics(train_contexts, train_targets, args.n_samples)
    val_metrics = compute_temporal_metrics(val_contexts, val_targets, args.n_samples)

    # Compare distributions
    match = compare_distributions(train_metrics, val_metrics)

    # Plot distributions
    print("Generating plots...")
    plot_distributions(train_metrics, val_metrics, args.output_dir)

    # Save results
    results_file = Path(args.output_dir) / 'split_parity_results.txt'
    with open(results_file, 'w') as f:
        f.write("TRAIN/VAL SPLIT PARITY CHECK\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Train: {args.train}\n")
        f.write(f"Val:   {args.val}\n")
        f.write(f"Samples analyzed: {args.n_samples}\n\n")

        f.write("TRAIN METRICS:\n")
        for key, val in train_metrics.items():
            if not isinstance(val, np.ndarray):
                f.write(f"  {key}: {val:.4f}\n")

        f.write("\nVAL METRICS:\n")
        for key, val in val_metrics.items():
            if not isinstance(val, np.ndarray):
                f.write(f"  {key}: {val:.4f}\n")

        f.write(f"\nDISTRIBUTIONS MATCH: {'YES' if match else 'NO'}\n")

    print(f"Saved results: {results_file}")

    # Exit code: 0 if match, 1 if mismatch
    return 0 if match else 1


if __name__ == '__main__':
    exit(main())
