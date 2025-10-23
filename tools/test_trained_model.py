#!/usr/bin/env python3
"""
Test the trained Two-Tower model with dual-path decoder.

Usage:
    python tools/test_trained_model.py --checkpoint runs/stable_sync_*/epoch_005.pt
"""

import sys
import argparse
import numpy as np
import torch
import faiss
from pathlib import Path
from collections import Counter

sys.path.insert(0, 'src')
from retrieval.query_tower import QueryTower
from retrieval.miner_sync import SyncFaissMiner
from training.dual_path_decoder import DualPathDecoder


def load_checkpoint(ckpt_path):
    """Load trained checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    return checkpoint


def test_retrieval_quality(query_tower, miner, bank, n_tests=10):
    """Test retrieval quality with trained query tower."""
    print("\n" + "=" * 60)
    print("TEST 1: Retrieval Quality")
    print("=" * 60)

    query_tower.eval()
    similarities = []

    for i in range(n_tests):
        # Random context window
        start = np.random.randint(0, len(bank) - 10)
        ctx = bank[start:start+10]  # (10, 768)

        # Encode with trained query tower
        ctx_torch = torch.from_numpy(ctx).float().unsqueeze(0)
        with torch.no_grad():
            q = query_tower(ctx_torch)
        q_np = q.cpu().numpy()[0]

        # Retrieve top-10
        I, D = miner.search(q_np[None, :], k=10)

        # Average similarity of top-10
        avg_sim = float(D[0].mean())
        similarities.append(avg_sim)

        if i < 3:  # Show first 3 examples
            print(f"\nQuery {i+1}:")
            print(f"  Context: vectors {start} to {start+10}")
            print(f"  Top-10 avg similarity: {avg_sim:.4f}")
            print(f"  Best match: index {I[0,0]} (sim={D[0,0]:.4f})")

    print(f"\n📊 Overall Statistics:")
    print(f"  Mean similarity: {np.mean(similarities):.4f}")
    print(f"  Std similarity: {np.std(similarities):.4f}")
    print(f"  Min similarity: {np.min(similarities):.4f}")
    print(f"  Max similarity: {np.max(similarities):.4f}")


def test_dual_path_generation(query_tower, miner, bank, decoder, n_steps=20):
    """Test generation with dual-path decoder."""
    print("\n" + "=" * 60)
    print("TEST 2: Dual-Path Generation")
    print("=" * 60)

    query_tower.eval()

    # Random starting context
    start = np.random.randint(0, len(bank) - 10)
    context = bank[start:start+10].copy()

    print(f"\nStarting context: vectors {start} to {start+10}")
    print(f"Generating {n_steps} steps with {decoder.cfg.lane_name} lane...")
    print(f"  tau_snap={decoder.cfg.tau_snap:.2f}, tau_novel={decoder.cfg.tau_novel:.2f}")
    print()

    decisions = []
    cosines = []
    alphas = []

    for step in range(n_steps):
        # Encode context with query tower
        ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
        with torch.no_grad():
            q = query_tower(ctx_torch)
        q_np = q.cpu().numpy()[0]

        # Retrieve candidates
        I, D = miner.search(q_np[None, :], k=20)
        neighbors = [
            (f"bank_{i}", bank[i], D[0, j])
            for j, i in enumerate(I[0])
        ]

        # Mock LVM prediction (use mean of context for simplicity)
        v_hat = context[-3:].mean(axis=0)  # Last 3 vectors
        v_hat = v_hat / np.linalg.norm(v_hat)

        # Dual-path decision
        v_out, rec = decoder.step(v_hat, neighbors)

        # Update context
        context = np.vstack([context, v_out])

        # Record telemetry
        decisions.append(rec.decision)
        cosines.append(rec.c_max)
        if rec.alpha is not None:
            alphas.append(rec.alpha)

        # Print progress
        if (step + 1) % 5 == 0 or step < 3:
            alpha_str = f", α={rec.alpha:.2f}" if rec.alpha else ""
            print(f"  Step {step+1:2d}: {rec.decision:15s} (c_max={rec.c_max:.3f}{alpha_str})")

    # Analyze results
    print(f"\n📊 Generation Statistics:")
    dist = Counter(decisions)
    for decision, count in dist.most_common():
        pct = 100 * count / n_steps
        print(f"  {decision:15s}: {count:2d} ({pct:5.1f}%)")

    print(f"\n📈 Cosine Statistics:")
    print(f"  Mean: {np.mean(cosines):.4f}")
    print(f"  Range: [{np.min(cosines):.4f}, {np.max(cosines):.4f}]")

    if alphas:
        print(f"\n🎚️  Alpha Statistics (BLEND):")
        print(f"  Mean: {np.mean(alphas):.4f}")
        print(f"  Range: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")


def test_lane_comparison(query_tower, miner, bank):
    """Compare different lane configurations."""
    print("\n" + "=" * 60)
    print("TEST 3: Lane Comparison")
    print("=" * 60)

    lanes = {
        'conservative': (0.94, 0.88),
        'neutral': (0.92, 0.85),
        'creative': (0.90, 0.82),
    }

    results = {}
    n_steps = 30

    for lane_name, (tau_snap, tau_novel) in lanes.items():
        decoder = DualPathDecoder(
            lane=lane_name,
            tau_snap=tau_snap,
            tau_novel=tau_novel
        )

        # Random starting context
        start = np.random.randint(0, len(bank) - 10)
        context = bank[start:start+10].copy()

        decisions = []

        for step in range(n_steps):
            ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
            with torch.no_grad():
                q = query_tower(ctx_torch)
            q_np = q.cpu().numpy()[0]

            I, D = miner.search(q_np[None, :], k=20)
            neighbors = [(f"bank_{i}", bank[i], D[0, j]) for j, i in enumerate(I[0])]

            v_hat = context[-3:].mean(axis=0)
            v_hat = v_hat / np.linalg.norm(v_hat)

            v_out, rec = decoder.step(v_hat, neighbors)
            context = np.vstack([context, v_out])
            decisions.append(rec.decision)

        results[lane_name] = Counter(decisions)

    print(f"\nDecision distribution by lane ({n_steps} steps each):\n")
    print(f"{'Lane':<15} {'SNAP':<8} {'BLEND':<8} {'NOVEL':<8} {'NOVEL_DUP':<12}")
    print("-" * 60)

    for lane_name, dist in results.items():
        snap_pct = 100 * dist['SNAP'] / n_steps
        blend_pct = 100 * dist['BLEND'] / n_steps
        novel_pct = 100 * dist['NOVEL'] / n_steps
        dup_pct = 100 * dist['NOVEL_DUP_DROP'] / n_steps

        print(f"{lane_name:<15} {snap_pct:5.1f}%   {blend_pct:5.1f}%   {novel_pct:5.1f}%   {dup_pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test trained Two-Tower model")
    parser.add_argument('--checkpoint', default='runs/stable_sync_20251022_115526/epoch_005.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--lane', default='neutral', choices=['conservative', 'neutral', 'creative'],
                        help='Lane configuration for dual-path decoder')
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINED TWO-TOWER MODEL TEST")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz', allow_pickle=True)
    bank = data['vectors']
    print(f"  Bank: {bank.shape}")

    index = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
    print(f"  FAISS index: {index.ntotal} vectors")

    # Load trained model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    # Initialize query tower with trained weights
    query_tower = QueryTower()
    query_tower.load_state_dict(checkpoint['model_q'])
    query_tower.eval()
    print("✓ Query tower loaded with trained weights")

    # Initialize miner
    miner = SyncFaissMiner(index, nprobe=8)
    print("✓ FAISS miner initialized")

    # Initialize decoder
    lane_configs = {
        'conservative': (0.94, 0.88),
        'neutral': (0.92, 0.85),
        'creative': (0.90, 0.82),
    }
    tau_snap, tau_novel = lane_configs[args.lane]
    decoder = DualPathDecoder(
        lane=args.lane,
        tau_snap=tau_snap,
        tau_novel=tau_novel
    )
    print(f"✓ Dual-path decoder initialized ({args.lane} lane)")

    # Run tests
    test_retrieval_quality(query_tower, miner, bank, n_tests=10)
    test_dual_path_generation(query_tower, miner, bank, decoder, n_steps=20)
    test_lane_comparison(query_tower, miner, bank)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
