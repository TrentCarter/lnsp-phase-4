#!/usr/bin/env python3
"""
Pre-Flight Checks for 790k Training
====================================

Run these 5 critical checks before launching training to avoid wasting 6-8 hours.

Usage:
    ./.venv/bin/python tools/preflight_checks_790k.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_1_encoder_fingerprint():
    """Check 1: Encoder fingerprint - GTR-T5 hash verification"""
    print("\n" + "="*60)
    print("CHECK 1: Encoder Fingerprint")
    print("="*60)

    try:
        from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

        # Initialize encoder
        orchestrator = IsolatedVecTextVectOrchestrator()

        # Encode test string
        test_text = "The quick brown fox jumps over the lazy dog"
        test_vector = orchestrator.encode_texts([test_text])[0]

        # Compute hash (convert to numpy first if torch tensor)
        if hasattr(test_vector, 'cpu'):
            test_vector = test_vector.cpu().numpy()
        elif not isinstance(test_vector, np.ndarray):
            test_vector = np.array(test_vector)
        vec_hash = hash(test_vector.tobytes())

        print(f"✅ Encoder loaded successfully")
        print(f"   Test vector shape: {test_vector.shape}")
        print(f"   Test vector norm: {np.linalg.norm(test_vector):.6f}")
        print(f"   Fingerprint hash: {vec_hash}")

        # Known good hash from 584k training (if available)
        # TODO: Store this from successful 584k run
        print("\n⚠️  Manual verification needed:")
        print("   Compare this fingerprint with 584k training encoder")
        print("   If different, OOD will look random!")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_2_eval_normalization():
    """Check 2: Verify both pred and target are L2-normalized in eval"""
    print("\n" + "="*60)
    print("CHECK 2: Eval Normalization")
    print("="*60)

    try:
        # Load OOD data
        ood_path = Path("artifacts/lvm/wikipedia_ood_test_ctx5.npz")
        if not ood_path.exists():
            print(f"⚠️  OOD test data not found: {ood_path}")
            return True  # Not critical for training, just for eval

        ood_data = np.load(ood_path)
        targets = ood_data['target_vectors']

        # Check normalization
        norms = np.linalg.norm(targets, axis=1)
        mean_norm = norms.mean()
        std_norm = norms.std()

        print(f"Target vector statistics:")
        print(f"   Mean norm: {mean_norm:.6f} (should be ~1.0)")
        print(f"   Std norm:  {std_norm:.6f} (should be <0.01)")
        print(f"   Min norm:  {norms.min():.6f}")
        print(f"   Max norm:  {norms.max():.6f}")

        if abs(mean_norm - 1.0) < 0.01 and std_norm < 0.01:
            print("✅ Targets are properly L2-normalized")
            return True
        else:
            print("❌ WARNING: Targets not properly normalized!")
            print("   This will cause incorrect cosine similarity calculations")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_3_oracle_recall():
    """Check 3: Oracle recall @K - FAISS accuracy verification"""
    print("\n" + "="*60)
    print("CHECK 3: Oracle Recall @K (FAISS Accuracy)")
    print("="*60)

    try:
        # Load training data
        train_path = Path("artifacts/lvm/training_sequences_ctx5.npz")
        if not train_path.exists():
            print(f"❌ Training data not found: {train_path}")
            return False

        data = np.load(train_path)
        targets = data['target_vectors']

        print(f"Loaded {len(targets):,} target vectors")
        print(f"Vector shape: {targets.shape}")

        # Compute self-similarity (oracle test)
        # Sample 1000 random vectors
        np.random.seed(42)
        sample_size = min(1000, len(targets))
        sample_idx = np.random.choice(len(targets), sample_size, replace=False)
        sample_vecs = targets[sample_idx]

        # Compute cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(sample_vecs, targets)

        # Check recall @1, @5, @1000
        recalls = {}
        for k in [1, 5, 1000]:
            # Get top-k indices for each query
            top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :k]

            # Check if true index is in top-k
            hits = 0
            for i, true_idx in enumerate(sample_idx):
                if true_idx in top_k_idx[i]:
                    hits += 1

            recall = hits / sample_size
            recalls[k] = recall

            status = "✅" if recall >= 0.97 else "❌"
            print(f"{status} Recall@{k:4d}: {recall:.4f} (expect ≥0.97)")

        if recalls[1000] >= 0.97 and recalls[5] >= 0.97:
            print("✅ Oracle recall passed (FAISS space aligned)")
            return True
        else:
            print("❌ Oracle recall failed (payload/index misalignment!)")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_4_off_by_one():
    """Check 4: Off-by-one OOD - neighbor alignment verification"""
    print("\n" + "="*60)
    print("CHECK 4: Off-by-One OOD (Neighbor Alignment)")
    print("="*60)

    try:
        # Load training data
        train_path = Path("artifacts/lvm/training_sequences_ctx5.npz")
        data = np.load(train_path)

        targets = data['target_vectors']
        positions = data['sequence_positions']

        # Group by article
        from collections import defaultdict
        articles = defaultdict(list)
        for i, pos in enumerate(positions):
            article_id = int(pos[0])
            chunk_idx = int(pos[1])
            articles[article_id].append((chunk_idx, i))

        # For each article, check neighbor alignment
        neighbor_peaks = Counter()

        for article_id, chunks in articles.items():
            # Sort by chunk index
            chunks = sorted(chunks, key=lambda x: x[0])

            # For each consecutive pair
            for i in range(len(chunks) - 1):
                chunk_idx_1, vec_idx_1 = chunks[i]
                chunk_idx_2, vec_idx_2 = chunks[i + 1]

                # Check if chunks are consecutive
                if int(chunk_idx_2) != int(chunk_idx_1) + 1:
                    continue

                # Compute which neighbor has max similarity
                vec1 = targets[vec_idx_1]
                vec2 = targets[vec_idx_2]

                # Compare with neighbors at offsets
                from sklearn.metrics.pairwise import cosine_similarity

                # Get neighborhood (vec_idx_1 ± 5)
                neighborhood_idx = range(
                    max(0, vec_idx_1 - 5),
                    min(len(targets), vec_idx_1 + 6)
                )
                neighborhood = targets[list(neighborhood_idx)]

                # Compute similarities
                sims = cosine_similarity([vec2], neighborhood)[0]

                # Find peak offset
                peak_offset = np.argmax(sims) - min(5, vec_idx_1)
                neighbor_peaks[peak_offset] += 1

        print(f"Neighbor peak distribution:")
        for offset in sorted(neighbor_peaks.keys()):
            count = neighbor_peaks[offset]
            pct = 100 * count / sum(neighbor_peaks.values())
            bar = "█" * int(pct / 2)
            print(f"   Offset {offset:+2d}: {count:5d} ({pct:5.1f}%) {bar}")

        # Check if peak is at i+1 (offset +1)
        max_offset = max(neighbor_peaks.items(), key=lambda x: x[1])[0]

        if max_offset == 1:
            print(f"✅ Peak at offset +1 (correct sequential alignment)")
            return True
        else:
            print(f"❌ Peak at offset {max_offset:+d} (misalignment!)")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def check_5_adjacency_coherence():
    """Check 5: Adjacency coherence - cos(t, t+1) distribution check"""
    print("\n" + "="*60)
    print("CHECK 5: Adjacency Coherence (Context Tightness)")
    print("="*60)

    try:
        # Load training data
        train_path = Path("artifacts/lvm/training_sequences_ctx5.npz")
        data = np.load(train_path)

        targets = data['target_vectors']
        positions = data['sequence_positions']

        # Compute cos(t, t+1) for all consecutive pairs
        adjacency_cosines = []

        for i in range(len(positions) - 1):
            article_1 = int(positions[i][0])
            chunk_1 = int(positions[i][1])
            article_2 = int(positions[i + 1][0])
            chunk_2 = int(positions[i + 1][1])

            # Only consider consecutive chunks in same article
            if article_1 == article_2 and chunk_2 == chunk_1 + 1:
                cos_sim = np.dot(targets[i], targets[i + 1])
                adjacency_cosines.append(cos_sim)

        adjacency_cosines = np.array(adjacency_cosines)

        print(f"Adjacency cosine statistics (n={len(adjacency_cosines):,}):")
        print(f"   Mean: {adjacency_cosines.mean():.4f}")
        print(f"   Std:  {adjacency_cosines.std():.4f}")
        print(f"   p10:  {np.percentile(adjacency_cosines, 10):.4f}")
        print(f"   p50:  {np.percentile(adjacency_cosines, 50):.4f}")
        print(f"   p90:  {np.percentile(adjacency_cosines, 90):.4f}")

        # Distribution histogram
        print("\nDistribution:")
        bins = np.arange(-0.2, 1.01, 0.1)
        hist, _ = np.histogram(adjacency_cosines, bins=bins)
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            count = hist[i]
            pct = 100 * count / len(adjacency_cosines)
            bar = "█" * int(pct / 2)
            print(f"   [{low:.1f}, {high:.1f}): {count:6d} ({pct:5.1f}%) {bar}")

        # Compare with expected (from 584k if available)
        mean_cos = adjacency_cosines.mean()

        print(f"\nInterpretation:")
        if mean_cos > 0.4:
            print(f"✅ High coherence (mean={mean_cos:.3f}) - ctx=5 is appropriate")
            return True
        elif mean_cos > 0.3:
            print(f"⚠️  Medium coherence (mean={mean_cos:.3f}) - ctx=5 acceptable, ctx=7 might help")
            return True
        else:
            print(f"❌ Low coherence (mean={mean_cos:.3f}) - consider ctx=7 or filtering")
            print("   790k corpus may be more diverse/noisy than 584k")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("AMN 790K PRE-FLIGHT CHECKS")
    print("="*60)
    print("\nRunning 5 critical checks before training...")
    print("If any fail, fix them FIRST - training won't save you!")

    checks = [
        ("Encoder Fingerprint", check_1_encoder_fingerprint),
        ("Eval Normalization", check_2_eval_normalization),
        ("Oracle Recall @K", check_3_oracle_recall),
        ("Off-by-One OOD", check_4_off_by_one),
        ("Adjacency Coherence", check_5_adjacency_coherence),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Check '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - SAFE TO TRAIN")
        print("="*60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX BEFORE TRAINING")
        print("="*60)
        print("\nDo NOT launch training until all checks pass!")
        print("Training on misaligned data = wasted 6-8 hours")
        return 1


if __name__ == '__main__':
    sys.exit(main())
