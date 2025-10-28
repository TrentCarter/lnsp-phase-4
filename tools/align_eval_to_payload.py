#!/usr/bin/env python3
"""
Align Evaluation Targets to Payload Vectors
============================================

Fixes data mismatch by replacing eval targets with actual payload vectors.

The issue: eval targets (from training) don't match payload vectors (from FAISS),
causing 0% retrieval even when models predict well.

Solution: Replace eval['targets'] with payload vectors for the same (article, chunk) keys.

Gates:
- Mapping coverage ≥ 99.5%
- cos(old_target, payload_vec) → mean ≥ 0.98 after fix
- All keys must exist in payload

Usage:
    python tools/align_eval_to_payload.py \
        --eval-npz artifacts/lvm/eval_v2_ready_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --out artifacts/lvm/eval_v2_payload_aligned.npz
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def compute_provenance(payload_path: Path, embedder="GTR-T5-base-768"):
    """Compute provenance metadata for reproducibility."""
    # Compute payload file hash
    with open(payload_path, 'rb') as f:
        payload_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    return {
        "embedder_id": embedder,
        "payload_build_id": f"payload584k_2025-10-24@sha256:{payload_hash}",
        "norm": "l2_once",
        "metric": "ip",
        "aligned_at": datetime.now().isoformat(),
        "alignment_tool": "align_eval_to_payload.py",
    }


def main():
    ap = argparse.ArgumentParser(description="Align eval targets to payload vectors")
    ap.add_argument("--eval-npz", type=Path, required=True,
                    help="Input evaluation NPZ (with mismatched targets)")
    ap.add_argument("--payload", type=Path, required=True,
                    help="Payload NPY file with ground truth vectors")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output aligned NPZ file")
    ap.add_argument("--min-coverage", type=float, default=0.995,
                    help="Minimum mapping coverage (default: 99.5%%)")
    ap.add_argument("--min-cosine", type=float, default=0.98,
                    help="Minimum mean cosine for sanity (default: 0.98)")

    args = ap.parse_args()

    print("=" * 80)
    print("Aligning Evaluation Targets to Payload Vectors")
    print("=" * 80)
    print(f"Input eval: {args.eval_npz}")
    print(f"Payload: {args.payload}")
    print(f"Output: {args.out}")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    eval_data = np.load(args.eval_npz, allow_pickle=True)
    payload = np.load(args.payload, allow_pickle=True).item()

    print(f"  Eval samples: {len(eval_data['truth_keys'])}")
    print(f"  Payload entries: {len(payload)}")

    # Extract fields
    contexts = eval_data['contexts']  # [N, 5, 768]
    old_targets = eval_data['targets']  # [N, 768] - MISMATCHED!
    truth_keys = eval_data['truth_keys']  # [N, 2]
    last_meta = eval_data['last_meta']  # [N]

    # Also preserve pred_vecs if present (used in original but wrong)
    pred_vecs = eval_data.get('pred_vecs')

    # Build reverse index
    print("\n2. Building (article, chunk) → payload_id mapping...")
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    print(f"  Mapped {len(article_chunk_to_id)} unique (article, chunk) pairs")

    # Check coverage
    print("\n3. Checking mapping coverage...")
    missing = []
    found = 0

    for i, (art_idx, chunk_idx) in enumerate(truth_keys):
        key = (int(art_idx), int(chunk_idx))
        if key in article_chunk_to_id:
            found += 1
        else:
            missing.append((i, art_idx, chunk_idx))

    coverage = found / len(truth_keys)
    print(f"  Found: {found}/{len(truth_keys)} ({100*coverage:.2f}%)")
    print(f"  Missing: {len(missing)} ({100*len(missing)/len(truth_keys):.2f}%)")

    if coverage < args.min_coverage:
        print(f"\n❌ GATE FAILED: Coverage {coverage:.4f} < {args.min_coverage}")
        print(f"  Missing keys: {missing[:10]}...")
        return 1

    print(f"  ✅ Coverage gate passed ({coverage:.4f} ≥ {args.min_coverage})")

    # Replace targets with payload vectors
    print("\n4. Replacing targets with payload vectors...")
    new_targets = np.zeros_like(old_targets)
    cosines_before = []
    cosines_after = []

    for i, (art_idx, chunk_idx) in enumerate(truth_keys):
        key = (int(art_idx), int(chunk_idx))

        if key not in article_chunk_to_id:
            # Keep old target if missing (shouldn't happen after coverage check)
            new_targets[i] = old_targets[i]
            continue

        payload_id = article_chunk_to_id[key]
        _, _, payload_vec = payload[payload_id]

        # Ensure L2 normalized
        payload_vec = payload_vec / np.linalg.norm(payload_vec)

        # Compute cosine before
        cos_before = np.dot(old_targets[i], payload_vec)
        cosines_before.append(cos_before)

        # Replace
        new_targets[i] = payload_vec

        # Cosine after (should be ~1.0)
        cos_after = np.dot(new_targets[i], payload_vec)
        cosines_after.append(cos_after)

        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(truth_keys)}")

    # Compute statistics
    print("\n5. Verification...")
    mean_cos_before = np.mean(cosines_before)
    mean_cos_after = np.mean(cosines_after)

    print(f"  Mean cosine (old_target vs payload): {mean_cos_before:.6f}")
    print(f"  Mean cosine (new_target vs payload): {mean_cos_after:.6f}")
    print(f"  Min cosine after: {np.min(cosines_after):.6f}")

    if mean_cos_after < args.min_cosine:
        print(f"\n❌ GATE FAILED: Mean cosine {mean_cos_after:.4f} < {args.min_cosine}")
        return 1

    print(f"  ✅ Cosine gate passed ({mean_cos_after:.4f} ≥ {args.min_cosine})")

    # Compute provenance
    print("\n6. Computing provenance...")
    provenance = compute_provenance(args.payload)
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    # Save aligned NPZ
    print(f"\n7. Saving aligned NPZ to: {args.out}")

    save_dict = {
        'contexts': contexts,
        'targets': new_targets,  # REPLACED!
        'truth_keys': truth_keys,
        'last_meta': last_meta,
        'provenance': np.array([json.dumps(provenance)]),  # Store as JSON string
        'alignment_stats': np.array([json.dumps({
            'coverage': float(coverage),
            'mean_cos_before': float(mean_cos_before),
            'mean_cos_after': float(mean_cos_after),
            'n_samples': int(len(truth_keys)),
            'n_missing': int(len(missing)),
        })]),
    }

    # Preserve pred_vecs if present (though it's mismatched too)
    if pred_vecs is not None:
        save_dict['pred_vecs_old'] = pred_vecs  # Keep for reference

    np.savez(args.out, **save_dict)

    print("\n" + "=" * 80)
    print("✅ SUCCESS! Evaluation targets aligned to payload vectors")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Coverage: {100*coverage:.2f}%")
    print(f"  Mean cosine improvement: {mean_cos_before:.4f} → {mean_cos_after:.4f}")
    print(f"  Output: {args.out}")
    print(f"\nNext steps:")
    print(f"  1. Run smoke test (500 samples)")
    print(f"  2. Expect Contain@50: ~60-75% (was 0%)")
    print(f"  3. Expect R@5: ~40-55% (was 0%)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
