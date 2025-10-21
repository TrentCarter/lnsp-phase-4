#!/usr/bin/env python3
"""
v4 Pre-flight Checklist

Runs all safety checks before launching v4 training:
1. Data sanity (split verification)
2. Leak check (no train/valid overlap)
3. Doc-space alignment check
4. Hard-neg filter confirmation
5. Curriculum schedule verification
"""

import argparse
import json
import numpy as np
import hashlib
from pathlib import Path


def check_data_sanity(pairs_path):
    """Verify train/valid split and counts"""
    print("\n=== 1. Data Sanity Check ===")

    data = np.load(pairs_path)

    counts = {
        'X_train': data['X_train'].shape[0],
        'Y_train': data['Y_train'].shape[0],
        'X_val': data['X_val'].shape[0],
        'Y_val': data['Y_val'].shape[0]
    }

    print("  Counts:")
    print(json.dumps(counts, indent=4))

    # Verify X and Y match
    assert counts['X_train'] == counts['Y_train'], "❌ Train X/Y mismatch!"
    assert counts['X_val'] == counts['Y_val'], "❌ Val X/Y mismatch!"

    # Check shapes
    assert data['X_train'].shape[1:] == (100, 768), f"❌ Train X shape wrong: {data['X_train'].shape}"
    assert data['Y_train'].shape[1:] == (768,), f"❌ Train Y shape wrong: {data['Y_train'].shape}"

    print(f"  ✓ Train: {counts['X_train']:,} pairs")
    print(f"  ✓ Val: {counts['X_val']:,} pairs")
    print(f"  ✓ Shapes correct: X=(N,100,768), Y=(N,768)")

    return counts


def check_leakage(pairs_path):
    """Check for exact duplicates between train and valid"""
    print("\n=== 2. Leakage Check ===")

    data = np.load(pairs_path)

    def hash_array(arr):
        return hashlib.sha1(np.ascontiguousarray(arr)).hexdigest()

    # Check context leakage
    print("  Checking context (X) leakage...")
    train_ctx = {hash_array(x) for x in data['X_train']}
    val_ctx = {hash_array(x) for x in data['X_val']}
    ctx_leaks = train_ctx & val_ctx

    print(f"    Train contexts: {len(train_ctx):,}")
    print(f"    Val contexts: {len(val_ctx):,}")
    print(f"    Leaks: {len(ctx_leaks)}")

    # Check target leakage
    print("  Checking target (Y) leakage...")
    train_tgt = {hash_array(y) for y in data['Y_train']}
    val_tgt = {hash_array(y) for y in data['Y_val']}
    tgt_leaks = train_tgt & val_tgt

    print(f"    Train targets: {len(train_tgt):,}")
    print(f"    Val targets: {len(val_tgt):,}")
    print(f"    Leaks: {len(tgt_leaks)}")

    if len(ctx_leaks) == 0 and len(tgt_leaks) == 0:
        print("  ✓ No leakage detected")
    else:
        print(f"  ⚠️  WARNING: {len(ctx_leaks)} context leaks, {len(tgt_leaks)} target leaks")
        print("    This may cause overfitting. Consider regenerating with better dedup.")

    return {'ctx_leaks': len(ctx_leaks), 'tgt_leaks': len(tgt_leaks)}


def check_doc_alignment(bank_path, sample_size=10000):
    """Check doc-space alignment (mean cosine)"""
    print("\n=== 3. Doc-Space Alignment Check ===")

    data = np.load(bank_path)
    vectors = data['vectors']

    # Random sample
    n = min(sample_size, len(vectors))
    idxs = np.random.choice(len(vectors), size=n, replace=False)
    sample = vectors[idxs]

    # Normalize
    sample_norm = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-9)

    # Compute pairwise cosines
    cos_matrix = sample_norm @ sample_norm.T
    mask = ~np.eye(n, dtype=bool)
    off_diag = cos_matrix[mask]

    mean_cos = np.mean(off_diag)
    std_cos = np.std(off_diag)

    print(f"  Random doc-doc cosines (sample {n:,}):")
    print(f"    Mean: {mean_cos:.4f}")
    print(f"    Std: {std_cos:.4f}")
    print(f"    Min: {np.min(off_diag):.4f}")
    print(f"    Max: {np.max(off_diag):.4f}")

    if abs(mean_cos) > 0.10:
        print(f"  ⚠️  Mean far from 0.0! Consider whitening docs.")
        print(f"    Recommendation: Generate whitened bank before training")
    else:
        print(f"  ✓ Bank is reasonably centered (identity doc tower OK)")

    return {'mean': float(mean_cos), 'std': float(std_cos)}


def check_curriculum_schedule(mine_schedule):
    """Verify curriculum schedule string"""
    print("\n=== 4. Curriculum Schedule Verification ===")

    print(f"  Schedule: {mine_schedule}")

    # Parse schedule
    schedule = {}
    for seg in mine_schedule.split(';'):
        epoch_range, spec = seg.split(':')
        start, end = map(int, epoch_range.split('-'))
        print(f"    Epochs {start}-{end}: {spec}")

        for epoch in range(start, end + 1):
            if spec == 'none':
                schedule[epoch] = None
            else:
                # Parse "8@0.82-0.92"
                num, cos_range = spec.split('@')
                cos_min, cos_max = map(float, cos_range.split('-'))
                schedule[epoch] = {'num': int(num), 'cos_min': cos_min, 'cos_max': cos_max}

    # Verify critical epochs
    assert schedule[1] is None, "❌ Epoch 1 should have no mining (warm start)"
    assert schedule[5] is None, "❌ Epoch 5 should have no mining (warm start)"
    assert schedule[6] is not None, "❌ Epoch 6 should start mining"
    assert schedule[11] is not None, "❌ Epoch 11 should have full mining"

    print(f"  ✓ Curriculum schedule valid")
    print(f"    Warm start: Epochs 1-5 (no mining)")
    print(f"    Gentle phase: Epochs 6-10 ({schedule[6]['num']} hards @ {schedule[6]['cos_min']:.2f}-{schedule[6]['cos_max']:.2f})")
    print(f"    Full phase: Epochs 11+ ({schedule[11]['num']} hards @ {schedule[11]['cos_min']:.2f}-{schedule[11]['cos_max']:.2f})")

    return schedule


def check_filter_threshold(filter_threshold):
    """Verify hard-neg filter is enabled"""
    print("\n=== 5. Hard-Neg Filter Check ===")

    print(f"  Filter threshold: {filter_threshold}")

    assert filter_threshold > 0.95, "❌ Filter threshold too low! Should be >0.95"

    print(f"  ✓ Filter enabled: Will drop negs with cos>{filter_threshold} to both q and pos")
    print(f"    This prevents near-duplicate negatives (Phase 2's main issue)")

    return filter_threshold


def generate_manifest(pairs_path, out_path, expansion_args):
    """Generate manifest for reproducibility"""
    print("\n=== 6. Generating Manifest ===")

    data = np.load(pairs_path)

    manifest = {
        'pairs_file': str(pairs_path),
        'train_count': int(data['X_train'].shape[0]),
        'val_count': int(data['X_val'].shape[0]),
        'expansion_args': expansion_args,
        'file_size_gb': Path(pairs_path).stat().st_size / 1e9,
        'sha256': None  # Could add file hash if needed
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ Manifest saved: {out_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="v4 Pre-flight Checklist")
    parser.add_argument('--pairs', type=str, required=True, help='Training pairs NPZ')
    parser.add_argument('--bank', type=str, required=True, help='Vector bank NPZ')
    parser.add_argument('--mine-schedule', type=str,
                        default='0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96',
                        help='Mining schedule')
    parser.add_argument('--filter-threshold', type=float, default=0.98,
                        help='Hard-neg filter threshold')
    parser.add_argument('--expansion-args', type=str, default='{}',
                        help='JSON string of expansion args for manifest')
    parser.add_argument('--manifest-out', type=str,
                        default='artifacts/twotower/pairs_v4_manifest.json',
                        help='Output manifest path')
    args = parser.parse_args()

    print("============================================================")
    print("V4 PRE-FLIGHT CHECKLIST")
    print("============================================================")
    print(f"Pairs: {args.pairs}")
    print(f"Bank: {args.bank}")
    print()

    results = {}

    # 1. Data sanity
    results['data_sanity'] = check_data_sanity(args.pairs)

    # 2. Leakage check
    results['leakage'] = check_leakage(args.pairs)

    # 3. Doc alignment
    results['doc_alignment'] = check_doc_alignment(args.bank)

    # 4. Curriculum schedule
    results['curriculum'] = check_curriculum_schedule(args.mine_schedule)

    # 5. Filter threshold
    results['filter_threshold'] = check_filter_threshold(args.filter_threshold)

    # 6. Generate manifest
    expansion_args = json.loads(args.expansion_args)
    results['manifest'] = generate_manifest(args.pairs, args.manifest_out, expansion_args)

    # Summary
    print("\n============================================================")
    print("PRE-FLIGHT SUMMARY")
    print("============================================================")

    all_clear = True

    if results['leakage']['ctx_leaks'] > 0 or results['leakage']['tgt_leaks'] > 0:
        print("⚠️  WARNING: Data leakage detected")
        all_clear = False

    if abs(results['doc_alignment']['mean']) > 0.10:
        print("⚠️  WARNING: Bank alignment off-center (consider whitening)")
        # Not a blocker, just a recommendation

    if all_clear:
        print("✅ ALL CHECKS PASSED")
        print("\nReady to launch v4 training!")
        print("  ./launch_v4_overnight.sh")
    else:
        print("⚠️  WARNINGS DETECTED")
        print("\nReview warnings above. You may proceed, but consider fixes for optimal results.")

    print()

    # Save results
    results_path = Path(args.pairs).parent / 'preflight_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
