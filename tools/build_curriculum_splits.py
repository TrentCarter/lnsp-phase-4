#!/usr/bin/env python3
"""
Build curriculum dataset splits based on forward-distinctness scores.

Creates 3 NPZ files:
- Stage A (top 30%): Most forward-distinct samples
- Stage B (top 70%): Top 70% of samples
- Stage C (full):    All samples (copy of original)

Used by train_unified.py with --curriculum flag.
"""

import argparse
import numpy as np
from pathlib import Path


def build_curriculum_splits(train_npz: Path, scores_npz: Path, output_dir: Path = None):
    """
    Build curriculum dataset splits.

    Args:
        train_npz: Original training NPZ
        scores_npz: Forward-distinctness scores NPZ
        output_dir: Where to save splits (default: same as train_npz)
    """
    print(f"📥 Loading training data: {train_npz}")
    train_data = np.load(train_npz, allow_pickle=True)

    print(f"📥 Loading scores: {scores_npz}")
    scores_data = np.load(scores_npz, allow_pickle=True)

    forward_distinctness = scores_data['forward_distinctness']
    threshold_top30 = scores_data['threshold_top30']
    threshold_top70 = scores_data['threshold_top70']

    N = len(forward_distinctness)
    print(f"   Total samples: {N:,}")

    # Create masks
    mask_top30 = forward_distinctness >= threshold_top30
    mask_top70 = forward_distinctness >= threshold_top70

    print(f"\n📊 Curriculum Masks:")
    print(f"   Stage A (top 30%): {mask_top30.sum():,} samples (threshold ≥ {threshold_top30:.4f})")
    print(f"   Stage B (top 70%): {mask_top70.sum():,} samples (threshold ≥ {threshold_top70:.4f})")
    print(f"   Stage C (full):    {N:,} samples (all)")

    # Prepare output directory
    if output_dir is None:
        output_dir = train_npz.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Function to create subset NPZ
    def create_subset(mask, stage_name):
        subset_data = {}
        for key in train_data.keys():
            if key in ['context_sequences', 'target_vectors']:
                # Filter sequences
                subset_data[key] = train_data[key][mask]
            elif key == 'num_sequences':
                subset_data[key] = mask.sum()
            elif key == 'metadata':
                # Update metadata
                metadata = train_data[key].item() if train_data[key].ndim == 0 else train_data[key]
                if isinstance(metadata, dict):
                    metadata = metadata.copy()
                    metadata['curriculum_stage'] = stage_name
                    metadata['original_num_sequences'] = N
                    metadata['filtered_num_sequences'] = mask.sum()
                subset_data[key] = metadata
            else:
                # Copy as-is
                subset_data[key] = train_data[key]

        return subset_data

    # Stage A: Top 30%
    print(f"\n🔨 Building Stage A (top 30%)...")
    stage_a_data = create_subset(mask_top30, "stage_a_top30")
    stage_a_path = output_dir / f"{train_npz.stem}_stage_a_top30.npz"
    np.savez_compressed(stage_a_path, **stage_a_data)
    print(f"   ✅ Saved: {stage_a_path}")

    # Stage B: Top 70%
    print(f"\n🔨 Building Stage B (top 70%)...")
    stage_b_data = create_subset(mask_top70, "stage_b_top70")
    stage_b_path = output_dir / f"{train_npz.stem}_stage_b_top70.npz"
    np.savez_compressed(stage_b_path, **stage_b_data)
    print(f"   ✅ Saved: {stage_b_path}")

    # Stage C: Full (just copy original)
    print(f"\n🔨 Building Stage C (full)...")
    stage_c_path = output_dir / f"{train_npz.stem}_stage_c_full.npz"
    # Just create a symlink or copy
    import shutil
    shutil.copy(train_npz, stage_c_path)
    print(f"   ✅ Saved: {stage_c_path}")

    print(f"\n✅ Curriculum splits complete!")
    print(f"   Stage A: {stage_a_path}")
    print(f"   Stage B: {stage_b_path}")
    print(f"   Stage C: {stage_c_path}")

    return stage_a_path, stage_b_path, stage_c_path


def main():
    parser = argparse.ArgumentParser(description="Build curriculum dataset splits")
    parser.add_argument("--train-npz", type=Path, required=True, help="Training NPZ file")
    parser.add_argument("--scores-npz", type=Path, required=True, help="Forward-distinctness scores NPZ")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: same as train-npz)")

    args = parser.parse_args()

    build_curriculum_splits(args.train_npz, args.scores_npz, args.output_dir)


if __name__ == "__main__":
    main()
