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


def build_curriculum_splits(
    train_npz: Path,
    scores_npz: Path,
    output_dir: Path = None,
    tau_sim_A: float = None,
    tau_adv_A: float = None,
    tau_sim_B: float = None,
    tau_adv_B: float = None
):
    """
    Build curriculum dataset splits using forward-advantage thresholds.

    Uses richer selection criteria beyond simple similarity:
    - Stage A: sim_prev ≥ τ_sim_A AND adv_prev ≥ τ_adv_A (direction + uniqueness)
    - Stage B: sim_prev ≥ τ_sim_B OR adv_prev ≥ τ_adv_B (looser)
    - Stage C: All samples

    Args:
        train_npz: Original training NPZ
        scores_npz: Forward-advantage scores NPZ (from compute_forward_distinctness.py)
        output_dir: Where to save splits (default: same as train_npz)
        tau_sim_A: Stage A similarity threshold (default: 0.66)
        tau_adv_A: Stage A advantage threshold (default: 0.08)
        tau_sim_B: Stage B similarity threshold (default: 0.58)
        tau_adv_B: Stage B advantage threshold (default: 0.05)
    """
    print(f"📥 Loading training data: {train_npz}")
    train_data = np.load(train_npz, allow_pickle=True)

    print(f"📥 Loading scores: {scores_npz}")
    scores_data = np.load(scores_npz, allow_pickle=True)

    # Load richer metrics (P5.1)
    sim_prev = scores_data['sim_prev']
    sim_prev2 = scores_data['sim_prev2']
    sim_other_max = scores_data['sim_other_max']
    adv_prev = scores_data['adv_prev']
    delta_prev2 = scores_data['delta_prev2']

    # Use provided thresholds or defaults from scores file
    if tau_sim_A is None:
        tau_sim_A = float(scores_data.get('tau_sim_A', 0.66))
    if tau_adv_A is None:
        tau_adv_A = float(scores_data.get('tau_adv_A', 0.08))
    if tau_sim_B is None:
        tau_sim_B = float(scores_data.get('tau_sim_B', 0.58))
    if tau_adv_B is None:
        tau_adv_B = float(scores_data.get('tau_adv_B', 0.05))

    N = len(sim_prev)
    print(f"   Total samples: {N:,}")

    print(f"\n🎯 Curriculum Thresholds (forward-advantage based):")
    print(f"   Stage A: sim_prev ≥ {tau_sim_A:.3f} AND adv_prev ≥ {tau_adv_A:.3f}")
    print(f"   Stage B: sim_prev ≥ {tau_sim_B:.3f} OR  adv_prev ≥ {tau_adv_B:.3f}")

    # Create masks using consultant's criteria
    mask_A = (sim_prev >= tau_sim_A) & (adv_prev >= tau_adv_A)
    mask_B = (sim_prev >= tau_sim_B) | (adv_prev >= tau_adv_B)

    print(f"\n📊 Curriculum Masks (initial):")
    print(f"   Stage A: {mask_A.sum():,} samples ({100*mask_A.mean():.1f}%)")
    print(f"   Stage B: {mask_B.sum():,} samples ({100*mask_B.mean():.1f}%)")
    print(f"   Stage C: {N:,} samples (100.0%)")

    # Sanity checks (fail-fast)
    def print_stats(tag, mask):
        """Print statistics for a curriculum stage"""
        sp = sim_prev[mask]
        ap = adv_prev[mask]
        dp2 = delta_prev2[mask]
        print(f"\n[CURR/{tag}] Statistics:")
        print(f"   n={mask.sum():,} ({100*mask.mean():.1f}%)")
        print(f"   sim_prev:    mean={sp.mean():.3f}, median={np.median(sp):.3f}, std={sp.std():.3f}")
        print(f"   adv_prev:    mean={ap.mean():.3f}, median={np.median(ap):.3f}, std={ap.std():.3f}")
        print(f"   delta_prev2: mean={dp2.mean():.3f}, median={np.median(dp2):.3f}, std={dp2.std():.3f}")
        print(f"   pct_adv>0:   {100*(ap > 0).mean():.1f}%")
        return sp, ap, dp2

    print("\n" + "="*60)
    print("SANITY CHECKS")
    print("="*60)

    sp_A, ap_A, dp2_A = print_stats("A", mask_A)
    sp_B, ap_B, dp2_B = print_stats("B", mask_B)

    # Fail-fast validation (Stage A must be forward-advantaged)
    print(f"\n🔍 Validation:")

    # Check 1: Stage A must have samples
    if mask_A.sum() == 0:
        print(f"   ❌ FAIL: Stage A has ZERO samples!")
        print(f"       Thresholds too strict: sim_prev≥{tau_sim_A}, adv_prev≥{tau_adv_A}")
        raise SystemExit("[CURR/ERROR] Stage A is empty. Relax thresholds or check score calculation.")

    # Check 2: Stage A must have positive mean advantage
    if ap_A.mean() < 0.02:
        print(f"   ❌ FAIL: Stage A mean(adv_prev) = {ap_A.mean():.3f} < 0.02")
        print(f"       Stage A is NOT forward-advantaged!")
        raise SystemExit("[CURR/ERROR] Stage A split is not forward-advantaged. Check thresholds or score calc.")

    # Check 3: Stage A must have majority positive advantage
    pct_positive = 100 * (ap_A > 0).mean()
    if pct_positive < 70:
        print(f"   ⚠️  WARNING: Stage A has only {pct_positive:.1f}% samples with adv_prev > 0")
        print(f"       Threshold may be too loose!")

    # Check 4: Stage B must be superset of Stage A
    if not np.all(mask_A <= mask_B):
        print(f"   ⚠️  WARNING: Stage B is NOT a superset of Stage A!")
        # Fix by taking union
        mask_B = mask_A | mask_B
        print(f"       Fixed: Stage B now includes all Stage A samples")
        print(f"       New Stage B count: {mask_B.sum():,} ({100*mask_B.mean():.1f}%)")

    print(f"   ✅ PASS: Stage A has {mask_A.sum():,} forward-advantaged samples")
    print(f"   ✅ PASS: Stage A mean(adv_prev) = {ap_A.mean():.3f} ≥ 0.02")
    print(f"   ✅ PASS: Stage A has {pct_positive:.1f}% samples with adv_prev > 0")

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

    # Stage A: Forward-advantaged subset
    print(f"\n🔨 Building Stage A (forward-advantaged)...")
    stage_a_data = create_subset(mask_A, "stage_a")
    stage_a_path = output_dir / f"{train_npz.stem}_stage_a.npz"
    np.savez_compressed(stage_a_path, **stage_a_data)
    print(f"   ✅ Saved: {stage_a_path}")

    # Stage B: Relaxed forward-advantaged subset
    print(f"\n🔨 Building Stage B (relaxed)...")
    stage_b_data = create_subset(mask_B, "stage_b")
    stage_b_path = output_dir / f"{train_npz.stem}_stage_b.npz"
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
    parser.add_argument("--scores-npz", type=Path, required=True, help="Forward-advantage scores NPZ")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: same as train-npz)")

    # P5.1: Threshold parameters (forward-advantage based)
    parser.add_argument("--tau-sim-A", type=float, default=None,
                        help="Stage A similarity threshold (default: 0.66)")
    parser.add_argument("--tau-adv-A", type=float, default=None,
                        help="Stage A advantage threshold (default: 0.08)")
    parser.add_argument("--tau-sim-B", type=float, default=None,
                        help="Stage B similarity threshold (default: 0.58)")
    parser.add_argument("--tau-adv-B", type=float, default=None,
                        help="Stage B advantage threshold (default: 0.05)")

    args = parser.parse_args()

    build_curriculum_splits(
        args.train_npz,
        args.scores_npz,
        args.output_dir,
        tau_sim_A=args.tau_sim_A,
        tau_adv_A=args.tau_adv_A,
        tau_sim_B=args.tau_sim_B,
        tau_adv_B=args.tau_adv_B
    )


if __name__ == "__main__":
    main()
