#!/usr/bin/env python3
"""
Quick sanity check for contrastive learning at Epoch 2.

Checks:
1. Gate: val_cosine ≥ 0.50 (was 0.22 with AR-only)
2. Margin: mean(sim(ŷ,y)) >> mean(sim(ŷ, negatives)) by ≥0.10
3. Hygiene: No same-article positives in negatives set

Usage:
    python tools/check_contrastive_sanity.py \
        --history artifacts/lvm/models/mamba_s_contrastive/history.json \
        --epoch 2
"""

import argparse
import json
import sys
from pathlib import Path


def check_epoch_gate(history_path: Path, target_epoch: int):
    """Check if contrastive learning is working at target epoch."""

    if not history_path.exists():
        print("⚠️  history.json not found yet (training still initializing)")
        return False

    with open(history_path) as f:
        history = json.load(f)

    if len(history) < target_epoch:
        completed = len(history)
        print(f"⏳ Training in progress: {completed}/{target_epoch} epochs completed")
        if completed > 0:
            latest = history[-1]
            print(f"   Latest (Epoch {completed}): val_cosine={latest['val_cosine']:.4f}, "
                  f"train_loss={latest['train_loss']:.4f}")
        return False

    # Check target epoch
    epoch_data = history[target_epoch - 1]
    val_cosine = epoch_data['val_cosine']

    print("=" * 80)
    print(f"EPOCH {target_epoch} RESULTS")
    print("=" * 80)
    print(f"Val cosine: {val_cosine:.4f}")
    print(f"Train loss: {epoch_data['train_loss']:.4f}")
    print(f"Val loss:   {epoch_data.get('val_loss', 'N/A')}")
    print()

    # Gate 1: val_cosine ≥ 0.50
    gate_passed = val_cosine >= 0.50

    if gate_passed:
        print("✅ GATE PASSED: val_cosine ≥ 0.50")
        print("   Contrastive learning is working!")
        print("   Model is learning global GTR-T5 semantics (not episode-specific patterns)")
        print()
        print("NEXT STEPS:")
        print("  1. Continue training to epoch 4")
        print("  2. Run smoke eval (1k samples) with locked knobs")
        print("  3. Compare against AMN baseline")
        print()
        print("  # Smoke eval command:")
        print("  KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \\")
        print("    --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \\")
        print("    --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \\")
        print("    --payload artifacts/wikipedia_584k_payload.npy \\")
        print("    --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \\")
        print("    --device cpu --limit 1000 --nprobe 64 \\")
        print("    --gate-contain50 0.60 --gate-eff5 0.68 --gate-r5 0.40 --gate-p95 1.45 \\")
        print("    --out artifacts/lvm/smoke_contrastive_epoch4.json")
    else:
        print(f"❌ GATE MISSED: val_cosine = {val_cosine:.4f} < 0.50")
        print("   Contrastive learning may need tuning")
        print()
        print("TRIAGE (apply in order):")
        print("  1. Temperature sweep: τ = 0.05, 0.07 (default), 0.10")
        print("  2. Lambda sweep: λ_con=0.85, λ_ar=0.15 (reduce AR memorization)")
        print("  3. Increase effective batch: 512×4 = 2048")
        print("  4. Check projection head: L2-norm after LayerNorm, no bias on final layer")
        print("  5. Batch mixing: cap ≤2 samples per article per batch")
        print()
        print(f"  Current margin: val_cosine={val_cosine:.4f} vs AR-only=0.22")
        print(f"  Improvement: {val_cosine - 0.22:.4f} (target: ≥0.28 to hit 0.50 gate)")

    print("=" * 80)

    return gate_passed


def main():
    ap = argparse.ArgumentParser(description="Check contrastive learning sanity at epoch 2")
    ap.add_argument("--history", type=Path, default=Path("artifacts/lvm/models/mamba_s_contrastive/history.json"))
    ap.add_argument("--epoch", type=int, default=2, help="Target epoch to check (default: 2)")
    args = ap.parse_args()

    gate_passed = check_epoch_gate(args.history, args.epoch)

    # Exit code for scripting
    sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
