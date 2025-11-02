#!/usr/bin/env python3
"""
Generic OOD Evaluation for Any LVM Model

Evaluates any LVM model on out-of-distribution test data.

Usage:
    ./.venv/bin/python tools/eval_model_ood.py \
        --model artifacts/lvm/models/amn_790k_20251030_110346/best_model.pt
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.model import load_lvm_model


def cosine_similarity(pred, target):
    """Compute cosine similarity"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean().item()


def evaluate_model(model, contexts, targets, device, batch_size=32):
    """Evaluate model on test data"""
    model.eval()

    # Convert to tensors
    if not isinstance(contexts, torch.Tensor):
        contexts = torch.from_numpy(contexts).float()
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).float()

    contexts = contexts.to(device)
    targets = targets.to(device)

    cosine_scores = []
    mse_scores = []

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]

            # Forward pass
            pred = model(batch_ctx)

            # Metrics
            cos_sim = cosine_similarity(pred, batch_tgt)
            mse = F.mse_loss(pred, batch_tgt).item()

            cosine_scores.append(cos_sim)
            mse_scores.append(mse)

    return {
        'cosine_mean': np.mean(cosine_scores),
        'cosine_std': np.std(cosine_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
    }


def neighbor_sweep(model, contexts, device, batch_size=32):
    """
    Neighbor sweep diagnostic: Test predictions against multiple target offsets.

    Given context [t-4, t-3, t-2, t-1, t], model should predict t+1.
    This tests cos(pred, {t-4, t-3, t-2, t-1, t, t+1}) to detect misalignment.

    If peak is at t or t-1 instead of t+1, the OOD builder shifted window wrong.
    """
    model.eval()

    # Convert to tensors
    if not isinstance(contexts, torch.Tensor):
        contexts = torch.from_numpy(contexts).float()
    contexts = contexts.to(device)

    # Extract vectors at different positions from context window
    # contexts shape: (N, 5, 768) = [t-4, t-3, t-2, t-1, t]
    t_minus_4 = contexts[:, 0, :]  # t-4
    t_minus_3 = contexts[:, 1, :]  # t-3
    t_minus_2 = contexts[:, 2, :]  # t-2
    t_minus_1 = contexts[:, 3, :]  # t-1
    t_current = contexts[:, 4, :]  # t

    # Generate predictions
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            pred = model(batch_ctx)
            all_preds.append(pred)

    predictions = torch.cat(all_preds, dim=0)

    # Compute cosine similarity against each position
    results = {}
    test_vectors = {
        't-4': t_minus_4,
        't-3': t_minus_3,
        't-2': t_minus_2,
        't-1': t_minus_1,
        't': t_current,
    }

    for offset_name, test_vec in test_vectors.items():
        cos_sim = cosine_similarity(predictions, test_vec)
        results[offset_name] = cos_sim

    return results


def sign_flip_test(model, contexts, targets, device, batch_size=32):
    """
    Sign-flip diagnostic: Test cos(pred, -target) for inversion detection.

    If cos(pred, -target) jumps to ~+0.5 while cos(pred, target) is negative,
    targets were accidentally inverted when writing the OOD file.
    """
    model.eval()

    # Convert to tensors
    if not isinstance(contexts, torch.Tensor):
        contexts = torch.from_numpy(contexts).float()
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).float()

    contexts = contexts.to(device)
    targets = targets.to(device)

    normal_scores = []
    flipped_scores = []

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i:i+batch_size]
            batch_tgt = targets[i:i+batch_size]

            # Forward pass
            pred = model(batch_ctx)

            # Normal cosine (pred vs target)
            cos_normal = cosine_similarity(pred, batch_tgt)

            # Flipped cosine (pred vs -target)
            cos_flipped = cosine_similarity(pred, -batch_tgt)

            normal_scores.append(cos_normal)
            flipped_scores.append(cos_flipped)

    return {
        'normal': np.mean(normal_scores),
        'flipped': np.mean(flipped_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LVM model on OOD data")
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--ood-data', default='artifacts/lvm/wikipedia_ood_test_ctx5.npz',
                        help='OOD test data file')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--neighbor-sweep', action='store_true',
                        help='Run neighbor sweep diagnostic (tests pred vs multiple target offsets)')
    parser.add_argument('--sign-flip-test', action='store_true',
                        help='Run sign-flip test (tests cos(pred, -target) for inversion detection)')
    args = parser.parse_args()

    model_path = Path(args.model)
    ood_path = Path(args.ood_data)

    print("=" * 60)
    print("LVM OOD EVALUATION")
    print("=" * 60)
    print()

    # Determine device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"OOD Data: {ood_path}")
    print()

    # Load checkpoint to determine model type
    print("📦 Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_type = checkpoint.get('model_type', 'unknown')
    val_cosine = checkpoint.get('val_cosine', 'N/A')

    print(f"   Model Type: {model_type}")
    print(f"   In-Dist Val Cosine: {val_cosine:.4f}" if isinstance(val_cosine, float) else f"   In-Dist Val Cosine: {val_cosine}")
    print()

    # Load model
    print(f"📦 Loading {model_type.upper()} model...")
    model = load_lvm_model(model_type, str(model_path), device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded (params: {params:,})")
    print()

    # Load OOD data
    if not ood_path.exists():
        print(f"❌ ERROR: OOD test data not found: {ood_path}")
        sys.exit(1)

    print("📊 Loading OOD test data...")
    ood_data = np.load(ood_path, allow_pickle=True)
    ood_ctx = ood_data['context_sequences']
    ood_tgt = ood_data['target_vectors']
    print(f"   ✅ Loaded {len(ood_ctx):,} OOD test sequences")
    print(f"   Context shape: {ood_ctx.shape}")
    print(f"   Target shape: {ood_tgt.shape}")
    print()

    # Evaluate
    print("🔬 Evaluating OOD performance...")
    results = evaluate_model(model, ood_ctx, ood_tgt, device)
    print()

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"OOD Cosine Similarity: {results['cosine_mean']:.4f} ± {results['cosine_std']:.4f}")
    print(f"OOD MSE Loss:          {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print()

    # Compare with training val score
    print("COMPARISON:")
    print(f"  In-Distribution (Val): {val_cosine:.4f}" if isinstance(val_cosine, float) else f"  In-Distribution (Val): {val_cosine}")
    print(f"  Out-of-Distribution:   {results['cosine_mean']:.4f}")

    if isinstance(val_cosine, float):
        delta = results['cosine_mean'] - val_cosine
        print(f"  Δ (OOD - In-Dist):     {delta:+.4f}")

        if delta > 0:
            print(f"  ✅ Excellent! Model generalizes better to OOD data (+{delta:.4f})")
        elif delta > -0.05:
            print(f"  ✅ Good! Model maintains performance on OOD data ({delta:.4f})")
        else:
            print(f"  ⚠️  Model struggles with OOD data ({delta:.4f})")

    print()
    print("=" * 60)

    # Run diagnostics if requested
    if args.neighbor_sweep or args.sign_flip_test:
        print()
        print("=" * 60)
        print("DIAGNOSTICS")
        print("=" * 60)
        print()

    if args.neighbor_sweep:
        print("🔍 Neighbor Sweep Diagnostic")
        print("-" * 60)
        print("Testing predictions against multiple target offsets...")
        print("(If peak is at t or t-1 instead of t+1, OOD window shifted wrong)")
        print()

        sweep_results = neighbor_sweep(model, ood_ctx, device)

        print("Cosine similarity: pred vs...")
        for offset, score in sorted(sweep_results.items()):
            marker = " ← PEAK!" if score == max(sweep_results.values()) else ""
            print(f"  {offset:>4s}: {score:+.4f}{marker}")

        # Also test against t+1 (actual target)
        print(f"  t+1:  {results['cosine_mean']:+.4f} ← Expected peak")
        print()

        # Interpretation
        max_offset = max(sweep_results, key=sweep_results.get)
        max_score = sweep_results[max_offset]

        print("INTERPRETATION:")
        if max_score > results['cosine_mean']:
            print(f"  🚨 MISALIGNMENT DETECTED!")
            print(f"     Peak is at {max_offset} ({max_score:+.4f}), NOT at t+1 ({results['cosine_mean']:+.4f})")
            print(f"     → OOD builder shifted window wrong!")
        else:
            print(f"  ✅ Alignment looks correct (peak at t+1)")
        print()

    if args.sign_flip_test:
        print("🔄 Sign-Flip Diagnostic")
        print("-" * 60)
        print("Testing cos(pred, -target) for inversion detection...")
        print()

        flip_results = sign_flip_test(model, ood_ctx, ood_tgt, device)

        print(f"  cos(pred, +target): {flip_results['normal']:+.4f}")
        print(f"  cos(pred, -target): {flip_results['flipped']:+.4f}")
        print()

        # Interpretation
        print("INTERPRETATION:")
        if flip_results['flipped'] > 0.3 and flip_results['normal'] < 0:
            print(f"  🚨 SIGN INVERSION DETECTED!")
            print(f"     Flipped cosine is positive ({flip_results['flipped']:+.4f})")
            print(f"     → Targets were accidentally inverted when writing OOD file!")
        elif abs(flip_results['flipped'] - (-flip_results['normal'])) < 0.01:
            print(f"  ✅ Sign is correct (flipped ≈ -normal)")
        else:
            print(f"  ⚠️  Unclear - may need further investigation")
        print()
        print("=" * 60)


if __name__ == '__main__':
    main()
