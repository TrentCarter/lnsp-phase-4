#!/usr/bin/env python3
"""
Fit directional adapter for ranking (NOT AR-LVM).

Learns a cheap 768×768 linear transform W that slightly prefers forward
over backward, used only in reranking (not autoregressive generation).

Fits W to maximize: E[cos(s_last·W, next·W) - cos(s_last·W, prev·W)]
using ridge regression.

Usage:
    python tools/fit_directional_adapter.py \
        --seq-npz artifacts/lvm/training_sequences_ctx5.npz \
        --out-npz artifacts/directional_adapter.npz \
        --lambda 0.01
"""
import argparse
import numpy as np
import os

def main():
    ap = argparse.ArgumentParser(description="Fit directional adapter for ranking")
    ap.add_argument("--seq-npz", required=True,
                   help="Training sequences NPZ: contexts [N,5,768], targets [N,768]")
    ap.add_argument("--out-npz", required=True,
                   help="Output NPZ for adapter matrix W")
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-2,
                   help="Ridge regularization lambda")
    ap.add_argument("--max-samples", type=int, default=50000,
                   help="Maximum sequences to use (for speed)")
    args = ap.parse_args()

    print(f"Loading sequences: {args.seq_npz}")
    Z = np.load(args.seq_npz, allow_pickle=True)

    # Check available keys
    if "contexts" in Z:
        C = Z["contexts"].astype(np.float32)  # [N, 5, 768]
        T = Z["targets"].astype(np.float32)   # [N, 768]
    elif "context_vecs" in Z:
        C = Z["context_vecs"].astype(np.float32)
        T = Z["target_vecs"].astype(np.float32)
    else:
        print(f"ERROR: Cannot find contexts/targets in {args.seq_npz}")
        print(f"Available keys: {list(Z.keys())}")
        return

    print(f"  Loaded {C.shape[0]} sequences")

    # Subsample if needed
    if C.shape[0] > args.max_samples:
        print(f"  Subsampling to {args.max_samples} sequences...")
        indices = np.random.choice(C.shape[0], args.max_samples, replace=False)
        C = C[indices]
        T = T[indices]

    # Extract previous (C[:,-2,:]) and current (C[:,-1,:])
    P = C[:, -2, :].astype(np.float32)  # Previous sentence
    X = C[:, -1, :].astype(np.float32)  # Last context sentence (query side)

    # Target is "next" sentence
    # Compute forward residual: Y = T - P
    # This captures the "forward direction" from prev to next
    Y = (T - P).astype(np.float32)

    print(f"\nFitting directional adapter:")
    print(f"  Query (X): {X.shape}")
    print(f"  Target residual (Y): {Y.shape}")
    print(f"  Regularization λ: {args.lam}")

    # Solve W from XW ≈ Y using ridge regression:
    # W = (X^T X + λI)^{-1} X^T Y
    XT = X.T  # [768, N]
    A = XT @ X + args.lam * np.eye(X.shape[1], dtype=np.float32)  # [768, 768]
    B = XT @ Y  # [768, 768]

    print("  Solving linear system...")
    W = np.linalg.solve(A, B)  # [768, 768]

    # Validate: measure before/after improvement
    print("\nValidation:")

    # Without adapter
    cos_next_before = np.sum(X * T, axis=1)
    cos_prev_before = np.sum(X * P, axis=1)
    delta_before = cos_next_before - cos_prev_before

    # With adapter
    X_adapted = X @ W
    T_adapted = T @ W
    P_adapted = P @ W

    # Normalize
    X_adapted /= (np.linalg.norm(X_adapted, axis=1, keepdims=True) + 1e-9)
    T_adapted /= (np.linalg.norm(T_adapted, axis=1, keepdims=True) + 1e-9)
    P_adapted /= (np.linalg.norm(P_adapted, axis=1, keepdims=True) + 1e-9)

    cos_next_after = np.sum(X_adapted * T_adapted, axis=1)
    cos_prev_after = np.sum(X_adapted * P_adapted, axis=1)
    delta_after = cos_next_after - cos_prev_after

    print(f"  Before adapter: Δ = {delta_before.mean():.4f}")
    print(f"  After adapter:  Δ = {delta_after.mean():.4f}")
    print(f"  Improvement:    Δ = +{(delta_after.mean() - delta_before.mean()):.4f}")

    if delta_after.mean() <= delta_before.mean():
        print("\n⚠️  WARNING: Adapter did not improve forward bias!")
        print("  Consider adjusting λ or using more data.")
    else:
        print(f"\n✓ Adapter improved forward bias by +{(delta_after.mean() - delta_before.mean()):.4f}")

    # Save
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        W=W.astype(np.float32),
        lambda_reg=args.lam,
        delta_before=delta_before.mean(),
        delta_after=delta_after.mean(),
        improvement=delta_after.mean() - delta_before.mean()
    )

    print(f"\n✓ Saved adapter to: {args.out_npz}")
    print(f"  Matrix shape: {W.shape}")
    print(f"  File size: {os.path.getsize(args.out_npz) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
