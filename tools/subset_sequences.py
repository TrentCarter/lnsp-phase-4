#!/usr/bin/env python3
import argparse, numpy as np, sys, os, json, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to full sequences .npz")
    ap.add_argument("--output", required=True, help="Path to write subset .npz")
    ap.add_argument("--n", type=int, default=10000, help="Total sequences to sample")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    data = np.load(args.input, allow_pickle=True)
    keys = list(data.keys())

    # Identify the length dimension robustly
    lens = {}
    for k in keys:
        try:
            lens[k] = len(data[k])
        except Exception:
            pass
    if not lens:
        print("Could not infer sequence length from any key.", file=sys.stderr); sys.exit(2)
    # Use the mode length across arrays
    lengths = list(lens.values())
    N = max(set(lengths), key=lengths.count)

    idx = np.arange(N)
    rng.shuffle(idx)
    idx = idx[:min(args.n, N)]

    # Optional: try stratify by article if present
    if "article_ids" in keys:
        # Keep distribution proportional
        aid = data["article_ids"][:N]
        unique, counts = np.unique(aid, return_counts=True)
        target = args.n
        take = []
        for u, c in zip(unique, counts):
            n_u = max(1, int(round(target * (c / N))))
            take.extend(rng.choice(np.where(aid == u)[0], size=min(n_u, c), replace=False))
        idx = np.array(take[:min(len(take), args.n)])
        rng.shuffle(idx)

    subset = {}
    for k in keys:
        arr = data[k]
        if hasattr(arr, "__len__") and len(arr) == N:
            subset[k] = arr[idx]
        else:
            subset[k] = arr  # keep metadata/scalars as-is

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **subset)

    # Brief manifest
    manifest = {k: (np.asarray(subset[k]).shape if hasattr(subset[k], "shape") else "scalar")
                for k in subset.keys()}
    print(json.dumps({"wrote": args.output, "N_subset": len(idx), "keys": manifest}, indent=2))

if __name__ == "__main__":
    main()
