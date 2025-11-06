#!/usr/bin/env python3
"""
Sentence-level temporal delta probe.

Tests if sentence-level granularity shows forward temporal signal
compared to paragraph-level. Measures Δ = cos(last_sent, next_sent) - cos(last_sent, prev_sent).

Usage:
    python tools/sentence_delta_check.py --glob "data/datasets/narrative/*.txt" --max-files 20
    python tools/sentence_delta_check.py --glob "data/datasets/arxiv/*.txt" --max-files 10
"""
import argparse
import glob
import os
import re
import json
import numpy as np
import requests
import sys

ENCODER_URL = os.environ.get("ENCODER_URL", "http://localhost:8767/embed")

def sents(txt):
    """Split text into sentences, conservative approach."""
    txt = txt.replace("\r", "")

    # Replace common abbreviations with placeholders
    abbr_map = [
        ("Mr.", "Mr<DOT>"), ("Mrs.", "Mrs<DOT>"), ("Ms.", "Ms<DOT>"),
        ("Dr.", "Dr<DOT>"), ("Prof.", "Prof<DOT>"), ("Sr.", "Sr<DOT>"),
        ("Jr.", "Jr<DOT>"), ("St.", "St<DOT>"), ("vs.", "vs<DOT>"),
        ("etc.", "etc<DOT>"), ("e.g.", "e<DOT>g<DOT>"), ("i.e.", "i<DOT>e<DOT>")
    ]
    for abbr, placeholder in abbr_map:
        txt = txt.replace(abbr, placeholder)

    # Split on sentence boundaries
    parts = re.split(r'[.!?]\s+(?=[A-Z"\'])', txt)

    # Restore abbreviations
    parts = [p.replace("<DOT>", ".") for p in parts]

    # Clean and filter: at least 20 alphabetic characters
    parts = [p.strip() for p in parts if p.strip()]
    return [p for p in parts if sum(ch.isalpha() for ch in p) >= 20]

def encode(texts, bs=128):
    """Encode texts using GTR-T5 encoder service."""
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        r = requests.post(ENCODER_URL, json={"texts": batch})
        r.raise_for_status()
        V = np.array(r.json()["embeddings"], dtype=np.float32)
        # Normalize
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        out.append(V)
    return np.vstack(out) if out else np.zeros((0, 768), np.float32)

def main():
    ap = argparse.ArgumentParser(description="Sentence-level temporal delta probe")
    ap.add_argument("--glob", required=True, help="Glob pattern for input text files")
    ap.add_argument("--max-files", type=int, default=20, help="Maximum files to process")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))[:args.max_files]

    if not files:
        print(f"No files found matching: {args.glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} files...", file=sys.stderr)

    seqC = []  # All context sentences (5 per sequence)
    seqT = []  # All target sentences (1 per sequence)

    for p in files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
        except Exception as e:
            print(f"Skipping {p}: {e}", file=sys.stderr)
            continue

        S = sents(t)
        print(f"  {os.path.basename(p)}: {len(S)} sentences", file=sys.stderr)

        # Create sequences: ctx[i-5:i] → target[i]
        for i in range(5, len(S)):
            ctx = S[i-5:i]
            nxt = S[i]
            seqC.extend(ctx)
            seqT.append(nxt)

    if not seqT:
        print("No sequences created.", file=sys.stderr)
        sys.exit(2)

    print(f"\nCreated {len(seqT)} sequences. Encoding...", file=sys.stderr)

    # Encode all texts
    V = encode(seqC + seqT)
    D = V.shape[1]  # 768
    N = len(seqT)

    # Reshape: context = [N, 5, D], target = [N, D]
    C = V[:5*N].reshape(N, 5, D)
    T = V[5*N:]

    # Last context sentence
    C_last = C[:, -1, :]

    # Previous = second-to-last context sentence (C[:, -2, :])
    C_prev = C[:, -2, :]

    # Cosine similarities
    cos_next = np.sum(C_last * T, axis=1)
    cos_prev = np.sum(C_last * C_prev, axis=1)
    delta = cos_next - cos_prev

    # Results
    result = {
        "N": int(N),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std()),
        "q25_q50_q75": [float(np.quantile(delta, q)) for q in (0.25, 0.5, 0.75)],
        "cos_next": float(cos_next.mean()),
        "cos_prev": float(cos_prev.mean()),
        "files_processed": len(files)
    }

    print("\n" + "="*60)
    print(json.dumps(result, indent=2))
    print("="*60)

    # Decision gate
    if result["mean_delta"] >= 0.01:
        print("\n✅ GATE PASSED: Δ ≥ 0.01 - adopt sentence-aware retrieval")
    elif result["mean_delta"] >= 0.005:
        print("\n⚠️  MARGINAL: Δ = {:.4f} - consider adopting for finer granularity".format(result["mean_delta"]))
    else:
        print("\n⚠️  WEAK SIGNAL: Δ = {:.4f} - still adopt for retrieval benefits".format(result["mean_delta"]))

if __name__ == "__main__":
    main()
