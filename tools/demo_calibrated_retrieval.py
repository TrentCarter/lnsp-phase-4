#!/usr/bin/env python3
"""Demo script for calibrated retrieval (Tiny Bite #1).

This demonstrates the per-lane calibrated retrieval with α-weighted fusion.
Uses ontology data (NOT FactoidWiki).

Usage:
    # Without calibration (baseline)
    python tools/demo_calibrated_retrieval.py \\
        --npz artifacts/ontology_13k.npz \\
        --index artifacts/ontology_13k_ivf_flat_ip.index \\
        --query "software ontology" \\
        --alpha 0.2

    # With calibration (after training)
    python tools/demo_calibrated_retrieval.py \\
        --npz artifacts/ontology_13k.npz \\
        --index artifacts/ontology_13k_ivf_flat_ip.index \\
        --query "software ontology" \\
        --alpha 0.2 \\
        --calibrators artifacts/calibrators/
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend
from src.lvm.calibrated_retriever import CalibratedRetriever


def main():
    parser = argparse.ArgumentParser(description="Demo calibrated retrieval")
    parser.add_argument("--npz", required=True, help="Path to ontology NPZ (NOT fw10k!)")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--query", default="software ontology", help="Query text")
    parser.add_argument("--alpha", type=float, default=0.2, help="TMD fusion weight")
    parser.add_argument("--calibrators", help="Path to calibrators directory (optional)")
    parser.add_argument("--k", type=int, default=8, help="Top-K results")
    args = parser.parse_args()

    print(f"\n=== Calibrated Retrieval Demo ===")
    print(f"NPZ: {args.npz}")
    print(f"Index: {args.index}")
    print(f"Query: \"{args.query}\"")
    print(f"Alpha: {args.alpha}")
    print(f"Calibrators: {args.calibrators or 'None (uncalibrated)'}")
    print(f"Top-K: {args.k}\n")

    # Load FAISS index
    print("Loading FAISS index...")
    faiss_db = FaissDB(index_path=args.index, meta_npz_path=args.npz)
    faiss_db.load()

    # Load embedding backend
    print("Loading GTR-T5 embedding backend...")
    embedding_backend = EmbeddingBackend()

    # Initialize calibrated retriever
    print("Initializing calibrated retriever...")
    retriever = CalibratedRetriever(
        faiss_db=faiss_db,
        embedding_backend=embedding_backend,
        npz_path=args.npz,
        alpha=args.alpha,
    )

    # Load calibrators if provided
    if args.calibrators:
        print(f"Loading calibrators from {args.calibrators}...")
        retriever.load_calibrators(args.calibrators)

    # Simple TMD bits for demo (you'd extract this from LLM in production)
    # Using domain=15 (Technology) as a reasonable default for software queries
    tmd_bits = (15, 0, 0)  # (domain, task, modifier)
    tmd_dense = np.zeros(16, dtype=np.float32)
    tmd_dense[0] = 15  # domain

    print(f"\nUsing TMD bits: {tmd_bits} (domain=Technology)")
    print(f"\nRetrieving with α={args.alpha} fusion...")

    # Retrieve
    result = retriever.retrieve(
        concept_text=args.query,
        tmd_bits=tmd_bits,
        tmd_dense=tmd_dense,
        k=args.k,
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"RETRIEVAL RESULTS (FOUND={result.FOUND})")
    print(f"{'='*80}\n")

    print(f"Concept: \"{result.concept_text}\"")
    print(f"TMD Lane: {result.tmd_lane}")
    print(f"Alpha Used: {result.alpha_used}")
    print(f"Accepted Candidates: {len(result.accepted_candidates)}/{len(result.candidates)}\n")

    # Print all candidates
    print("Top-K Candidates:")
    print(f"{'Rank':<6} {'Score':<10} {'CalibProb':<12} {'Accept':<8} {'Concept Text':<50}")
    print("-" * 90)

    for rank, cand in enumerate(result.candidates, 1):
        accept_mark = "✓" if cand.accepted else "✗"
        concept_preview = (cand.concept_text[:47] + "...") if len(cand.concept_text) > 50 else cand.concept_text

        print(f"{rank:<6} {cand.raw_score:<10.4f} {cand.calibrated_prob:<12.4f} {accept_mark:<8} {concept_preview:<50}")

    # Print accepted candidates summary
    if result.accepted_candidates:
        print(f"\n{'='*80}")
        print(f"ACCEPTED CANDIDATES (threshold τ={retriever.tau_lanes.get(result.tmd_lane, retriever.default_tau):.3f})")
        print(f"{'='*80}\n")

        for cand in result.accepted_candidates:
            print(f"✓ {cand.concept_text}")
            print(f"  Score: {cand.raw_score:.4f} → Calibrated P(match): {cand.calibrated_prob:.4f}\n")
    else:
        print(f"\n⚠️  No candidates exceeded threshold τ={retriever.tau_lanes.get(result.tmd_lane, retriever.default_tau):.3f}")
        print(f"   This concept would trigger create-on-miss (Stage 2b in PRD)\n")

    print(f"\n{'='*80}")
    print(f"Demo complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
