#!/usr/bin/env python3
"""Create validation queries from ontology data for α-tuning.

This generates self-retrieval queries: each concept becomes both query and target.
Strategy: Sample concepts uniformly across TMD domains for balanced evaluation.

Usage:
    python tools/create_validation_queries.py \\
        --npz artifacts/ontology_13k.npz \\
        --output eval/validation_queries.jsonl \\
        --n 200 \\
        --seed 42
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def create_validation_queries(
    npz_path: str,
    n_queries: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Create validation queries by sampling concepts across TMD domains.

    Args:
        npz_path: Path to corpus NPZ
        n_queries: Number of validation queries to generate
        seed: Random seed for reproducibility

    Returns:
        List of query dictionaries: {"query": str, "target_id": str}
    """
    # Load corpus
    npz = np.load(npz_path, allow_pickle=True)
    concept_texts = [str(x) for x in npz.get("concept_texts", [])]
    cpe_ids = [str(x) for x in npz.get("cpe_ids", [])]
    tmd_dense = np.asarray(npz.get("tmd_dense"), dtype=np.float32)

    if len(concept_texts) != len(cpe_ids):
        raise ValueError(f"Mismatch: {len(concept_texts)} concepts vs {len(cpe_ids)} IDs")

    # Extract TMD domains (first component of dense vector)
    domains = tmd_dense[:, 0].astype(int)

    # Group concepts by domain
    domain_to_indices = {}
    for idx, domain in enumerate(domains):
        if domain not in domain_to_indices:
            domain_to_indices[domain] = []
        domain_to_indices[domain].append(idx)

    # Sample uniformly across domains
    rng = np.random.RandomState(seed)
    queries = []

    # Calculate per-domain quota
    n_domains = len(domain_to_indices)
    per_domain = n_queries // n_domains
    remainder = n_queries % n_domains

    for domain_id, indices in sorted(domain_to_indices.items()):
        # Sample from this domain
        quota = per_domain + (1 if len(queries) < remainder else 0)
        n_sample = min(quota, len(indices))

        sampled_indices = rng.choice(indices, size=n_sample, replace=False)

        for idx in sampled_indices:
            queries.append({
                "query": concept_texts[idx],
                "target_id": cpe_ids[idx],
                "domain": int(domain_id),
            })

    # Shuffle final list
    rng.shuffle(queries)

    return queries[:n_queries]


def main():
    parser = argparse.ArgumentParser(description="Create validation queries from ontology data")
    parser.add_argument("--npz", required=True, help="Path to corpus NPZ")
    parser.add_argument("--output", default="eval/validation_queries.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=200, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"VALIDATION QUERY GENERATION")
    print(f"{'='*80}\n")
    print(f"NPZ: {args.npz}")
    print(f"Output: {args.output}")
    print(f"N queries: {args.n}")
    print(f"Seed: {args.seed}\n")

    # Create queries
    queries = create_validation_queries(
        npz_path=args.npz,
        n_queries=args.n,
        seed=args.seed,
    )

    # Analyze domain distribution
    domain_counts = {}
    for q in queries:
        domain = q["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"Generated {len(queries)} validation queries")
    print(f"\nDomain distribution:")
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        pct = 100.0 * count / len(queries)
        print(f"  Domain {domain:2d}: {count:3d} queries ({pct:5.1f}%)")

    # Save queries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")

    print(f"\n✓ Saved {len(queries)} queries to {output_path}\n")


if __name__ == "__main__":
    main()
