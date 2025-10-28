#!/usr/bin/env python3
"""
Audit ground-truth key distributions in evaluation NPZ bundles.

Outputs counts for each (article_index, chunk_index) pair to diagnose label
collapse that can inflate retrieval metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def load_truth_keys(npz_path: Path) -> Iterable[Tuple[int, int]]:
    data = np.load(npz_path, allow_pickle=True)
    if "truth_keys" not in data:
        raise KeyError(f"{npz_path} missing 'truth_keys'")
    keys = data["truth_keys"]
    if keys.ndim != 2 or keys.shape[1] != 2:
        raise ValueError(f"'truth_keys' must have shape [N,2]; got {keys.shape}")
    return [tuple(map(int, pair)) for pair in keys]


def compute_histogram(pairs: Iterable[Tuple[int, int]]) -> Tuple[Counter, int]:
    counter = Counter(pairs)
    total = sum(counter.values())
    return counter, total


def summarize(counter: Counter, total: int, top_k: int) -> dict:
    if total == 0:
        return {
            "total": 0,
            "unique_pairs": 0,
            "top_pairs": [],
            "top_k_coverage": 0.0,
            "entropy": 0.0,
            "gini": 0.0,
        }

    unique_pairs = len(counter)
    top_pairs: List[Tuple[Tuple[int, int], int]] = counter.most_common(top_k)
    coverage = sum(cnt for _, cnt in top_pairs) / total

    # Shannon entropy (natural log)
    probs = np.array([cnt / total for cnt in counter.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    # Gini coefficient for imbalance (0 = uniform, 1 = all mass on one key)
    sorted_counts = np.array(sorted(counter.values()))
    cum_counts = np.cumsum(sorted_counts)
    gini = float(
        (2 * np.sum((np.arange(1, unique_pairs + 1) * sorted_counts)))
        / (unique_pairs * total)
        - (unique_pairs + 1) / unique_pairs
    )

    top_list = [
        {
            "article_index": pair[0],
            "chunk_index": pair[1],
            "count": cnt,
            "fraction": cnt / total,
        }
        for pair, cnt in top_pairs
    ]

    return {
        "total": total,
        "unique_pairs": unique_pairs,
        "top_pairs": top_list,
        "top_k": top_k,
        "top_k_coverage": coverage,
        "entropy": entropy,
        "gini": gini,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True, help="Evaluation NPZ file")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument("--top", type=int, default=20, help="Top-N pairs to report")
    ap.add_argument(
        "--dump-full",
        action="store_true",
        help="Include full histogram in output JSON (may be large).",
    )
    args = ap.parse_args()

    truth_pairs = load_truth_keys(args.npz)
    counter, total = compute_histogram(truth_pairs)
    summary = summarize(counter, total, args.top)

    if args.dump_full:
        summary["histogram"] = [
            {
                "article_index": pair[0],
                "chunk_index": pair[1],
                "count": cnt,
                "fraction": cnt / total,
            }
            for pair, cnt in counter.most_common()
        ]

    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
