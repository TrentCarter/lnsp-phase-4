"""Hop reduction benchmark for the synthetic vecRAG pipeline."""

from __future__ import annotations

import numpy as np

from tests.helpers import neo4j_expand, sample_queries, vecrag_search


def run_benchmark(num_queries: int = 100) -> tuple[float, float, float]:
    """Execute the hop reduction benchmark.

    Returns
    -------
    tuple[float, float, float]
        Baseline mean hops, shortcut mean hops, and fractional reduction.
    """
    queries = sample_queries(num_queries)
    baseline_hops: list[float] = []
    shortcut_hops: list[float] = []

    for query in queries:
        seeds = vecrag_search(query, top_k=3)
        if not seeds:
            continue

        baseline = neo4j_expand(seeds, max_hops=3, use_shortcuts=False)
        shortcuts = neo4j_expand(seeds, max_hops=3, use_shortcuts=True)
        if baseline:
            baseline_hops.append(float(np.mean([item["hops"] for item in baseline])))
        if shortcuts:
            shortcut_hops.append(float(np.mean([item["hops"] for item in shortcuts])))

    if not baseline_hops or not shortcut_hops:
        raise RuntimeError("Benchmark did not produce hop measurements")

    baseline_mean = float(np.mean(baseline_hops))
    shortcut_mean = float(np.mean(shortcut_hops))
    reduction = (baseline_mean - shortcut_mean) / baseline_mean

    print(f"Baseline hops: {baseline_mean:.2f} ± {np.std(baseline_hops):.2f}")
    print(f"Shortcut hops: {shortcut_mean:.2f} ± {np.std(shortcut_hops):.2f}")
    print(f"Reduction: {reduction * 100:.1f}%")

    assert reduction > 0.40
    return baseline_mean, shortcut_mean, reduction


if __name__ == "__main__":
    run_benchmark()
