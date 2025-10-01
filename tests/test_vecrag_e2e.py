"""End-to-end regression tests for the synthetic vecRAG pipeline."""

from __future__ import annotations

import numpy as np

from tests.helpers import neo4j_expand, vecrag_search


class TestVecRAGPipeline:
    """Exercise query → FAISS shortlist → graph expansion behaviour."""

    def test_basic_retrieval(self) -> None:
        results = vecrag_search("BLAST is analysis software", top_k=10)

        assert len(results) == 10
        assert results[0]["similarity"] > 0.6
        assert all("cpe_id" in r for r in results)
        assert all("concept_text" in r for r in results)

    def test_graph_walk_without_shortcuts(self) -> None:
        seeds = vecrag_search("sequence alignment algorithm", top_k=3)
        expanded = neo4j_expand(seeds, max_hops=3, use_shortcuts=False)

        assert len(expanded) > len(seeds)
        hop_counts = [item["hops_from_seed"] for item in expanded]
        assert np.mean(hop_counts) > 5.0

    def test_graph_walk_with_shortcuts(self) -> None:
        seeds = vecrag_search("sequence alignment algorithm", top_k=3)
        expanded = neo4j_expand(seeds, max_hops=3, use_shortcuts=True)

        assert len(expanded) > len(seeds)
        hop_counts = [item["hops_from_seed"] for item in expanded]
        assert np.mean(hop_counts) < 3.0

    def test_tmd_filtering(self) -> None:
        raw = vecrag_search("bioinformatics software tool", top_k=6)
        lanes = {item["tmd_lane"] for item in raw}
        assert 67890 in lanes  # Shadow entry proves cross-lane mixing
        assert any(item["cpe_id"] == "CPE:SHADOW" for item in raw)

        filtered = vecrag_search("bioinformatics software tool", top_k=6, tmd_lane=12345)
        assert filtered
        assert all(item["tmd_lane"] == 12345 for item in filtered)
        assert all(item["cpe_id"] != "CPE:SHADOW" for item in filtered)

    def test_shortcuts_reduce_hops_per_target(self) -> None:
        seeds = vecrag_search("sequence alignment algorithm", top_k=2)
        baseline = neo4j_expand(seeds, max_hops=3, use_shortcuts=False)
        shortcuts = neo4j_expand(seeds, max_hops=3, use_shortcuts=True)

        shortcut_map = {
            (entry["seed_cpe_id"], entry["cpe_id"]): entry["hops"]
            for entry in shortcuts
        }

        assert shortcut_map, "Expected shortcut expansions to be present"

        for entry in baseline:
            key = (entry["seed_cpe_id"], entry["cpe_id"])
            assert key in shortcut_map, "Shortcut expansion missing baseline target"
            assert shortcut_map[key] <= entry["hops"]

    def test_unknown_query_returns_empty(self) -> None:
        assert vecrag_search("nonexistent query", top_k=5) == []
