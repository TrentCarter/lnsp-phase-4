#!/usr/bin/env python3
"""GraphRAG backend for benchmarking: Vector + Graph hybrid retrieval.

Implements 3-tier graph-augmented retrieval:
1. Local Context: 1-hop neighbor expansion from top-K vector results
2. Global Context: Graph walks using RELATES_TO + SHORTCUT_6DEG edges
3. Hybrid Fusion: Reciprocal rank fusion of vector + graph scores
"""
from __future__ import annotations
import time
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class GraphRAGBackend:
    """Graph-augmented vector retrieval using Neo4j."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j driver not installed (pip install neo4j)")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Build concept_text → index mapping from Neo4j
        self.text_to_idx: Dict[str, int] = {}
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE c.text IS NOT NULL
                RETURN c.text as text, c.cpe_id as cpe_id
            """)
            for i, record in enumerate(result):
                self.text_to_idx[record["text"]] = i

    def close(self):
        self.driver.close()

    def _get_1hop_neighbors(self, concept_text: str) -> List[Tuple[str, float]]:
        """Get 1-hop neighbors with confidence scores from Neo4j.

        Tries multiple strategies:
        1. Direct Concept->Concept edges
        2. Concept->Entity->Concept paths (2-hop)
        3. Uses confidence scores when available
        """
        with self.driver.session() as session:
            # Strategy 1: Direct Concept->Concept edges
            result = session.run("""
                MATCH (c:Concept {text: $text})-[r:RELATES_TO]->(neighbor:Concept)
                WHERE neighbor.text IS NOT NULL AND neighbor.text <> $text
                RETURN DISTINCT neighbor.text as neighbor_text,
                       coalesce(r.confidence, 0.5) as confidence
                ORDER BY confidence DESC
                LIMIT 15
            """, text=concept_text)

            neighbors = []
            for record in result:
                neighbors.append((record["neighbor_text"], float(record["confidence"])))

            # Strategy 2: 2-hop paths via Entity nodes (if not enough direct neighbors)
            if len(neighbors) < 5:
                result = session.run("""
                    MATCH (c:Concept {text: $text})-[r1:RELATES_TO]->(e:Entity)
                          -[r2:RELATES_TO]-(neighbor:Concept)
                    WHERE neighbor.text IS NOT NULL AND neighbor.text <> $text
                    RETURN DISTINCT neighbor.text as neighbor_text,
                           (coalesce(r1.confidence, 0.5) * coalesce(r2.confidence, 0.5)) as confidence
                    ORDER BY confidence DESC
                    LIMIT 10
                """, text=concept_text)

                for record in result:
                    neighbors.append((record["neighbor_text"], float(record["confidence"]) * 0.7))  # Discount 2-hop

            return neighbors[:20]  # Max 20 neighbors total

    def _get_graph_walks(self, concept_text: str, max_length: int = 3) -> List[Tuple[str, float]]:
        """Get graph walk sequences using SHORTCUT_6DEG + RELATES_TO edges."""
        with self.driver.session() as session:
            # Use shortcuts for longer paths
            result = session.run("""
                MATCH path = (c:Concept {text: $text})-[:SHORTCUT_6DEG*1..2]-(neighbor:Concept)
                WHERE neighbor.text IS NOT NULL AND neighbor.text <> $text
                RETURN DISTINCT neighbor.text as neighbor_text, length(path) as path_len
                ORDER BY path_len ASC
                LIMIT 15
            """, text=concept_text)

            walks = []
            for record in result:
                # Decay score by path length
                score = 0.8 ** record["path_len"]
                walks.append((record["neighbor_text"], score))

            return walks

    def _rrf_fusion(
        self,
        vector_indices: List[int],
        vector_scores: List[float],
        graph_neighbors: Set[str],
        graph_scores: Dict[str, float],
        doc_ids_to_idx: Dict[str, int],
        k: int = 60
    ) -> Tuple[List[int], List[float]]:
        """Reciprocal Rank Fusion: combine vector + graph results.

        RRF formula: score = sum(1/(k + rank_i)) for all sources
        k=60 is standard from literature
        """
        scores: Dict[int, float] = {}

        # Add vector scores (rank-based)
        for rank, (idx, vec_score) in enumerate(zip(vector_indices, vector_scores), start=1):
            scores[idx] = scores.get(idx, 0.0) + (1.0 / (k + rank))

        # Add graph scores (confidence-weighted)
        for neighbor_text, graph_score in graph_scores.items():
            idx = doc_ids_to_idx.get(neighbor_text)
            if idx is not None:
                # Boost by graph confidence
                scores[idx] = scores.get(idx, 0.0) + (graph_score * 0.5)

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        fused_indices = [idx for idx, _ in sorted_items]
        fused_scores = [score for _, score in sorted_items]

        return fused_indices, fused_scores


def run_graphrag(
    graphrag_backend: GraphRAGBackend,
    vector_indices: List[List[int]],
    vector_scores: List[List[float]],
    queries_text: List[str],
    concept_texts: List[str],
    topk: int,
    mode: str = "local"  # "local", "global", or "hybrid"
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """Run graph-augmented retrieval on top of vector results.

    Args:
        graphrag_backend: GraphRAG backend instance
        vector_indices: Initial vector retrieval indices
        vector_scores: Initial vector retrieval scores
        queries_text: Query texts
        concept_texts: Corpus concept texts (for mapping)
        topk: Number of results to return
        mode: "local" (1-hop), "global" (walks), or "hybrid" (both)

    Returns:
        Fused indices, fused scores, latencies
    """
    text_to_idx = {text: i for i, text in enumerate(concept_texts)}

    fused_indices: List[List[int]] = []
    fused_scores: List[List[float]] = []
    latencies: List[float] = []

    for query_idx, (qt, vec_idxs, vec_scores) in enumerate(zip(queries_text, vector_indices, vector_scores)):
        t0 = time.perf_counter()

        # Start with vector results
        graph_neighbors: Set[str] = set()
        graph_scores: Dict[str, float] = {}

        # Expand using graph
        top_vec_texts = [concept_texts[i] for i in vec_idxs[:5] if i < len(concept_texts)]

        for concept_text in top_vec_texts:
            if mode in ("local", "hybrid"):
                # Get 1-hop neighbors
                neighbors = graphrag_backend._get_1hop_neighbors(concept_text)
                for neighbor_text, confidence in neighbors:
                    graph_neighbors.add(neighbor_text)
                    # Keep max confidence if duplicate
                    graph_scores[neighbor_text] = max(
                        graph_scores.get(neighbor_text, 0.0),
                        confidence
                    )

            if mode in ("global", "hybrid"):
                # Get graph walk sequences
                walks = graphrag_backend._get_graph_walks(concept_text, max_length=3)
                for neighbor_text, walk_score in walks:
                    graph_neighbors.add(neighbor_text)
                    graph_scores[neighbor_text] = max(
                        graph_scores.get(neighbor_text, 0.0),
                        walk_score
                    )

        # Fuse results using RRF
        fused_idx, fused_sc = graphrag_backend._rrf_fusion(
            vec_idxs,
            vec_scores,
            graph_neighbors,
            graph_scores,
            text_to_idx,
            k=60
        )

        # Trim to topk
        fused_indices.append(fused_idx[:topk])
        fused_scores.append(fused_sc[:topk])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return fused_indices, fused_scores, latencies
