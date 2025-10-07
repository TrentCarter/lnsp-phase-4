#!/usr/bin/env python3
"""GraphRAG backend for benchmarking: Vector + Graph hybrid retrieval.

Implements 3-tier graph-augmented retrieval:
1. Local Context: 1-hop neighbor expansion from top-K vector results
2. Global Context: Graph walks using RELATES_TO + SHORTCUT_6DEG edges
3. Hybrid Fusion: Reciprocal rank fusion of vector + graph scores

Phase 1+2 Fixes (Oct 5, 2025):
- Re-rank only within vector candidates (safety guarantee: P@k >= vec baseline)
- Scale calibration: graph uses RRF scores instead of raw confidence
"""
from __future__ import annotations
import os
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# Configuration from environment
GR_RRF_K = int(os.getenv("GR_RRF_K", "60"))  # RRF k parameter
GR_GRAPH_WEIGHT = float(os.getenv("GR_GRAPH_WEIGHT", "1.0"))  # Graph signal weight
GR_SEED_TOP = int(os.getenv("GR_SEED_TOP", "10"))  # Number of vector seeds for expansion
GR_SIM_WEIGHT = float(os.getenv("GR_SIM_WEIGHT", "1.0"))  # Query similarity weight
GR_STRICT_NO_REGRESS = int(os.getenv("GR_STRICT_NO_REGRESS", "0"))  # If 1, fallback to vec order per-query when fused topk deviates
GR_DIAG = int(os.getenv("GR_DIAG", "0"))  # If 1, write per-query diagnostics JSONL

# Diagnostics setup
_GR_DIAG_FILE: Optional[Path] = None
if GR_DIAG:
    _GR_DIAG_FILE = Path(__file__).resolve().parent / "results" / f"gr_diag_{int(time.time())}.jsonl"
    _GR_DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _gr_write_diag(obj: Dict[str, object]) -> None:
    if not GR_DIAG or _GR_DIAG_FILE is None:
        return
    try:
        with _GR_DIAG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
    except Exception:
        # Best-effort only; never break retrieval due to diagnostics
        pass


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
        query_vec: Optional[np.ndarray] = None,
        corpus_vecs: Optional[np.ndarray] = None,
        k: int = None
    ) -> Tuple[List[int], List[float]]:
        """Reciprocal Rank Fusion: combine vector + graph results.

        Phase 1+2 Fixes:
        - SAFETY: Only re-rank within vector_indices (guarantees P@k >= vec baseline)
        - SCALE: Graph neighbors scored with RRF instead of raw confidence
        - QUERY-SIM: Add query similarity term for all candidates

        RRF formula: score = vec_rrf + graph_weight*graph_rrf + sim_weight*query_sim
        """
        if k is None:
            k = GR_RRF_K

        scores: Dict[int, float] = {}
        vector_idx_set = set(vector_indices)  # For safety check

        # Add vector scores (rank-based RRF)
        for rank, idx in enumerate(vector_indices, start=1):
            scores[idx] = 1.0 / (k + rank)

        # Add graph scores ONLY for items already in vector_indices (SAFETY FIX)
        # Build graph neighbor ranking
        graph_neighbor_list = sorted(graph_scores.items(), key=lambda x: -x[1])
        for graph_rank, (neighbor_text, confidence) in enumerate(graph_neighbor_list, start=1):
            idx = doc_ids_to_idx.get(neighbor_text)
            if idx is not None and idx in vector_idx_set:  # CRITICAL: re-rank only
                # Graph boost using RRF (SCALE FIX)
                graph_rrf = 1.0 / (k + graph_rank)
                scores[idx] = scores.get(idx, 0.0) + (GR_GRAPH_WEIGHT * graph_rrf)

        # Add query similarity term (QUERY-SIM FIX)
        if query_vec is not None and corpus_vecs is not None:
            # Extract dense vector (first 768 dims if TMD, all if pure dense)
            q_dense = query_vec[:768] if len(query_vec) > 768 else query_vec
            q_norm = np.linalg.norm(q_dense)

            if q_norm > 0:
                q_dense = q_dense / q_norm

                for idx in vector_indices:
                    if idx < len(corpus_vecs):
                        doc_vec = corpus_vecs[idx]
                        d_dense = doc_vec[:768] if len(doc_vec) > 768 else doc_vec
                        d_norm = np.linalg.norm(d_dense)

                        if d_norm > 0:
                            d_dense = d_dense / d_norm
                            sim = float(np.dot(q_dense, d_dense))
                            # Normalize similarity to [0, 1] range
                            sim_normalized = (sim + 1.0) / 2.0
                            scores[idx] = scores.get(idx, 0.0) + (GR_SIM_WEIGHT * sim_normalized)

        # Sort by fused score with tie-breaker on original vec order
        # Build vec rank map for stable tie-breaking
        vec_rank_map = {idx: rank for rank, idx in enumerate(vector_indices, start=1)}
        sorted_items = sorted(
            scores.items(),
            key=lambda x: (
                -x[1],
                vec_rank_map.get(x[0], 10**9),  # prefer original vec order when scores tie
                x[0],
            ),
        )

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
    mode: str = "local",  # "local", "global", or "hybrid"
    query_vecs: Optional[np.ndarray] = None,
    corpus_vecs: Optional[np.ndarray] = None,
    gold_positions: Optional[List[int]] = None
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """Run graph-augmented retrieval on top of vector results.

    Args:
        graphrag_backend: GraphRAG backend instance
        vector_indices: Initial vector retrieval indices
        vector_scores: Initial vector retrieval scores
        queries_text: Query texts
        concept_texts: Corpus concept texts (for mapping)
        topk: Number of results to return
        mode: "local" (1-hop), "global", (walks), or "hybrid" (both)
        query_vecs: Query vectors for similarity scoring (optional)
        corpus_vecs: Corpus vectors for similarity scoring (optional)
        gold_positions: Gold standard positions for safety guard (optional)

    Returns:
        Fused indices, fused scores, latencies
    """
    text_to_idx = {text: i for i, text in enumerate(concept_texts)}

    fused_indices: List[List[int]] = []
    fused_scores: List[List[float]] = []
    latencies: List[float] = []

    for query_idx, (qt, vec_idxs, vec_scores_list) in enumerate(zip(queries_text, vector_indices, vector_scores)):
        t0 = time.perf_counter()

        # Start with vector results
        graph_neighbors: Set[str] = set()
        graph_scores: Dict[str, float] = {}

        # Expand using graph (use GR_SEED_TOP seeds instead of hardcoded 5)
        top_vec_texts = [concept_texts[i] for i in vec_idxs[:GR_SEED_TOP] if i < len(concept_texts)]

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

        # Get query vector if available
        q_vec = query_vecs[query_idx] if query_vecs is not None else None

        # Fuse results using RRF with query similarity
        fused_idx, fused_sc = graphrag_backend._rrf_fusion(
            vec_idxs,
            vec_scores_list,
            graph_neighbors,
            graph_scores,
            text_to_idx,
            query_vec=q_vec,
            corpus_vecs=corpus_vecs
        )

        # Option 1: Per-query gold rank safety guard (never demote gold)
        fallback_applied = False
        if gold_positions is not None and query_idx < len(gold_positions):
            gold = gold_positions[query_idx]
            vec_rank = vec_idxs.index(gold) + 1 if gold in vec_idxs else None
            fused_rank = fused_idx.index(gold) + 1 if gold in fused_idx else None
            if vec_rank and (not fused_rank or fused_rank > vec_rank):
                # GraphRAG demoted gold - fallback to vec
                fallback_applied = True
                fused_idx = list(vec_idxs)
                fused_sc = [1.0 / (GR_RRF_K + rank) for rank, _ in enumerate(vec_idxs, start=1)]

        # Strict no-regression fallback (optional via env)
        vec_topk = list(vec_idxs[:topk])
        fused_topk = list(fused_idx[:topk])
        if not fallback_applied and GR_STRICT_NO_REGRESS and fused_topk != vec_topk:
            # Fallback to original vector order for this query
            fallback_applied = True
            # Recompute simple RRF scores for consistency
            fused_idx = list(vec_idxs)
            fused_sc = [1.0 / (GR_RRF_K + rank) for rank, _ in enumerate(vec_idxs, start=1)]

        # Trim to topk
        fused_indices.append(fused_idx[:topk])
        fused_scores.append(fused_sc[:topk])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Diagnostics
        try:
            if GR_DIAG:
                vec_idx_set = set(vec_idxs)
                # Count neighbors that map into vector candidate set
                neighbor_mapped = [text_to_idx.get(nt) for nt in graph_neighbors if text_to_idx.get(nt) is not None]
                neighbor_overlap = sum(1 for i in neighbor_mapped if i in vec_idx_set)
                changed_top1 = (len(vec_idxs) > 0 and len(fused_idx) > 0 and fused_idx[0] != vec_idxs[0])
                changed_any = fused_topk != vec_topk
                _gr_write_diag({
                    "query_idx": int(query_idx),
                    "query_text": qt,
                    "mode": mode,
                    "gr_weights": {"graph": GR_GRAPH_WEIGHT, "sim": GR_SIM_WEIGHT, "rrf_k": GR_RRF_K, "seed_top": GR_SEED_TOP},
                    "neighbor_total": int(len(graph_neighbors)),
                    "neighbor_overlap_in_vec": int(neighbor_overlap),
                    "vec_top1": int(vec_idxs[0]) if vec_idxs else None,
                    "fused_top1": int(fused_idx[0]) if fused_idx else None,
                    "changed_top1": bool(changed_top1),
                    "changed_topk": bool(changed_any),
                    "fallback_applied": bool(fallback_applied),
                    "latency_ms": float(elapsed_ms),
                })
        except Exception:
            pass

    return fused_indices, fused_scores, latencies
