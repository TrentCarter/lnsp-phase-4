#!/usr/bin/env python3
"""
LightRAG-Style Graph Retrieval
================================

CORRECT approach (matches LightRAG paper):
1. Extract entities/topics from QUERY
2. Vector match query concepts to graph nodes
3. Traverse graph from matched nodes (1-hop neighbors)
4. Re-rank by distance from query vector

WRONG approach (what we were doing):
1. vecRAG gets top-5 results
2. Expand those results
3. ❌ If top-5 are wrong, expansion makes it worse
"""

from __future__ import annotations
import os
import time
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class LightRAGStyleRetriever:
    """Graph retrieval following LightRAG's dual-level approach."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j driver not installed")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    def _extract_query_concepts(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        *,
        corpus_vecs: Optional[np.ndarray] = None,
        concept_texts: Optional[List[str]] = None,
        prefer_neo4j_vectors: Optional[bool] = None,
    ) -> List[str]:
        """
        Extract key concepts from query by matching query vector to corpus vectors.

        This is the KEY difference from our old approach:
        - OLD: Expand from vecRAG top-5 results (wrong seeds 45% of time)
        - NEW: Match QUERY to concepts directly (always correct seed)

        FIXED: Use corpus_vectors from NPZ instead of Neo4j (which lacks vectors)

        Args:
            query_text: Query string
            query_vector: Query embedding vector (768-dim)
            corpus_vectors: All concept vectors from NPZ (N x 768)
            concept_texts: All corpus concept texts
            top_k: How many query concepts to extract
            prefer_neo4j_vectors: Decide preference from env if not specified
        """
        # Decide preference from env if not specified
        if prefer_neo4j_vectors is None:
            prefer_neo4j_vectors = bool(int(os.getenv("LRAG_USE_NEO4J_VECTORS", "0")))

        # Prepare query vector (strip TMD dims if present)
        query_vec_clean = query_vector[:768] if len(query_vector) > 768 else query_vector
        qn = np.linalg.norm(query_vec_clean)
        if qn > 0:
            query_vec_clean = query_vec_clean / qn

        # Try Neo4j-based concept similarity if preferred
        if prefer_neo4j_vectors:
            try:
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (c:Concept)
                        WHERE c.vector IS NOT NULL AND c.text IS NOT NULL
                        RETURN c.text as text, c.vector as vector
                        """
                    )
                    concepts = list(result)
                    if concepts:
                        similarity = []
                        for concept in concepts:
                            vec = concept["vector"]
                            if isinstance(vec, str):
                                import json as _json
                                vec = _json.loads(vec)
                            cvec = np.asarray(vec, dtype=np.float32)
                            cn = np.linalg.norm(cvec)
                            if cn > 0:
                                cvec = cvec / cn
                                similarity.append((concept["text"], float(np.dot(query_vec_clean, cvec))))
                        similarity.sort(key=lambda x: -x[1])
                        return [t for t, _ in similarity[:top_k]]
            except Exception:
                # Fall through to NPZ-based seeding
                pass

        # NPZ-based seeding from provided corpus vectors
        if corpus_vecs is not None and concept_texts is not None and len(corpus_vecs) == len(concept_texts):
            # Use first 768 dims to match dense encoder; normalize
            cv = corpus_vecs
            if cv.shape[1] > 768:
                cv = cv[:, :768]
            norms = np.linalg.norm(cv, axis=1, keepdims=True)
            safe = norms.squeeze(-1) > 0
            cvn = np.zeros_like(cv)
            cvn[safe] = cv[safe] / norms[safe].reshape(-1, 1)
            sims = (cvn @ query_vec_clean.reshape(-1, 1)).reshape(-1)
            top_idx = np.argsort(-sims)[:top_k]
            return [concept_texts[i] for i in top_idx]

        # No seeding possible
        return []

    def _get_graph_neighborhood(self, concept_text: str, hops: int = 1) -> List[Tuple[str, float]]:
        """
        Get neighborhood of a concept in graph.

        Args:
            concept_text: Starting concept
            hops: How many hops to traverse (default 1)

        Returns:
            List of (neighbor_text, confidence) tuples
        """
        with self.driver.session() as session:
            if hops == 1:
                result = session.run(
                    """
                    MATCH (c:Concept {text: $text})-[r:RELATES_TO]-(neighbor:Concept)
                    WHERE neighbor.text IS NOT NULL AND neighbor.text <> $text
                    RETURN DISTINCT neighbor.text as neighbor_text, coalesce(r.confidence, 0.5) as confidence
                    ORDER BY confidence DESC LIMIT 20
                    """,
                    {"text": concept_text},
                )
                neighbors = [
                    (record["neighbor_text"], float(record["confidence"])) for record in result
                ]
            else:
                # Simple multi-hop not implemented; keep behavior explicit
                neighbors = []

        return neighbors


    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        concept_texts: List[str],
        text_to_idx: Dict[str, int],
        topk: int = 10,
        *,
        corpus_vecs: Optional[np.ndarray] = None,
        prefer_neo4j_vectors: Optional[bool] = None,
    ) -> Tuple[List[int], List[float], float]:
        """
        LightRAG-style retrieval.
        Steps:
          1) Seed from query→concept similarity (Neo4j vectors if present, else NPZ corpus vectors).
          2) Expand neighbors via RELATES_TO.
          3) Score by alpha*cos_sim(query, neighbor) + (1-alpha)*graph_confidence; fallback to NPZ top concepts if empty.
        """
        t0 = time.perf_counter()

        # Step 1: Extract query concepts (entities/topics from query)
        seed_k = int(os.getenv("LRAG_SEED_K", "3"))
        query_concepts = self._extract_query_concepts(
            query_text,
            query_vector,
            top_k=seed_k,
            corpus_vecs=corpus_vecs,
            concept_texts=concept_texts,
            prefer_neo4j_vectors=prefer_neo4j_vectors,
        )
        if not query_concepts:
            # Fallback: no concepts found
            return [], [], (time.perf_counter() - t0) * 1000.0
        # Step 2: Traverse graph from query concepts
        graph_neighbors: Set[str] = set()
        graph_scores: Dict[str, float] = {}

        for concept_text in query_concepts:
            neighbors = self._get_graph_neighborhood(concept_text, hops=1)

            for neighbor_text, confidence in neighbors:
                graph_neighbors.add(neighbor_text)
                # Keep max confidence if same neighbor from multiple query concepts
                graph_scores[neighbor_text] = max(
                    graph_scores.get(neighbor_text, 0.0),
                    confidence
                )

        # Step 3+4: Map to corpus indices and re-score by cosine similarity to query
        # Prepare 768-dim normalized query vector for cosine
        q_clean = query_vector[:768] if len(query_vector) > 768 else query_vector
        qn = float(np.linalg.norm(q_clean))
        if qn > 0:
            q_clean = q_clean / qn

        alpha = float(os.getenv("LRAG_SIM_ALPHA", "0.7"))
        scored_results: List[Tuple[int, float]] = []
        for neighbor_text, graph_score in graph_scores.items():
            idx = text_to_idx.get(neighbor_text)
            if idx is None:
                continue
            score = float(graph_score)
            if corpus_vecs is not None and 0 <= idx < len(concept_texts):
                # Use first 768 dims for cosine against query
                nv = corpus_vecs[idx]
                nv = nv[:768] if nv.shape[0] > 768 else nv
                nn = float(np.linalg.norm(nv))
                if nn > 0:
                    cos = float(np.dot(q_clean, nv / nn))
                    score = alpha * cos + (1.0 - alpha) * float(graph_score)
            scored_results.append((idx, score))

        # Sort by score descending; stable tie-break on index
        scored_results.sort(key=lambda x: (-x[1], x[0]))

        # If graph produced no mapped hits, optionally fall back to NPZ top concepts
        if not scored_results and corpus_vecs is not None:
            # Reuse seeding top concepts directly as results
            qc = self._extract_query_concepts(
                query_text,
                query_vector,
                top_k=topk,
                corpus_vecs=corpus_vecs,
                concept_texts=concept_texts,
                prefer_neo4j_vectors=False,
            )
            scored_results = [(text_to_idx[t], 1.0) for t in qc if t in text_to_idx]

        # Return top-k
        indices = [idx for idx, _ in scored_results[:topk]]
        scores = [score for _, score in scored_results[:topk]]
        latency = (time.perf_counter() - t0) * 1000.0

        return indices, scores, latency


def run_lightrag_style(
    queries_text: List[str],
    query_vectors: np.ndarray,
    concept_texts: List[str],
    topk: int = 10
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """
    Run LightRAG-style retrieval on all queries.

    Args:
        queries_text: List of query strings
        query_vectors: Query embedding matrix (n_queries x dim)
        concept_texts: Corpus concept texts
        topk: Number of results per query

    Returns:
        (all_indices, all_scores, all_latencies)
    """
    retriever = LightRAGStyleRetriever()
    text_to_idx = {text: i for i, text in enumerate(concept_texts)}

    all_indices = []
    all_scores = []
    all_latencies = []

    for i, (qt, qv) in enumerate(zip(queries_text, query_vectors)):
        indices, scores, latency = retriever.search(
            query_text=qt,
            query_vector=qv,
            concept_texts=concept_texts,
            text_to_idx=text_to_idx,
            topk=topk
        )

        all_indices.append(indices)
        all_scores.append(scores)
        all_latencies.append(latency)

    retriever.close()

    return all_indices, all_scores, all_latencies
