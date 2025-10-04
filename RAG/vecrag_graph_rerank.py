#!/usr/bin/env python3
"""
vecRAG + Graph Re-ranker: Use graph to boost/validate vecRAG results.

Strategy:
1. Get top-K results from vecRAG (FAISS)
2. For each result, check graph connectivity/centrality
3. Boost scores based on graph features:
   - Number of connections to other top-K results
   - Graph centrality (PageRank-like)
   - Presence in graph neighborhoods of top results
4. Re-rank based on combined vector + graph scores
"""
from typing import List, Tuple, Dict, Set
import time
import numpy as np

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class VecRAGGraphReranker:
    """Use graph structure to re-rank vecRAG results."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j driver not installed")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_concept_connections(self, concept_texts: List[str]) -> Dict[str, int]:
        """Get number of connections for each concept.

        Returns:
            {concept_text: connection_count}
        """
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $texts as text
                MATCH (c:Concept {text: text})
                OPTIONAL MATCH (c)-[r:RELATES_TO]-(neighbor:Concept)
                RETURN c.text as concept_text, count(DISTINCT neighbor) as connections
            """, texts=concept_texts)

            return {record["concept_text"]: record["connections"] for record in result}

    def get_mutual_connections(self, concept_texts: List[str]) -> Dict[str, int]:
        """Count how many of the top-K results each concept connects to.

        This measures "centrality within the result set" - concepts that connect
        to many other top results get boosted.

        Returns:
            {concept_text: mutual_connection_count}
        """
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $texts as text
                MATCH (c:Concept {text: text})
                OPTIONAL MATCH (c)-[:RELATES_TO]-(neighbor:Concept)
                WHERE neighbor.text IN $texts AND neighbor.text <> text
                RETURN c.text as concept_text, count(DISTINCT neighbor) as mutual_connections
            """, texts=concept_texts)

            return {record["concept_text"]: record["mutual_connections"] for record in result}

    def rerank_with_graph(
        self,
        indices: List[int],
        scores: List[float],
        concept_texts_corpus: List[str],
        boost_factor: float = 0.1
    ) -> Tuple[List[int], List[float]]:
        """Re-rank vecRAG results using graph features.

        Args:
            indices: Original vecRAG result indices
            scores: Original vecRAG scores
            concept_texts_corpus: Full corpus of concept texts
            boost_factor: How much to boost based on graph (0.0-1.0)

        Returns:
            (reranked_indices, reranked_scores)
        """
        # Get concept texts for current results
        result_texts = [concept_texts_corpus[idx] for idx in indices if idx < len(concept_texts_corpus)]

        # Get graph features
        mutual_connections = self.get_mutual_connections(result_texts)

        # Calculate boosted scores
        boosted_scores = []
        for idx, vec_score in zip(indices, scores):
            if idx >= len(concept_texts_corpus):
                boosted_scores.append(vec_score)
                continue

            text = concept_texts_corpus[idx]

            # Graph boost: normalize by max possible connections (len(result_texts) - 1)
            max_mutual = len(result_texts) - 1
            mutual_count = mutual_connections.get(text, 0)
            graph_boost = (mutual_count / max_mutual) if max_mutual > 0 else 0.0

            # Combined score: vec_score + (boost_factor * graph_boost)
            combined_score = vec_score + (boost_factor * graph_boost)
            boosted_scores.append(combined_score)

        # Re-rank by combined scores
        sorted_pairs = sorted(zip(indices, boosted_scores), key=lambda x: -x[1])
        reranked_indices = [idx for idx, _ in sorted_pairs]
        reranked_scores = [score for _, score in sorted_pairs]

        return reranked_indices, reranked_scores


def run_vecrag_graph_rerank(
    db,  # FaissDB instance
    queries: List[np.ndarray],
    concept_texts_corpus: List[str],
    topk: int,
    reranker: VecRAGGraphReranker,
    boost_factor: float = 0.1
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """Run vecRAG with graph-based re-ranking.

    Args:
        db: FaissDB instance
        queries: Query vectors
        concept_texts_corpus: Full corpus concept texts
        topk: Number of results to return
        reranker: VecRAGGraphReranker instance
        boost_factor: How much to boost based on graph connectivity

    Returns:
        (indices, scores, latencies)
    """
    all_indices: List[List[int]] = []
    all_scores: List[List[float]] = []
    latencies: List[float] = []

    for q in queries:
        t0 = time.perf_counter()

        # Step 1: Get vecRAG results
        q = q.reshape(1, -1).astype(np.float32)
        n = float(np.linalg.norm(q))
        q = q / n if n > 0 else q

        # Get more results than needed (2x topk) for re-ranking
        search_k = min(topk * 2, 100)
        D, I = db.search(q, search_k)

        vec_indices = [int(x) for x in I[0]]
        vec_scores = [float(s) for s in D[0]]

        # Step 2: Re-rank using graph
        reranked_indices, reranked_scores = reranker.rerank_with_graph(
            vec_indices,
            vec_scores,
            concept_texts_corpus,
            boost_factor=boost_factor
        )

        # Step 3: Return top-K after re-ranking
        all_indices.append(reranked_indices[:topk])
        all_scores.append(reranked_scores[:topk])

        latencies.append((time.perf_counter() - t0) * 1000.0)

    return all_indices, all_scores, latencies
