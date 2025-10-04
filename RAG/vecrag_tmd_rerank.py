#!/usr/bin/env python3
"""
vecRAG with TMD Re-ranking (TMD-ReRank)

Strategy:
1. Extract 768D semantic embedding from 784D vectors (ignore TMD portion)
2. Search FAISS index using full 784D (gets reasonable initial ranking)
3. Get top-K*2 results (e.g., top-20 for final top-10)
4. Re-rank using TMD alignment:
   - Generate TMD for query text using LLM/pattern-based extractor
   - Extract TMD from each result vector (first 16 dims)
   - Calculate TMD similarity (cosine or dot product)
   - Combine: final_score = alpha * vec_score + (1-alpha) * tmd_score
5. Return top-K after re-ranking

TMD Fields (16 dimensions):
  [0:3]   - entity_type (one-hot: concept/entity/relation)
  [3:6]   - semantic_role (one-hot: subject/predicate/object)
  [6:9]   - context_scope (one-hot: local/domain/global)
  [9:12]  - temporal_aspect (one-hot: static/dynamic/temporal)
  [12:16] - confidence_metrics (4 floats: extraction confidence, relation strength, etc.)
"""
from typing import List, Tuple
import time
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.tmd_extractor_v2 import extract_tmd_from_text as extract_tmd_pattern_based
    HAS_TMD_EXTRACTOR = True
except ImportError:
    HAS_TMD_EXTRACTOR = False

try:
    from src.llm_tmd_extractor import extract_tmd_with_llm
    HAS_LLM_TMD_EXTRACTOR = True
except ImportError:
    HAS_LLM_TMD_EXTRACTOR = False

try:
    from src.utils.tmd import encode_tmd16
    HAS_TMD_ENCODER = True
except ImportError:
    HAS_TMD_ENCODER = False


def compute_tmd_similarity(query_tmd: np.ndarray, result_tmds: np.ndarray) -> np.ndarray:
    """
    Compute TMD similarity between query and multiple results.

    Args:
        query_tmd: (16,) array - TMD from query vector
        result_tmds: (N, 16) array - TMDs from N result vectors

    Returns:
        (N,) array of TMD similarity scores [0, 1]
    """
    # Normalize TMDs
    query_norm = query_tmd / (np.linalg.norm(query_tmd) + 1e-8)
    result_norms = result_tmds / (np.linalg.norm(result_tmds, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    similarities = np.dot(result_norms, query_norm)

    # Clamp to [0, 1] range
    similarities = np.clip(similarities, 0.0, 1.0)

    return similarities


def extract_tmd_from_784d(vector: np.ndarray) -> np.ndarray:
    """Extract TMD (first 16 dims) from 784D vector."""
    return vector[:16]


def extract_embedding_from_784d(vector: np.ndarray) -> np.ndarray:
    """Extract embedding (last 768 dims) from 784D vector."""
    return vector[16:]


def generate_tmd_for_query(query_text: str, use_llm: bool = True) -> np.ndarray:
    """
    Generate TMD vector for query text using THE SAME encoding as corpus.

    Uses LLM-based extraction (default) or pattern-based extraction + fixed random projection.
    This preserves ALL categorical signal from:
    - 16 domains × 32 tasks × 64 modifiers = 32,768 unique TMD codes
    - Encoded as 15D binary (4+5+6 bits) → projected to 16D → L2-normalized

    Args:
        query_text: Query string
        use_llm: Use LLM-based TMD extraction (default: True)

    Returns:
        16D TMD vector (SAME encoding as corpus!)
    """
    if not HAS_TMD_ENCODER:
        # Fallback: return zero vector
        return np.zeros(16, dtype=np.float32)

    # Step 1: Extract TMD codes from query text
    if use_llm and HAS_LLM_TMD_EXTRACTOR:
        # Use LLM-based extraction
        tmd_dict = extract_tmd_with_llm(query_text)
    elif HAS_TMD_EXTRACTOR:
        # Fallback to pattern-based extraction
        tmd_dict = extract_tmd_pattern_based(query_text)
    else:
        # No extractor available
        return np.zeros(16, dtype=np.float32)

    # Extract codes (LLM returns 0-based, pattern returns 1-based)
    domain = tmd_dict.get('domain_code', 9)
    task = tmd_dict.get('task_code', 1)
    modifier = tmd_dict.get('modifier_code', 27)

    # Adjust if extractor returns 1-based codes (pattern-based does this)
    # LLM-based extractor should return 0-based (check src/llm_tmd_extractor.py)
    if not use_llm or not HAS_LLM_TMD_EXTRACTOR:
        # Pattern-based returns 1-based, convert to 0-based
        if domain > 0:
            domain -= 1  # Convert 1-16 → 0-15
        if task > 0:
            task -= 1    # Convert 1-32 → 0-31
        if modifier > 0:
            modifier -= 1  # Convert 1-64 → 0-63

    # Clamp to valid ranges
    domain = max(0, min(15, domain))
    task = max(0, min(31, task))
    modifier = max(0, min(63, modifier))

    # Step 2: Use SAME encoding function as corpus ingestion
    tmd_16d = encode_tmd16(domain, task, modifier)

    return tmd_16d


def run_vecrag_tmd_rerank(
    db,  # FaissDB instance
    queries: List[np.ndarray],
    query_texts: List[str],  # NEW: Need query texts for TMD generation
    corpus_vectors: np.ndarray,  # (N, 784) full corpus vectors
    topk: int,
    alpha: float = 0.7  # Weight for vector score vs TMD score
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """
    Run vecRAG with TMD-based re-ranking.

    Args:
        db: FaissDB instance
        queries: List of query vectors (784D each)
        query_texts: List of query text strings (for TMD generation)
        corpus_vectors: Full corpus vectors (N, 784) for TMD extraction
        topk: Number of final results to return
        alpha: Weight for vec_score in combination (0.0-1.0)
               final_score = alpha * vec_score + (1-alpha) * tmd_score

    Returns:
        (indices, scores, latencies)
    """
    all_indices: List[List[int]] = []
    all_scores: List[List[float]] = []
    latencies: List[float] = []

    # Get more results than needed for re-ranking
    search_k = min(topk * 2, corpus_vectors.shape[0])

    for q, q_text in zip(queries, query_texts):
        t0 = time.perf_counter()

        # Step 1: Initial FAISS search using full 784D vector
        q = q.reshape(1, -1).astype(np.float32)
        n = float(np.linalg.norm(q))
        q = q / n if n > 0 else q

        D, I = db.search(q, search_k)

        vec_indices = [int(x) for x in I[0]]
        vec_scores = [float(s) for s in D[0]]

        # Step 2: Generate TMD from query TEXT (not from vector!)
        query_tmd = generate_tmd_for_query(q_text)

        # Step 3: Extract TMDs from retrieved results
        result_vectors = corpus_vectors[vec_indices]  # (search_k, 784)
        result_tmds = result_vectors[:, :16]  # (search_k, 16)

        # Step 4: Compute TMD similarities
        tmd_similarities = compute_tmd_similarity(query_tmd, result_tmds)

        # Step 5: Combine scores
        # Normalize vec_scores to [0, 1] range
        vec_scores_norm = np.array(vec_scores)
        if len(vec_scores_norm) > 0 and vec_scores_norm.max() > 0:
            vec_scores_norm = (vec_scores_norm - vec_scores_norm.min()) / (vec_scores_norm.max() - vec_scores_norm.min() + 1e-8)

        combined_scores = alpha * vec_scores_norm + (1.0 - alpha) * tmd_similarities

        # Step 6: Re-rank by combined scores
        sorted_pairs = sorted(
            zip(vec_indices, combined_scores.tolist()),
            key=lambda x: -x[1]
        )

        # Step 7: Return top-K
        reranked_indices = [idx for idx, _ in sorted_pairs[:topk]]
        reranked_scores = [score for _, score in sorted_pairs[:topk]]

        all_indices.append(reranked_indices)
        all_scores.append(reranked_scores)

        latencies.append((time.perf_counter() - t0) * 1000.0)

    return all_indices, all_scores, latencies


def run_vecrag_tmd_only(
    db,  # FaissDB instance
    queries: List[np.ndarray],
    corpus_vectors: np.ndarray,
    topk: int
) -> Tuple[List[List[int]], List[List[float]], List[float]]:
    """
    Retrieve using ONLY TMD matching (ignoring semantic embedding).

    This is a diagnostic mode to see how much TMD alone contributes.

    Args:
        db: FaissDB instance (not used, but kept for API compatibility)
        queries: List of query vectors (784D each)
        corpus_vectors: Full corpus vectors (N, 784)
        topk: Number of results to return

    Returns:
        (indices, scores, latencies)
    """
    all_indices: List[List[int]] = []
    all_scores: List[List[float]] = []
    latencies: List[float] = []

    # Pre-extract all corpus TMDs
    corpus_tmds = corpus_vectors[:, :16]  # (N, 16)

    for q in queries:
        t0 = time.perf_counter()

        # Extract TMD from query
        query_tmd = extract_tmd_from_784d(q)

        # Compute TMD similarities to ALL corpus items
        tmd_similarities = compute_tmd_similarity(query_tmd, corpus_tmds)

        # Get top-K by TMD similarity
        top_indices = np.argsort(-tmd_similarities)[:topk]
        top_scores = tmd_similarities[top_indices]

        all_indices.append(top_indices.tolist())
        all_scores.append(top_scores.tolist())

        latencies.append((time.perf_counter() - t0) * 1000.0)

    return all_indices, all_scores, latencies
