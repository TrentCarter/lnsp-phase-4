#!/usr/bin/env python3
"""
Retrieval Reranking Strategies for LVM Output Decoding
=======================================================

Implements four core strategies to improve retrieval quality:
1. Deduplication - Remove duplicate (article, chunk) pairs
2. MMR (Maximal Marginal Relevance) - Diversify top-k results
3. Sequence-bias reranking - Prefer same-article, next-chunk continuations
4. Cosine-softmax weighting - Proper confidence scoring for vector outputs

Based on consultant recommendations for improving R@1 from LVM predictions.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# 1. DEDUPLICATION
# ============================================================================

def dedup_candidates(
    candidates: List[Tuple[str, float, Dict]]
) -> List[Tuple[str, float, Dict]]:
    """
    Remove duplicate candidates based on (article_index, chunk_index).

    Args:
        candidates: List of (text, cosine_similarity, metadata) tuples

    Returns:
        Deduplicated candidates (preserves first occurrence of each unique key)
    """
    seen = set()
    deduped = []

    for text, cosine, meta in candidates:
        # Create unique key from article and chunk indices
        key = (meta.get("article_index"), meta.get("chunk_index"))

        if key in seen:
            continue

        seen.add(key)
        deduped.append((text, cosine, meta))

    return deduped


# ============================================================================
# 2. MMR (MAXIMAL MARGINAL RELEVANCE)
# ============================================================================

def mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    lambda_: float = 0.7,
    k: int = 10
) -> List[int]:
    """
    Maximal Marginal Relevance for diversifying retrieval results.

    Balances relevance to query with diversity from already-selected candidates.

    Args:
        query_vec: Query vector [768] (L2 normalized)
        cand_vecs: Candidate vectors [N, 768] (L2 normalized)
        lambda_: Trade-off between relevance (high) and diversity (low). Default 0.7.
        k: Number of candidates to select

    Returns:
        List of selected candidate indices in MMR order
    """
    selected = []
    remaining = list(range(len(cand_vecs)))

    # Normalize query vector
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    # Compute similarity to query for all candidates
    sims = cand_vecs @ q  # [N] cosine similarities

    while remaining and len(selected) < k:
        if not selected:
            # First selection: highest similarity to query
            i = int(np.argmax(sims[remaining]))
            selected.append(remaining.pop(i))
            continue

        # Diversity term: max similarity to already-selected candidates
        sel_mat = cand_vecs[selected] @ cand_vecs[remaining].T  # [|selected|, |remaining|]
        div = np.max(sel_mat, axis=0)  # [|remaining|] max sim to any selected

        # MMR score: λ * relevance - (1-λ) * redundancy
        mmr_score = lambda_ * sims[remaining] - (1 - lambda_) * div

        # Select candidate with highest MMR score
        i = int(np.argmax(mmr_score))
        selected.append(remaining.pop(i))

    return selected


# ============================================================================
# 3. SEQUENCE-BIAS RERANKING
# ============================================================================

def rerank_with_sequence_bias(
    query_vec: np.ndarray,
    candidates: List[Tuple[str, float, Dict]],
    last_ctx_meta: Dict,
    w_cos: float = 1.0,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0
) -> List[Tuple[float, str, float, Dict]]:
    """
    Rerank candidates with bias toward sequential continuations.

    Scoring formula:
        score = w_cos * cosine
              + w_same_article * 1{same article}
              + w_next_gap * exp(-max(0, candidate.chunk - last.chunk - 0.5) / tau)

    The sequence bonus prefers candidates from the same article with chunk index
    close to (last_chunk + 1), decaying exponentially as the gap increases.

    Args:
        query_vec: Predicted next vector [768]
        candidates: List of (text, cosine, metadata) tuples
        last_ctx_meta: Metadata of last context chunk with keys:
            - "article_index": int
            - "chunk_index": int
        w_cos: Weight for cosine similarity (default 1.0)
        w_same_article: Bonus for same article (default 0.05)
        w_next_gap: Weight for next-chunk bonus (default 0.12)
        tau: Temperature for exponential decay of gap penalty (default 3.0)

    Returns:
        Reranked candidates as [(score, text, cosine, metadata), ...]
        sorted by score (descending)
    """
    a0 = last_ctx_meta.get("article_index", -1)
    c0 = last_ctx_meta.get("chunk_index", -1)

    scored = []

    for text, cos, meta in candidates:
        # Check if same article
        same_article = 1.0 if int(meta.get("article_index", -2)) == int(a0) else 0.0

        # Compute chunk gap
        gap = int(meta.get("chunk_index", -1)) - int(c0)

        # Forward gap: we want gap ≈ +1 (next chunk)
        forward_gap = max(0.0, gap - 0.5)

        # Sequence bonus: exponential decay as we move away from next chunk
        seq_bonus = math.exp(-forward_gap / tau) if same_article else 0.0

        # Combined score
        score = w_cos * cos + w_same_article * same_article + w_next_gap * seq_bonus

        scored.append((score, text, cos, meta))

    # Sort by score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)

    return scored


# ============================================================================
# 4. COSINE-SOFTMAX WEIGHTING
# ============================================================================

def cosine_softmax_weights(
    cosines: np.ndarray,
    temperature: float = 0.05
) -> np.ndarray:
    """
    Compute softmax weights from cosine similarities.

    For vector outputs (not logits), use cosine-based softmax instead of
    margin-based confidence. Temperature controls sharpness of distribution.

    Args:
        cosines: Array of cosine similarities [k]
        temperature: Softmax temperature (lower = sharper). Default 0.05.

    Returns:
        Normalized weights [k] that sum to 1.0
    """
    # Scale by temperature
    z = cosines / max(1e-8, temperature)

    # Numerical stability: subtract max
    z = z - z.max()

    # Softmax
    w = np.exp(z)
    w = w / (w.sum() + 1e-12)

    return w


# ============================================================================
# 5. DIRECTIONAL SCORE (OPTIONAL)
# ============================================================================

def directional_score(
    pred_vec: np.ndarray,
    last_vec: np.ndarray,
    cand_vec: np.ndarray
) -> float:
    """
    Compute directional alignment score.

    Measures how well the candidate's direction from last_vec aligns with
    the predicted direction. Helps distinguish "same topic" from "next step".

    Args:
        pred_vec: AMN's predicted next vector [768]
        last_vec: Last context vector [768]
        cand_vec: Candidate vector [768]

    Returns:
        Directional alignment score (cosine of direction vectors)
    """
    # Direction from last to prediction
    v1 = pred_vec - last_vec
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)

    # Direction from last to candidate
    v2 = cand_vec - last_vec
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    # Alignment
    return float(v1 @ v2)


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

def rerank_pipeline(
    query_vec: np.ndarray,
    candidates: List[Tuple[str, float, np.ndarray, Dict]],
    last_ctx_meta: Dict,
    candidate_vecs: Optional[np.ndarray] = None,
    k_final: int = 10,
    mmr_lambda: float = 0.7,
    use_mmr: bool = True,
    use_sequence_bias: bool = True,
    w_cos: float = 1.0,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0
) -> List[Tuple[float, str, float, Dict]]:
    """
    Integrated reranking pipeline combining all strategies.

    Pipeline order:
    1. Deduplication (remove duplicate article/chunk pairs)
    2. MMR (optional, for diversity)
    3. Sequence-bias reranking (optional, for sequential continuations)

    Args:
        query_vec: Predicted next vector [768]
        candidates: List of (text, cosine, vector, metadata) tuples
        last_ctx_meta: Metadata of last context chunk
        candidate_vecs: Optional pre-stacked candidate vectors [N, 768]
        k_final: Final number of candidates to return (default 10)
        mmr_lambda: MMR lambda parameter (default 0.7)
        use_mmr: Whether to apply MMR (default True)
        use_sequence_bias: Whether to apply sequence-bias reranking (default True)
        w_cos: Cosine weight for sequence reranking (default 1.0)
        w_same_article: Same-article bonus (default 0.05)
        w_next_gap: Next-chunk bonus weight (default 0.12)
        tau: Gap penalty temperature (default 3.0)

    Returns:
        Reranked candidates as [(score, text, cosine, metadata), ...]
    """
    # Step 1: Deduplication
    # Convert to format for dedup (text, cosine, meta)
    cands_for_dedup = [(text, cos, meta) for text, cos, vec, meta in candidates]
    deduped = dedup_candidates(cands_for_dedup)

    # Reconstruct with vectors
    deduped_with_vecs = []
    for text, cos, meta in deduped:
        # Find matching vector from original candidates
        for orig_text, orig_cos, orig_vec, orig_meta in candidates:
            if (orig_meta.get("article_index") == meta.get("article_index") and
                orig_meta.get("chunk_index") == meta.get("chunk_index")):
                deduped_with_vecs.append((text, cos, orig_vec, meta))
                break

    # Step 2: MMR (optional)
    if use_mmr and len(deduped_with_vecs) > k_final:
        # Stack vectors for MMR
        if candidate_vecs is None:
            vecs = np.stack([vec for _, _, vec, _ in deduped_with_vecs])
        else:
            # Use pre-stacked vectors (more efficient)
            vecs = candidate_vecs[:len(deduped_with_vecs)]

        # Apply MMR
        mmr_indices = mmr(query_vec, vecs, lambda_=mmr_lambda, k=k_final)
        deduped_with_vecs = [deduped_with_vecs[i] for i in mmr_indices]

    # Step 3: Sequence-bias reranking (optional)
    if use_sequence_bias:
        # Convert to format for reranking (text, cosine, meta)
        cands_for_rerank = [(text, cos, meta) for text, cos, _, meta in deduped_with_vecs]

        # Apply sequence-bias reranking
        reranked = rerank_with_sequence_bias(
            query_vec,
            cands_for_rerank,
            last_ctx_meta,
            w_cos=w_cos,
            w_same_article=w_same_article,
            w_next_gap=w_next_gap,
            tau=tau
        )

        return reranked[:k_final]
    else:
        # Just sort by cosine
        result = [(cos, text, cos, meta) for text, cos, _, meta in deduped_with_vecs]
        result.sort(key=lambda x: x[0], reverse=True)
        return result[:k_final]


if __name__ == "__main__":
    # Quick test
    print("Reranking strategies module loaded successfully!")
    print("\nAvailable functions:")
    print("  - dedup_candidates()")
    print("  - mmr()")
    print("  - rerank_with_sequence_bias()")
    print("  - cosine_softmax_weights()")
    print("  - directional_score()")
    print("  - rerank_pipeline() [integrated]")
