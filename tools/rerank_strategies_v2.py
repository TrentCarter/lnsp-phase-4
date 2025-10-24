#!/usr/bin/env python3
"""
Retrieval Reranking Strategies v2 (Production-Ready)
====================================================

Production-grade implementation from consultant with:
- Type-safe Candidate tuples
- Robust edge case handling
- Integrated directional bonus
- Clean API design

Based on consultant code review and recommendations.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable
import math
import numpy as np

# Type alias for candidate tuples
# Convention: (text, cosine_to_query, meta, vec768)
# meta should include: {"article_index": int, "chunk_index": int, "cpe_id": str}
Candidate = Tuple[str, float, Dict[str, Any], np.ndarray]


def dedup_candidates(cands: Iterable[Candidate]) -> List[Candidate]:
    """Drop duplicates by (article_index, chunk_index)."""
    seen = set()
    out: List[Candidate] = []
    for text, cos, meta, vec in cands:
        key = (int(meta.get("article_index", -1)), int(meta.get("chunk_index", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append((text, float(cos), meta, vec))
    return out


def _l2norm(x: np.ndarray) -> np.ndarray:
    """Safe L2 normalization with epsilon for numerical stability."""
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_: float = 0.7, k: int = 10) -> List[int]:
    """
    Maximal Marginal Relevance.

    Args:
        query_vec: Query vector [768]
        cand_vecs: Candidate vectors [N,768] (will be L2-normalized)
        lambda_: Trade-off between relevance and diversity (default 0.7)
        k: Number of candidates to select

    Returns:
        Indices of selected items (len k or less if N<k)
    """
    if cand_vecs.size == 0:
        return []

    q = _l2norm(query_vec.reshape(1, -1))[0]
    C = _l2norm(cand_vecs)
    sims = C @ q  # [N] cosine similarities

    selected: List[int] = []
    remaining = list(range(cand_vecs.shape[0]))

    while remaining and len(selected) < k:
        if not selected:
            # First selection: highest similarity to query
            best_local = max(remaining, key=lambda i: sims[i])
            selected.append(best_local)
            remaining.remove(best_local)
            continue

        # Diversity term: max similarity to already-selected candidates
        sel_mat = C[selected] @ C[remaining].T  # [len(sel), len(rem)]
        div = sel_mat.max(axis=0)               # [len(rem)]

        # MMR score: λ * relevance - (1-λ) * redundancy
        mmr_score = lambda_ * sims[remaining] - (1.0 - lambda_) * div

        # Select candidate with highest MMR score
        j = int(np.argmax(mmr_score))
        pick = remaining[j]
        selected.append(pick)
        remaining.pop(j)

    return selected


def rerank_with_sequence_bias(
    candidates: List[Candidate],
    last_ctx_meta: Dict[str, Any],
    w_cos: float = 1.0,
    w_same_article: float = 0.05,
    w_next_gap: float = 0.12,
    tau: float = 3.0,
    directional_bonus: float = 0.0,
    pred_vec: np.ndarray | None = None,
    last_vec: np.ndarray | None = None,
) -> List[Tuple[float, Candidate]]:
    """
    Rerank with sequence bias and optional directional bonus.

    Scoring formula:
        score = w_cos * cosine
              + w_same_article * 1{same article}
              + w_next_gap * exp(-max(0, chunk_gap-0.5)/tau)  (push gap≈+1)
              + directional_bonus * dir_score  (optional)

    Args:
        candidates: List of (text, cosine, meta, vec) tuples
        last_ctx_meta: Metadata of last context chunk
        w_cos: Weight for cosine similarity (default 1.0)
        w_same_article: Bonus for same article (default 0.05)
        w_next_gap: Weight for next-chunk bonus (default 0.12)
        tau: Temperature for exponential decay (default 3.0)
        directional_bonus: Weight for directional alignment (default 0.0, disabled)
        pred_vec: Predicted next vector (needed for directional bonus)
        last_vec: Last context vector (needed for directional bonus)

    Returns:
        List of (score, candidate) tuples sorted by score (descending)
    """
    a0 = int(last_ctx_meta.get("article_index", -1))
    c0 = int(last_ctx_meta.get("chunk_index", -1))

    out: List[Tuple[float, Candidate]] = []

    for cand in candidates:
        text, cos, meta, vec = cand
        ai = int(meta.get("article_index", -999999))
        ci = int(meta.get("chunk_index", -999999))

        # Same article bonus
        same_article = 1.0 if ai == a0 else 0.0

        # Chunk gap (we want gap ≈ +1 for sequential continuation)
        gap = ci - c0
        forward_gap = max(0.0, gap - 0.5) if same_article else 999.0
        seq_bonus = math.exp(-forward_gap / tau) if same_article else 0.0

        # Base score
        score = w_cos * float(cos) + w_same_article * same_article + w_next_gap * seq_bonus

        # Optional directional bonus
        if directional_bonus and pred_vec is not None and last_vec is not None and vec is not None:
            v1 = pred_vec - last_vec
            v2 = vec - last_vec
            v1 = _l2norm(v1.reshape(1, -1))[0]
            v2 = _l2norm(v2.reshape(1, -1))[0]
            dir_score = float(v1 @ v2)
            score += directional_bonus * dir_score

        out.append((score, cand))

    # Sort by score (descending)
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def cosine_softmax_weights(cosines: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """
    Stable softmax over cosine scores.

    Args:
        cosines: Array of cosine similarities [k]
        temperature: Softmax temperature (default 0.05)

    Returns:
        Normalized weights [k] summing to 1.0
    """
    if cosines.size == 0:
        return cosines

    z = cosines / max(1e-8, temperature)
    z = z - z.max()  # Numerical stability
    w = np.exp(z)
    return w / (w.sum() + 1e-12)


if __name__ == "__main__":
    print("✅ Reranking strategies v2 (production) loaded successfully!")
    print("\nAvailable functions:")
    print("  - dedup_candidates()")
    print("  - mmr()")
    print("  - rerank_with_sequence_bias()")
    print("  - cosine_softmax_weights()")
    print("  - _l2norm() [helper]")
    print("\nType alias:")
    print("  - Candidate = Tuple[str, float, Dict[str, Any], np.ndarray]")
