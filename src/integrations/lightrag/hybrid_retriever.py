"""Hybrid retrieval helper powered by LightRAG scoring utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .config import LightRAGConfig

try:
    from third_party.lightrag.functional import (
        get_top_k_indices_scores,
        normalize_vector,
    )
except Exception:  # pragma: no cover - fallback when LightRAG snippets unavailable
    def normalize_vector(vector: Iterable[float]) -> List[float]:  # type: ignore
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr) or 1.0
        return (arr / norm).tolist()

    def get_top_k_indices_scores(
        scores: Iterable[float], top_k: int
    ) -> tuple[list[int], list[float]]:  # type: ignore
        arr = np.asarray(list(scores), dtype=np.float32)
        idx = np.argsort(arr)[-top_k:][::-1]
        return idx.tolist(), arr[idx].tolist()


Candidate = Dict[str, Any]


@dataclass
class _Cache:
    matrix: Optional[np.ndarray] = None
    dirty: bool = True


class LightRAGHybridRetriever:
    """Blend Faiss shortlist scores with LightRAG-style vector heuristics."""

    def __init__(
        self,
        config: LightRAGConfig,
        dim: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.dim = dim
        self._logger = logger or logging.getLogger(__name__)
        self.available: bool = config.query_enabled

        self._vectors: List[np.ndarray] = []
        self._metadata: List[Candidate] = []
        self._cache = _Cache()

        if not self.available:
            self._logger.debug("LightRAG hybrid retriever disabled via config")

    @classmethod
    def from_config(
        cls, config: LightRAGConfig, dim: int, logger: Optional[logging.Logger] = None
    ) -> "LightRAGHybridRetriever":
        return cls(config=config, dim=dim, logger=logger)

    def register_document(self, vector: Iterable[float], metadata: Candidate) -> None:
        """Register a new document vector for LightRAG-style scoring."""

        if not self.available:
            return

        arr = np.asarray(list(vector), dtype=np.float32)
        if arr.size != self.dim:
            raise ValueError(
                f"Expected vector of dimension {self.dim}, received {arr.size}"
            )

        normalized = np.asarray(normalize_vector(arr), dtype=np.float32)
        self._vectors.append(normalized)
        self._metadata.append(dict(metadata))
        self._cache.dirty = True

    def search(
        self,
        query_vector: Iterable[float],
        top_k: int,
        fallback_candidates: Optional[List[Candidate]] = None,
    ) -> List[Candidate]:
        """Return blended candidates ranked with LightRAG-inspired scores."""

        if not self.available or not self._vectors:
            return fallback_candidates or []

        query = np.asarray(normalize_vector(list(query_vector)), dtype=np.float32)
        if query.size != self.dim:
            raise ValueError(
                f"Expected query of dimension {self.dim}, received {query.size}"
            )

        matrix = self._ensure_matrix()
        if matrix is None:
            return fallback_candidates or []

        scores = matrix @ query
        k = min(top_k, scores.shape[0])
        indices, score_values = get_top_k_indices_scores(scores, k)

        results: List[Candidate] = []
        for rank, (idx, score) in enumerate(zip(indices, score_values), start=1):
            payload = dict(self._metadata[idx])
            payload.update(
                {
                    "score": float(score),
                    "rank": rank,
                    "retriever": "lightrag_hybrid",
                }
            )
            results.append(payload)

        if fallback_candidates and self.config.allow_fallback:
            results = self._blend_with_fallback(results, fallback_candidates, top_k)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_matrix(self) -> Optional[np.ndarray]:
        if not self._cache.dirty and self._cache.matrix is not None:
            return self._cache.matrix

        if not self._vectors:
            self._cache.matrix = None
            self._cache.dirty = False
            return None

        self._cache.matrix = np.vstack(self._vectors)
        self._cache.dirty = False
        return self._cache.matrix

    def _blend_with_fallback(
        self, primary: List[Candidate], fallback: List[Candidate], top_k: int
    ) -> List[Candidate]:
        merged: Dict[Any, Candidate] = {}

        for item in fallback:
            merged[item["cpe_id"]] = dict(item)

        for item in primary:
            cpe_id = item["cpe_id"]
            prior = merged.get(cpe_id)
            if prior is not None:
                blended = (
                    self.config.query_weight * item.get("score", 0.0)
                    + (1.0 - self.config.query_weight) * prior.get("score", 0.0)
                )
                prior.update(item)
                prior["score"] = float(blended)
            else:
                merged[cpe_id] = dict(item)

        ranked = sorted(
            merged.values(), key=lambda payload: payload.get("score", 0.0), reverse=True
        )

        for idx, candidate in enumerate(ranked, start=1):
            candidate.setdefault("retriever", "lightrag_hybrid")
            candidate["rank"] = idx

        return ranked[:top_k]
