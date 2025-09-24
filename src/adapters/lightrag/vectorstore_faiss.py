"""External FAISS bridge for LightRAG GraphRAG runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ...db_faiss import FaissDB
from ...utils.norms import l2_normalize

logger = logging.getLogger(__name__)


class LightRAGFaissVectorStore:
    """Wrap the existing FaissDB with LightRAG-friendly helpers."""

    def __init__(self, index_path: str, meta_npz_path: str, expected_dim: int = 768) -> None:
        self._index_path = Path(index_path)
        self._meta_npz_path = Path(meta_npz_path)
        self._expected_dim = expected_dim
        self._faiss = FaissDB(index_path=str(self._index_path), meta_npz_path=str(self._meta_npz_path))
        self._ensure_meta()
        self._faiss.load(str(self._index_path))

        if self._faiss.dim != expected_dim:
            raise RuntimeError(
                f"FAISS dimension mismatch: expected {expected_dim}D, got {self._faiss.dim}D"
            )

        logger.info(
            "Loaded FAISS index %s (%d vectors, dim=%d)",
            self._index_path,
            getattr(self._faiss.index, "ntotal", 0),
            self._faiss.dim,
        )

    def _ensure_meta(self) -> None:
        if not self._meta_npz_path.exists():
            raise FileNotFoundError(f"Metadata NPZ not found: {self._meta_npz_path}")

        npz = np.load(self._meta_npz_path, allow_pickle=True)
        vectors = None
        for key in ("vectors", "fused"):
            if key in npz:
                vectors = npz[key]
                break

        if vectors is None:
            raise ValueError(
                f"{self._meta_npz_path} missing 'vectors' or 'fused' array required for zero-vector check"
            )

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._expected_dim:
            raise ValueError(
                f"Metadata NPZ has shape {vectors.shape}; expected (*,{self._expected_dim})"
            )

        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms < 1e-6):
            raise RuntimeError(
                "Zero or near-zero vectors detected in metadata NPZ; aborting GraphRAG run"
            )

    @property
    def dim(self) -> int:
        return self._faiss.dim

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        if query_vectors.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {query_vectors.shape[1]} does not match FAISS dim {self.dim}"
            )

        queries = l2_normalize(query_vectors.astype(np.float32))
        scores, indices = self._faiss.search(queries, top_k)
        return scores, indices

    def fetch_metadata(self, indices: Iterable[int]) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for idx in indices:
            if idx < 0:
                results.append({})
                continue

            payload: Dict[str, str] = {}
            if self._faiss.cpe_ids is not None and idx < len(self._faiss.cpe_ids):
                payload["cpe_id"] = str(self._faiss.cpe_ids[idx])
            if self._faiss.doc_ids is not None and idx < len(self._faiss.doc_ids):
                payload["doc_id"] = str(self._faiss.doc_ids[idx])
            if self._faiss.concept_texts is not None and idx < len(self._faiss.concept_texts):
                payload["concept_text"] = str(self._faiss.concept_texts[idx])
            if self._faiss.lane_indices is not None and idx < len(self._faiss.lane_indices):
                payload["lane_index"] = str(int(self._faiss.lane_indices[idx]))
            results.append(payload)
        return results


def get_vector_store(index_path: str, meta_npz_path: str, dim: int = 768) -> LightRAGFaissVectorStore:
    """Factory used by LightRAG configuration to instantiate the vector store."""

    return LightRAGFaissVectorStore(index_path=index_path, meta_npz_path=meta_npz_path, expected_dim=dim)


__all__ = ["LightRAGFaissVectorStore", "get_vector_store"]
