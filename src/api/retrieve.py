"""Lightweight FastAPI service for lane-aware retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

try:  # FastAPI is optional at runtime
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore

from ..integrations.lightrag import (
    LightRAGConfig,
    LightRAGHybridRetriever,
)
from ..prompt_extractor import PromptExtractor
from ..db_faiss import FaissDB
from ..tmd_encoder import pack_tmd, lane_index_from_bits
from ..vectorizer import EmbeddingBackend

DEFAULT_NPZ = os.getenv("FAISS_NPZ_PATH", "artifacts/fw1k_vectors.npz")


class Candidate(BaseModel):
    cpe_id: str
    score: float
    rank: int
    lane_index: Optional[int]
    retriever: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    lane_index: Optional[int]
    results: List[Candidate]


class RetrievalContext:
    def __init__(self, npz_path: str = DEFAULT_NPZ) -> None:
        config = LightRAGConfig.from_env()
        self.adapter = LightRAGHybridRetriever.from_config(config=config, dim=784)
        self.faiss_db = FaissDB(output_path=npz_path, retriever_adapter=self.adapter)
        self.loaded = False
        self.npz_path = npz_path

        if os.path.exists(npz_path):
            self.loaded = self.faiss_db.load(npz_path)
            if not self.loaded:
                print(f"[RetrievalContext] Failed to load vectors from {npz_path}")
        else:
            print(f"[RetrievalContext] NPZ not found at {npz_path}; retrieval will return empty results")

        self.extractor = PromptExtractor()
        self.embedder = EmbeddingBackend()

    def search(self, query: str, k: int) -> SearchResponse:
        if not self.loaded:
            return SearchResponse(query=query, lane_index=None, results=[])

        extraction = self.extractor.extract_cpe_from_text(query)
        d_code = extraction["domain_code"]
        t_code = extraction["task_code"]
        m_code = extraction["modifier_code"]
        lane_index = lane_index_from_bits(pack_tmd(d_code, t_code, m_code))

        concept_vec = self.embedder.encode([extraction["concept"]])[0]
        tmd_dense = np.array(extraction["tmd_dense"], dtype=np.float32)
        fused_query = np.concatenate([tmd_dense, concept_vec]).astype(np.float32)

        candidates = self.faiss_db.search(fused_query, topk=k, use_lightrag=True) or []
        return SearchResponse(
            query=query,
            lane_index=lane_index,
            results=[Candidate(**candidate) for candidate in candidates],
        )


@lru_cache(maxsize=1)
def get_context() -> RetrievalContext:
    return RetrievalContext()


if FastAPI is None:  # pragma: no cover
    app = None
else:
    app = FastAPI(title="LNSP Retrieval API", version="0.1.0")

    @app.get("/healthz")
    def healthcheck() -> Dict[str, str]:
        ctx = get_context()
        status = "ready" if ctx.loaded else "empty"
        return {"status": status, "npz_path": ctx.npz_path}

    @app.get("/search", response_model=SearchResponse)
    def search(q: str = Query(..., description="Natural language query"), k: int = Query(10, ge=1, le=50)) -> SearchResponse:
        ctx = get_context()
        try:
            return ctx.search(q, k)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))
