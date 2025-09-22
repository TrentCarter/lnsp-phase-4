"""Lightweight FastAPI service for lane-aware retrieval."""

from __future__ import annotations

import os
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # FastAPI is optional at runtime
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore

from ..schemas import SearchRequest, SearchResponse, SearchItem
from ..config import settings
from .cache import _search_cache
from ..integrations.lightrag import (
    LightRAGConfig,
    LightRAGHybridRetriever,
)
from ..prompt_extractor import extract_cpe_from_text
from ..db_faiss import FaissDB
from ..tmd_encoder import pack_tmd, lane_index_from_bits
from ..vectorizer import EmbeddingBackend

DEFAULT_NPZ = os.getenv("FAISS_NPZ_PATH", "artifacts/fw1k_vectors.npz")

LANE_MAP = {0: "L1_FACTOID", 1: "L2_GRAPH", 2: "L3_SYNTH"}
MODE_MAP = {"dense": "DENSE", "graph": "GRAPH", "hybrid": "HYBRID", "lexical": "HYBRID"}


class RetrievalContext:
    def __init__(self, npz_path: str = DEFAULT_NPZ) -> None:
        config = LightRAGConfig.from_env()
        self.adapter = LightRAGHybridRetriever.from_config(config=config, dim=784)
        self.faiss_db = FaissDB(output_path=npz_path, retriever_adapter=self.adapter)
        self.loaded = False
        self.npz_path = npz_path
        self.catalog: List[Dict[str, Any]] = []

        if os.path.exists(npz_path):
            self.loaded = self.faiss_db.load(npz_path)
            if not self.loaded:
                print(f"[RetrievalContext] Failed to load vectors from {npz_path}")
            else:
                self._build_catalog()
        else:
            print(f"[RetrievalContext] NPZ not found at {npz_path}; retrieval will return empty results")

        self.embedder = EmbeddingBackend()

    @staticmethod
    def _tokenize(text: str) -> Sequence[str]:
        if not text:
            return ()
        return tuple(re.findall(r"\w+", text.lower()))

    def _build_catalog(self) -> None:
        cpe_ids = getattr(self.faiss_db, "cpe_ids", [])
        doc_ids = getattr(self.faiss_db, "doc_ids", [])
        concepts = getattr(self.faiss_db, "concept_texts", [])

        size = len(cpe_ids)
        self.catalog = []
        for idx in range(size):
            cpe_id = str(cpe_ids[idx])
            doc_id = str(doc_ids[idx]) if len(doc_ids) > idx else ""
            concept = str(concepts[idx]) if len(concepts) > idx else ""
            tokens = set(self._tokenize(concept))
            self.catalog.append({
                "cpe_id": cpe_id,
                "doc_id": doc_id,
                "concept": concept,
                "tokens": tokens,
            })

    def _lexical_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.catalog:
            return []
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored: List[Dict[str, Any]] = []
        for item in self.catalog:
            overlap = len(query_tokens & item["tokens"])
            if overlap:
                scored.append({
                    "cpe_id": item["cpe_id"],
                    "doc_id": item["doc_id"],
                    "score": float(overlap),
                    "retriever": "lexical",
                    "lane_index": 0,
                    "metadata": {
                        "concept_text": item["concept"],
                        "doc_id": item["doc_id"],
                    },
                })

        if not scored:
            for item in self.catalog[:k]:
                scored.append({
                    "cpe_id": item["cpe_id"],
                    "doc_id": item["doc_id"],
                    "score": 0.0,
                    "retriever": "lexical",
                    "lane_index": 0,
                    "metadata": {
                        "concept_text": item["concept"],
                        "doc_id": item["doc_id"],
                    },
                })

        scored.sort(key=lambda x: (-x["score"], x.get("doc_id") or ""))
        for rank, item in enumerate(scored[:k], start=1):
            item["rank"] = rank
        return scored[:k]

    def _norm_hit(self, h: dict) -> SearchItem:
        """Normalize hit to standard SearchItem format."""
        cpe_id = h.get("cpe_id") or h.get("id") or h.get("uuid") or h.get("rid") or ""
        doc_id = h.get("doc_id") or (h.get("metadata") or {}).get("doc_id")
        return SearchItem(id=cpe_id, doc_id=doc_id, score=h.get("score"), why=h.get("why"))

    def search(self, req: SearchRequest, trace_id: Optional[str] = None) -> SearchResponse:
        if not self.loaded:
            return SearchResponse(lane=req.lane, mode="HYBRID", items=[], trace_id=trace_id)

        # Check cache first
        cached_items = _search_cache.get(req.lane, req.q, req.top_k)
        if cached_items is not None:
            return SearchResponse(lane=req.lane, mode="DENSE", items=cached_items, trace_id=trace_id)

        extraction = extract_cpe_from_text(req.q)
        d_code = extraction["domain_code"]
        t_code = extraction["task_code"]
        m_code = extraction["modifier_code"]
        lane_index = lane_index_from_bits(pack_tmd(d_code, t_code, m_code))

        concept_vec = self.embedder.encode([extraction["concept"]])[0]
        tmd_dense = np.array(extraction["tmd_dense"], dtype=np.float32)
        fused_query = np.concatenate([tmd_dense, concept_vec]).astype(np.float32)

        mode = settings.RETRIEVAL_MODE if settings.RETRIEVAL_MODE != "HYBRID" else "DENSE"
        candidates = self.faiss_db.search(fused_query, topk=req.top_k, use_lightrag=True) or []

        # Use lexical fallback if enabled and scores are degenerate
        if settings.LEXICAL_FALLBACK and all((c.get("score") or 0.0) <= settings.MIN_VALID_SCORE for c in candidates):
            candidates = self._lexical_search(req.q, req.top_k)
            mode = "HYBRID"

        items = [self._norm_hit(h) for h in candidates if h]

        # Cache the results
        _search_cache.put(req.lane, req.q, req.top_k, items)

        return SearchResponse(lane=req.lane, mode=mode, items=items, trace_id=trace_id)


@lru_cache(maxsize=1)
def get_context() -> RetrievalContext:
    return RetrievalContext()


if FastAPI is None:  # pragma: no cover
    app = None
else:
    from .middleware import TimingMiddleware

    app = FastAPI(title="LNSP Retrieval API", version="0.1.0")
    app.add_middleware(TimingMiddleware)

    @app.get("/healthz")
    def healthcheck() -> Dict[str, str]:
        ctx = get_context()
        status = "ready" if ctx.loaded else "empty"
        return {"status": status, "npz_path": ctx.npz_path}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request) -> SearchResponse:
        ctx = get_context()

        # Extract or generate trace_id
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())[:8]

        try:
            return ctx.search(req, trace_id=trace_id)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))
