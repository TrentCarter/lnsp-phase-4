"""Lightweight FastAPI service for lane-aware retrieval."""

from __future__ import annotations

import sys
assert sys.version_info[:2] == (3, 11) or sys.version_info[:2] == (3, 13), f"Require Python 3.11.x or 3.13.x (got {sys.version})"

import os
import re
import uuid
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # FastAPI is optional at runtime
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore

from ..schemas import CPESHDiagnostics, SearchRequest, SearchResponse, SearchItem
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

_ENV_NPZ = os.getenv("FAISS_NPZ_PATH")
_NPZ_10K_768 = "artifacts/fw10k_vectors_768.npz"
_NPZ_10K = "artifacts/fw10k_vectors.npz"
_NPZ_1K = "artifacts/fw1k_vectors.npz"
DEFAULT_NPZ = _ENV_NPZ or (_NPZ_10K_768 if os.path.exists(_NPZ_10K_768) else (_NPZ_10K if os.path.exists(_NPZ_10K) else _NPZ_1K))

LANE_MAP = {0: "L1_FACTOID", 1: "L2_GRAPH", 2: "L3_SYNTH"}
MODE_MAP = {"dense": "DENSE", "graph": "GRAPH", "hybrid": "HYBRID", "lexical": "HYBRID"}


class RetrievalContext:
    def __init__(self, npz_path: str = DEFAULT_NPZ) -> None:
        config = LightRAGConfig.from_env()
        # LNSP_FUSED=0 for pure 768D, LNSP_FUSED=1 for 784D fused
        self.use_fused = os.getenv("LNSP_FUSED", "0") == "1"
        self.dim = 784 if self.use_fused else 768
        self.adapter = LightRAGHybridRetriever.from_config(config=config, dim=self.dim)
        self.faiss_db = FaissDB(meta_npz_path=npz_path)
        self.loaded = False
        self.npz_path = npz_path
        self.catalog: List[Dict[str, Any]] = []

        # A2: Boot invariant checks
        self._validate_npz_schema(npz_path)
        self._validate_faiss_dimension()

        # Initialize embedder before loading index (needed for probe)
        self.embedder = EmbeddingBackend()

        # Load FAISS index
        self._load_faiss_index()

    def _validate_npz_schema(self, npz_path: str) -> None:
        """Validate NPZ file contains all required keys and correct dimensions."""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        required_keys = ["vectors", "ids", "doc_ids", "concept_texts", "tmd_dense", "lane_indices"]
        missing = [k for k in required_keys if k not in npz]
        if missing:
            raise ValueError(f"NPZ missing required keys: {missing} in {npz_path}")

        # Dimension check for 768D mode
        if not self.use_fused and npz["vectors"].shape[1] != 768:
            raise ValueError(f"Expected 768D vectors for LNSP_FUSED=0, got {npz['vectors'].shape[1]}D in {npz_path}")

        # Shape consistency check
        n_vectors = len(npz["vectors"])
        for key in ["ids", "doc_ids", "concept_texts", "lane_indices"]:
            if len(npz[key]) != n_vectors:
                raise ValueError(f"Shape mismatch: {key} has {len(npz[key])} items but vectors has {n_vectors}")

        print(f"[RetrievalContext] NPZ schema validation passed: {npz_path}")

    def _validate_faiss_dimension(self) -> None:
        """Validate FAISS metadata matches expected dimension."""
        meta_path = Path("artifacts/faiss_meta.json")
        if not meta_path.exists():
            print(f"[RetrievalContext] Warning: faiss_meta.json not found, skipping dimension check")
            return

        try:
            with meta_path.open('r') as f:
                meta = json.load(f)

            expected_dim = 768 if not self.use_fused else 784
            index_dim = meta.get("dimension", 0)

            if index_dim != expected_dim:
                raise RuntimeError(
                    f"FAISS dimension mismatch: expected {expected_dim}D for LNSP_FUSED={int(self.use_fused)}, "
                    f"got {index_dim}D. Rebuild index with correct dimensions."
                )

            print(f"[RetrievalContext] FAISS dimension validation passed: {index_dim}D")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid faiss_meta.json: {e}")

    def _probe_search_smoke(self) -> None:
        """Probe search functionality with known query to ensure system is functional."""
        if not self.loaded:
            return

        try:
            # Test with a simple query
            from ..schemas import SearchRequest
            test_req = SearchRequest(q="artificial intelligence", lane="L1_FACTOID", top_k=1)
            result = self.search(test_req, trace_id="boot_probe")

            if not result.items:
                raise RuntimeError("Search probe returned no results - system may be unhealthy")

            print(f"[RetrievalContext] Search probe passed: {len(result.items)} results")
        except Exception as e:
            raise RuntimeError(f"Search probe failed: {e}")

    def _load_faiss_index(self) -> None:
        """Load FAISS index from metadata."""
        meta_path = Path("artifacts/faiss_meta.json")
        if meta_path.exists():
            with meta_path.open('r') as f:
                meta = json.load(f)

            index_path = meta.get("index_path")
            if index_path and Path(index_path).exists():
                self.loaded = self.faiss_db.load(index_path)
                if self.loaded:
                    self._build_catalog()
                    # Run search smoke test after successful load
                    self._probe_search_smoke()
                else:
                    print(f"[RetrievalContext] Failed to load index from {index_path}")
            else:
                print(f"[RetrievalContext] Index path not found in {meta_path} or file does not exist")
        else:
            print(f"[RetrievalContext] faiss_meta.json not found; retrieval will be empty")

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

        # Extract hydrated fields
        metadata = h.get("metadata") or {}
        concept_text = metadata.get("concept_text") or h.get("concept_text")
        tmd_code = h.get("tmd_code")
        lane_index = h.get("lane_index")

        return SearchItem(
            id=cpe_id,
            doc_id=doc_id,
            score=h.get("score"),
            why=h.get("why"),
            concept_text=concept_text,
            tmd_code=tmd_code,
            lane_index=lane_index
        )

    def search(self, req: SearchRequest, trace_id: Optional[str] = None) -> SearchResponse:
        print(f"Trace {trace_id}: Received search request: {req}")
        if not self.loaded:
            print(f"Trace {trace_id}: Context not loaded, returning empty response.")
            return SearchResponse(lane=req.lane, mode="HYBRID", items=[], trace_id=trace_id)

        # Check cache first
        cached_items = _search_cache.get(req.lane, req.q, req.top_k)
        if cached_items is not None:
            return SearchResponse(lane=req.lane, mode="DENSE", items=cached_items, trace_id=trace_id)

        extraction = extract_cpe_from_text(req.q)
        d_code = extraction["domain_code"]
        t_code = extraction["task_code"]
        m_code = extraction["modifier_code"]

        concept_vec = self.embedder.encode([extraction["concept"]])[0].astype(np.float32)
        concept_norm = float(np.linalg.norm(concept_vec))
        concept_unit = concept_vec if concept_norm == 0 else (concept_vec / concept_norm)

        if self.use_fused:
            # 784D mode: use TMD + concept vector
            tmd_dense = np.array(extraction["tmd_dense"], dtype=np.float32)
            fused_query = np.concatenate([tmd_dense, concept_vec]).astype(np.float32)
        else:
            # 768D mode: use concept vector only, ensure normalization
            fused_query = concept_vec.copy()
            # L2 normalize for inner product search
            norm = np.linalg.norm(fused_query)
            if norm > 0:
                fused_query = fused_query / norm

        mode = settings.RETRIEVAL_MODE if settings.RETRIEVAL_MODE != "HYBRID" else "DENSE"
        candidates = self.faiss_db.search_legacy(fused_query, topk=req.top_k, use_lightrag=True) or []

        # Apply lane_index filter if specified
        if req.lane_index is not None:
            candidates = [c for c in candidates if c.get("lane_index") == req.lane_index]
            print(f"Trace {trace_id}: Filtered to {len(candidates)} candidates with lane_index={req.lane_index}")

        # L1_FACTOID: dense-only by default, lexical fallback via flag
        enable_lex = os.getenv("LNSP_LEXICAL_FALLBACK", "0") == "1"
        if req.lane == "L1_FACTOID" and not enable_lex:
            # Skip lexical fallback for L1_FACTOID unless explicitly enabled
            pass
        elif settings.LEXICAL_FALLBACK and all((c.get("score") or 0.0) <= settings.MIN_VALID_SCORE for c in candidates):
            # Use lexical fallback for other lanes or when explicitly enabled
            candidates = self._lexical_search(req.q, req.top_k)
            mode = "HYBRID"

        print(f"Trace {trace_id}: FAISS search returned {len(candidates)} candidates.")
        items = [self._norm_hit(h) for h in candidates if h]
        print(f"Trace {trace_id}: Normalized {len(items)} items.")

        # --- CPESH diagnostics (optional) ---
        diag = CPESHDiagnostics(
            concept=extraction.get("concept"),
            probe=extraction.get("probe"),
            expected=extraction.get("expected"),
            soft_negative=extraction.get("soft_negative"),
            hard_negative=extraction.get("hard_negative"),
        )

        # Compute similarity scores if negatives exist and concept vector is valid
        if concept_norm > 0:
            try:
                if diag.soft_negative:
                    soft_vec = self.embedder.encode([diag.soft_negative])[0].astype(np.float32)
                    soft_norm = float(np.linalg.norm(soft_vec))
                    if soft_norm > 0:
                        diag.soft_sim = float(np.dot(concept_unit, soft_vec / soft_norm))
                if diag.hard_negative:
                    hard_vec = self.embedder.encode([diag.hard_negative])[0].astype(np.float32)
                    hard_norm = float(np.linalg.norm(hard_vec))
                    if hard_norm > 0:
                        diag.hard_sim = float(np.dot(concept_unit, hard_vec / hard_norm))
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"Trace {trace_id}: CPESH sim computation skipped: {exc}")

        insufficient = False
        if not items:
            insufficient = True
        else:
            top_score = float(items[0].score or 0.0)
            hard_sim_val = float(diag.hard_sim) if diag.hard_sim is not None else 0.0
            if top_score < 0.10 and hard_sim_val >= 0.30:
                insufficient = True

        diag_payload: Optional[CPESHDiagnostics] = None
        if diag.model_dump(exclude_none=True):
            diag_payload = diag

        # Cache the results
        _search_cache.put(req.lane, req.q, req.top_k, items)

        return SearchResponse(
            lane=req.lane,
            mode=mode,
            items=items,
            trace_id=trace_id,
            diagnostics=diag_payload,
            insufficient_evidence=insufficient,
        )


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
    def healthcheck() -> Dict[str, Any]:
        import sys
        import json

        ctx = get_context()
        status = "ready" if ctx.loaded else "empty"

        # Load faiss metadata
        meta = {}
        try:
            with open("artifacts/faiss_meta.json", "r") as f:
                meta = json.load(f)
        except Exception:
            pass

        # Check lexical fallback setting
        lexical_l1 = os.getenv("LNSP_LEXICAL_FALLBACK", "0") == "1"

        return {
            "status": status,
            "npz_path": ctx.npz_path,
            "py": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "index": meta.get("index_type", "N/A"),
            "vectors": meta.get("num_vectors", 0),
            "nlist": meta.get("nlist", 0),
            "nprobe": int(os.getenv("FAISS_NPROBE", "16")),
            "lexical_L1": lexical_l1
        }

    @app.get("/admin/faiss")
    def admin_faiss() -> Dict[str, Any]:
        """Return FAISS configuration and corpus stats without heavy initialization.

        Fields: {nlist, nprobe, metric, dim, vectors}
        Values are sourced from artifacts/faiss_meta.json and environment variables only
        to ensure this endpoint responds quickly even when the retrieval context has
        not been initialized.
        """
        import json

        # Read metadata file
        meta: Dict[str, Any] = {}
        try:
            with open("artifacts/faiss_meta.json", "r") as f:
                meta = json.load(f)
        except Exception:
            pass

        metric = "IP"  # We build indices with inner-product over L2-normalized vectors

        # Override dimension based on LNSP_FUSED setting
        use_fused = os.getenv("LNSP_FUSED", "0") == "1"
        reported_dim = 784 if use_fused else 768

        return {
            "nlist": int(meta.get("nlist", 0) or 0),
            "nprobe": int(os.getenv("FAISS_NPROBE", "16") or 16),
            "metric": metric,
            "dim": reported_dim,
            "vectors": int(meta.get("num_vectors", 0) or 0),
        }

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request) -> SearchResponse:
        ctx = get_context()

        # Extract or generate trace_id
        trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())[:8]

        try:
            return ctx.search(req, trace_id=trace_id)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=str(exc))
