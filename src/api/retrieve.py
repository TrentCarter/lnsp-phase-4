"""Lightweight FastAPI service for lane-aware retrieval."""

from __future__ import annotations

import sys
assert sys.version_info[:2] == (3, 11) or sys.version_info[:2] == (3, 13), f"Require Python 3.11.x or 3.13.x (got {sys.version})"

import os
import re
import uuid
import json
import time
from collections import deque

# Set FAISS threading early to avoid segfaults
try:
    import faiss
    faiss.omp_set_num_threads(int(os.getenv("FAISS_NUM_THREADS", "1")))
except Exception:
    pass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # FastAPI is optional at runtime
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore

from pydantic import BaseModel, Field

from ..schemas import CPESHDiagnostics, SearchRequest, SearchResponse, SearchItem, CPESH
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
from ..utils.timestamps import (
    get_iso_timestamp,
    migrate_legacy_cache_entry,
    migrate_cpesh_record,
    update_cache_entry_access,
)
from ..utils.gating import CPESHGateConfig, apply_lane_overrides, should_use_cpesh, log_gating_decision, get_gating_metrics

_ENV_NPZ = os.getenv("FAISS_NPZ_PATH")
_NPZ_V2 = "artifacts/cpesh_active_v2.npz"
_NPZ_10K_768 = "artifacts/fw10k_vectors_768.npz"
_NPZ_10K = "artifacts/fw10k_vectors.npz"
_NPZ_1K = "artifacts/fw1k_vectors.npz"
DEFAULT_NPZ = _ENV_NPZ or (_NPZ_V2 if os.path.exists(_NPZ_V2) else (_NPZ_10K_768 if os.path.exists(_NPZ_10K_768) else (_NPZ_10K if os.path.exists(_NPZ_10K) else _NPZ_1K)))

LANE_MAP = {0: "L1_FACTOID", 1: "L2_GRAPH", 2: "L3_SYNTH"}
MODE_MAP = {"dense": "DENSE", "graph": "GRAPH", "hybrid": "HYBRID", "lexical": "HYBRID"}


def _graph_feature_enabled() -> bool:
    return os.getenv("LNSP_GRAPHRAG_ENABLED", "0") == "1"


class GraphHopRequest(BaseModel):
    node_id: str
    max_hops: int = Field(default=2, ge=1, le=4)
    top_k: int = Field(default=25, ge=1, le=200)


def _rrf_merge(result_lists: Sequence[Sequence[Dict[str, Any]]], top_k: int, k: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal rank fusion across multiple retrieval result lists."""
    if not result_lists:
        return []

    scores: Dict[str, float] = {}
    exemplars: Dict[str, Dict[str, Any]] = {}

    def key_for(item: Dict[str, Any]) -> Optional[str]:
        return item.get("cpe_id") or item.get("id") or item.get("doc_id")

    for lst in result_lists:
        for rank, item in enumerate(lst):
            identifier = key_for(item)
            if not identifier:
                continue
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank + 1)
            exemplars.setdefault(identifier, item)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: List[Dict[str, Any]] = []
    for identifier, fusion_score in ordered[:top_k]:
        exemplar = dict(exemplars[identifier])
        exemplar.setdefault("raw_score", exemplar.get("score"))
        exemplar["score"] = fusion_score
        exemplar["rrf_score"] = fusion_score
        fused.append(exemplar)

    # Reassign ranks based on fused order
    for idx, item in enumerate(fused, start=1):
        item["rank"] = idx

    return fused


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

        # load id quality map (doc_id -> quality)
        self.id_quality = {}
        qpath = Path("artifacts/id_quality.jsonl")
        if qpath.exists():
            for line in qpath.open():
                j = json.loads(line)
                self.id_quality[str(j["doc_id"])] = float(j.get("quality", 0.5))
        self.w_cos = float(os.getenv("LNSP_W_COS","0.85"))
        self.w_q   = float(os.getenv("LNSP_W_QUALITY","0.15"))

        # CPESH extraction limits and timeout controls
        self.cpesh_max_k = int(os.getenv("LNSP_CPESH_MAX_K", "5"))
        try:
            self.cpesh_timeout = float(os.getenv("LNSP_CPESH_TIMEOUT_S", "12"))
        except ValueError:
            self.cpesh_timeout = 12.0

        # CPESH cache initialization
        self.cpesh_cache_path = os.getenv("LNSP_CPESH_CACHE", "artifacts/cpesh_cache.jsonl")
        self.cpesh_cache = {}  # In-memory cache for fast lookups
        self._load_cpesh_cache()

        # local llama client (lazy)
        self._llm = None

        # A2: Boot invariant checks
        self._validate_npz_schema(npz_path)
        self._validate_faiss_dimension()

        # Initialize embedder before loading index (needed for probe)
        self.embedder = EmbeddingBackend()

        # Initialize CPESH gating configuration BEFORE loading index (needed for probe)
        self.gate_cfg = CPESHGateConfig(
            q_min=float(os.getenv("LNSP_CPESH_Q_MIN", "0.82")),
            cos_min=float(os.getenv("LNSP_CPESH_COS_MIN", "0.55")),
            nprobe_cpesh=int(os.getenv("LNSP_NPROBE_CPESH", "8")),
            nprobe_fallback=int(os.getenv("LNSP_NPROBE_DEFAULT", "16")),
            lane_overrides={"L1_FACTOID": {"q_min": 0.85}}
        )

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
        """Load FAISS index from metadata or environment variable."""
        # Check for environment variable override first
        env_index_path = os.getenv("LNSP_FAISS_INDEX")
        if env_index_path:
            if Path(env_index_path).exists():
                index_path = env_index_path
                print(f"[RetrievalContext] Using index from LNSP_FAISS_INDEX: {index_path}")
            else:
                print(f"[RetrievalContext] Warning: LNSP_FAISS_INDEX={env_index_path} does not exist, falling back to metadata")
                env_index_path = None

        # If no env var or it doesn't exist, use metadata
        if not env_index_path:
            meta_path = Path("artifacts/faiss_meta.json")
            if meta_path.exists():
                with meta_path.open('r') as f:
                    meta = json.load(f)
                index_path = meta.get("index_path")
            else:
                print(f"[RetrievalContext] faiss_meta.json not found; retrieval will be empty")
                return

        if index_path and Path(index_path).exists():
            self.loaded = self.faiss_db.load(index_path)
            if self.loaded:
                self._build_catalog()
                # Run search smoke test after successful load (if not disabled)
                if os.getenv("LNSP_DISABLE_STARTUP_PROBE", "1") != "1":
                    try:
                        self._probe_search_smoke()
                    except Exception as e:
                        # log-but-don't-fail startup
                        print(f"[startup] probe skipped: {e}")
            else:
                print(f"[RetrievalContext] Failed to load index from {index_path}")
        else:
            print(f"[RetrievalContext] Index path not found or file does not exist")

    def _ensure_llm(self):
        if self._llm is None:
            # Local-only client, no cloud fallback
            from ..llm.local_llama_client import LocalLlamaClient
            endpoint = os.getenv("LNSP_LLM_ENDPOINT","http://localhost:11434")
            model = os.getenv("LNSP_LLM_MODEL","llama3.1:8b")
            self._llm = LocalLlamaClient(endpoint, model)
        return self._llm

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
    def _load_cpesh_cache(self) -> None:
        """Load CPESH cache from disk into memory."""
        cache_path = Path(self.cpesh_cache_path)
        if not cache_path.exists():
            print(f"[RetrievalContext] CPESH cache file not found: {cache_path}")
            return

        try:
            with cache_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Migrate legacy entries to include timestamps
                        entry = migrate_legacy_cache_entry(entry)
                        doc_id = entry.get('doc_id')
                        if doc_id:
                            self.cpesh_cache[doc_id] = entry
                    except json.JSONDecodeError as e:
                        print(f"[RetrievalContext] Invalid JSON in cache line {line_num}: {e}")
                        continue

            print(f"[RetrievalContext] Loaded {len(self.cpesh_cache)} CPESH cache entries")
        except Exception as e:
            print(f"[RetrievalContext] Failed to load CPESH cache: {e}")

    def _save_cpesh_cache(self) -> None:
        """Save CPESH cache from memory to disk."""
        cache_path = Path(self.cpesh_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with cache_path.open('w', encoding='utf-8') as f:
                for doc_id, entry in self.cpesh_cache.items():
                    # Update last_accessed timestamp before saving
                    entry = update_cache_entry_access(entry)
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"[RetrievalContext] Saved {len(self.cpesh_cache)} CPESH cache entries")
        except Exception as e:
            print(f"[RetrievalContext] Failed to save CPESH cache: {e}")

    def get_cpesh_from_cache(self, doc_id: str) -> Optional[CPESH]:
        """Get CPESH from cache by doc_id."""
        if doc_id not in self.cpesh_cache:
            return None

        # Update access timestamp and count
        entry = update_cache_entry_access(self.cpesh_cache[doc_id])

        # Extract CPESH data
        cpesh_data_raw = entry.get('cpesh', {})
        if not isinstance(cpesh_data_raw, dict):
            return None

        try:
            cpesh_data = migrate_cpesh_record({**cpesh_data_raw})
            cpesh_data['access_count'] = entry.get('access_count', cpesh_data.get('access_count', 0))
            return CPESH(**cpesh_data)
        except Exception as e:
            print(f"[RetrievalContext] Failed to parse cached CPESH for {doc_id}: {e}")
            return None

    def put_cpesh_to_cache(self, doc_id: str, cpesh: CPESH) -> None:
        """Store CPESH in cache by doc_id."""
        now = get_iso_timestamp()
        payload = cpesh.model_dump(by_alias=True, exclude_none=True)
        # Maintain legacy keys for compatibility
        payload['concept'] = cpesh.concept
        payload['probe'] = cpesh.probe
        payload['expected'] = cpesh.expected
        payload['soft_negative'] = cpesh.soft_negative
        payload['hard_negative'] = cpesh.hard_negative
        payload['soft_sim'] = cpesh.soft_sim
        payload['hard_sim'] = cpesh.hard_sim
        payload['created_at'] = cpesh.created_at or now
        payload['last_accessed'] = now
        payload['access_count'] = max(1, int(cpesh.access_count or 0))

        entry = {
            'doc_id': doc_id,
            'cpesh': payload,
            'access_count': payload['access_count'],
            'quality': payload.get('quality'),
            'cosine': payload.get('soft_sim'),
        }

        self.cpesh_cache[doc_id] = entry

    def close(self) -> None:
        """Save CPESH cache to disk before shutdown."""
        if self.cpesh_cache:
            self._save_cpesh_cache()

    def _norm_hit(self, h: dict) -> SearchItem:
        cpe_id = h.get("cpe_id") or h.get("id") or h.get("uuid") or h.get("rid") or ""
        doc_id = h.get("doc_id") or (h.get("metadata") or {}).get("doc_id")

        # Extract hydrated fields
        metadata = h.get("metadata") or {}
        concept_text = metadata.get("concept_text") or h.get("concept_text")
        tmd_bits = h.get("tmd_bits", 0)
        lane_index = h.get("lane_index", 0)
        score = h.get("score")

        # Format TMD code as D.T.M string
        if tmd_bits:
            domain = (tmd_bits >> 12) & 0xF
            task = (tmd_bits >> 7) & 0x1F
            modifier = (tmd_bits >> 1) & 0x3F
            tmd_code = f"{domain}.{task}.{modifier}"
        else:
            tmd_code = "0.0.0"

        q = self.id_quality.get(str(doc_id), 0.5)
        final = None if score is None else (self.w_cos*float(score) + self.w_q*float(q))

        return SearchItem(
            id=cpe_id,
            doc_id=doc_id,
            score=score,
            why=h.get("why"),
            concept_text=concept_text,
            tmd_code=tmd_code,
            lane_index=lane_index,
            quality=q,
            final_score=final,
            word_count=metadata.get("word_count"),
            tmd_confidence=metadata.get("tmd_confidence")
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

        # Apply CPESH gating logic
        import time
        gate = apply_lane_overrides(self.gate_cfg, req.lane)
        cpesh_entry: Optional[Dict[str, Any]] = None
        cpesh_payload: Optional[Dict[str, Any]] = None
        if hasattr(self, "cpesh_cache") and req.doc_id:
            cpesh_entry = self.cpesh_cache.get(req.doc_id)
            if isinstance(cpesh_entry, dict):
                inner = cpesh_entry.get('cpesh')
                cpesh_payload = inner if isinstance(inner, dict) else cpesh_entry
            elif isinstance(cpesh_entry, CPESH):
                cpesh_payload = cpesh_entry.model_dump(by_alias=True)

        t0 = time.time()
        used_cpesh = False
        chosen_nprobe = gate.nprobe_fallback
        quality_score = None
        cosine_score = None

        gate_source = cpesh_payload or cpesh_entry

        if gate_source and should_use_cpesh(gate_source, gate):
            used_cpesh = True
            chosen_nprobe = gate.nprobe_cpesh
            quality_score = gate_source.get("quality") if isinstance(gate_source, dict) else None
            cosine_score = gate_source.get("cosine") if isinstance(gate_source, dict) else None

            base_candidates = self.faiss_db.search_legacy(
                fused_query,
                topk=req.top_k,
                use_lightrag=True,
                nprobe=chosen_nprobe,
            ) or []

            expected_text = None
            if isinstance(cpesh_payload, dict):
                expected_text = cpesh_payload.get("expected_answer") or cpesh_payload.get("expected")

            if expected_text:
                expected_vec_raw = self.embedder.encode([expected_text])[0].astype(np.float32)
                expected_norm = float(np.linalg.norm(expected_vec_raw))
                expected_vec = (
                    expected_vec_raw if expected_norm == 0 else (expected_vec_raw / expected_norm)
                )
                expected_candidates = self.faiss_db.search_legacy(
                    expected_vec,
                    topk=req.top_k,
                    use_lightrag=True,
                    nprobe=gate.nprobe_cpesh,
                ) or []
                candidates = _rrf_merge([base_candidates, expected_candidates], req.top_k)
            else:
                candidates = base_candidates
        else:
            candidates = self.faiss_db.search_legacy(
                fused_query,
                topk=req.top_k,
                use_lightrag=True,
                nprobe=chosen_nprobe,
            ) or []

        latency_ms = (time.time() - t0) * 1000.0

        # Log gating decision
        try:
            log_gating_decision(
                query_id=trace_id or "unknown",
                lane=req.lane,
                used_cpesh=used_cpesh,
                quality=quality_score,
                cosine=cosine_score,
                chosen_nprobe=chosen_nprobe,
                latency_ms=latency_ms
            )
        except Exception as e:
            print(f"Failed to log gating decision: {e}")

        mode = settings.RETRIEVAL_MODE if settings.RETRIEVAL_MODE != "HYBRID" else "DENSE"

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
        # re-rank by final_score (falls back to score)
        items.sort(key=lambda x: (x.final_score if x.final_score is not None else (x.score or 0.0)), reverse=True)
        print(f"Trace {trace_id}: Normalized {len(items)} items.")

        # --- Optional per-item CPESH extraction ---
        if req.return_cpesh and items:
            # cap extraction to avoid long calls
            req_k = req.cpesh_k if req.cpesh_k is not None else self.cpesh_max_k
            k = min(len(items), max(0, req_k))
            llm = self._ensure_llm()
            # embed the user query once for sim calc (cosine) when cpesh_mode=full
            qvec = None
            if req.cpesh_mode == "full":
                qvec_raw = self.embedder.encode([req.q])[0].astype(np.float32)
                qvec_norm = float(np.linalg.norm(qvec_raw))
                qvec = qvec_raw / qvec_norm if qvec_norm > 0 else qvec_raw
            for i in range(k):
                it = items[i]
                text = (it.concept_text or "").strip()
                if not text:
                    continue

                # Check cache first (cache hit - update last_accessed)
                cached_cpesh = self.get_cpesh_from_cache(it.doc_id or "")
                if cached_cpesh:
                    print(f"Trace {trace_id}: CPESH cache hit for {it.id}")
                    it.cpesh = cached_cpesh
                    continue

                # Cache miss - extract and store
                print(f"Trace {trace_id}: CPESH cache miss for {it.id}, extracting...")
                prompt = (
                    "Return JSON only for CPESH_EXTRACT.\n"
                    f'Factoid: "{text}"\n'
                    'Rules:\n'
                    '- concept: main entity/topic from the factoid\n'
                    '- probe: question about the concept (e.g., "What is X?")\n'
                    '- expected: correct answer sentence from the factoid\n'
                    '- soft_negative: DIFFERENT sentence from the factoid that is related but does NOT answer the probe\n'
                    '- hard_negative: completely UNRELATED fact (e.g., about photosynthesis, quantum physics, or plate tectonics)\n'
                    '- insufficient_evidence: true only if factoid lacks clear information\n'
                    '{"concept":"...","probe":"...","expected":"...",'
                    '"soft_negative":"...","hard_negative":"...",'
                    '"insufficient_evidence":false}'
                )
                try:
                    j = llm.complete_json(prompt, timeout_s=self.cpesh_timeout)
                    cp = CPESH(
                        concept=j.get("concept"),
                        probe=j.get("probe"),
                        expected=j.get("expected"),
                        soft_negative=j.get("soft_negative"),
                        hard_negative=j.get("hard_negative"),
                    )
                    # sims vs the query embedding if requested
                    if req.cpesh_mode == "full" and qvec is not None:
                        if cp.soft_negative:
                            sv_raw = self.embedder.encode([cp.soft_negative])[0].astype(np.float32)
                            sv_norm = float(np.linalg.norm(sv_raw))
                            sv = sv_raw / sv_norm if sv_norm > 0 else sv_raw
                            cp.soft_sim = float(np.dot(qvec, sv))
                        if cp.hard_negative:
                            hv_raw = self.embedder.encode([cp.hard_negative])[0].astype(np.float32)
                            hv_norm = float(np.linalg.norm(hv_raw))
                            hv = hv_raw / hv_norm if hv_norm > 0 else hv_raw
                            cp.hard_sim = float(np.dot(qvec, hv))
                    it.cpesh = cp

                    # Store in cache
                    if it.doc_id:
                        self.put_cpesh_to_cache(it.doc_id, cp)
                        print(f"Trace {trace_id}: CPESH cached for {it.id}")
                except Exception as e:
                    print(f"Trace {trace_id}: CPESH extract failed for {it.id}: {e}")
                    continue

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

        # Quality check
        quality_warning = False
        if items:
            total_words = sum(item.word_count for item in items if item.word_count is not None)
            avg_words = total_words / len(items) if len(items) > 0 else 0
            if avg_words < 150:
                quality_warning = True

        data_version = "v2" if "_v2" in self.npz_path else "v1"
        return SearchResponse(
            lane=req.lane,
            mode=mode,
            items=items,
            trace_id=trace_id,
            diagnostics=diag_payload,
            insufficient_evidence=insufficient,
            data_version=data_version,
            quality_warning=quality_warning
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

    # Feature-gated GraphRAG router
    if os.getenv("LNSP_GRAPHRAG_ENABLED", "0") == "1":
        try:
            from .graph import router as graph_router
            app.include_router(graph_router)
            print("[GraphRAG] Router loaded successfully")
        except Exception as e:
            # don't crash API if graph import fails
            print(f"[GraphRAG] Router load skipped: {e}")

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

    @app.get("/health/faiss")
    def health_faiss() -> Dict[str, Any]:
        """Robust FAISS health: never raises; reports what's known."""
        info = {
            "loaded": False, "trained": None, "ntotal": None,
            "dim": None, "type": None, "metric": None, "nlist": None,
            "error": None,
        }
        try:
            # Try to get context safely
            ctx = None
            try:
                ctx = get_context()
            except Exception as ctx_error:
                info["error"] = f"Context initialization failed: {ctx_error}"

            # Get index metadata from the new index_meta.json if available
            index_meta = {}
            try:
                with open("artifacts/index_meta.json", "r") as f:
                    all_meta = json.load(f)
                    # Get the most recent index metadata
                    if all_meta:
                        latest_path = max(all_meta.keys(), key=lambda x: all_meta[x].get("build_seconds", 0))
                        index_meta = all_meta[latest_path]
            except Exception:
                pass

            # Get basic info from files even if context failed
            if index_meta:
                info["type"] = index_meta.get("type", None)
                info["metric"] = index_meta.get("metric", None)
                info["nlist"] = index_meta.get("nlist", None)
                info["ntotal"] = index_meta.get("count", None)
                info["loaded"] = info["ntotal"] is not None and info["ntotal"] > 0

            # If context is available, get additional info
            if ctx is not None:
                idx = getattr(ctx, "index", None)
                info["type"] = info["type"] or getattr(ctx, "index_type", None)
                info["metric"] = info["metric"] or getattr(ctx, "metric", None)

                # FAISS attrs vary by index type; guard everything
                if idx is not None:
                    info["trained"] = bool(getattr(idx, "is_trained", False))
                    info["ntotal"] = info["ntotal"] or int(getattr(idx, "ntotal", 0))
                    info["dim"] = (getattr(idx, "d", None) or getattr(idx, "dim", None))
                    if hasattr(idx, "nlist"):
                        info["nlist"] = info["nlist"] or getattr(idx, "nlist", None)
                    info["loaded"] = info["loaded"] or (info["ntotal"] is not None and info["ntotal"] > 0)

        except Exception as e:
            info["error"] = str(e)

        return info

    @app.get("/cpesh/segments")
    def cpesh_segments() -> Dict[str, Any]:
        """List CPESH Parquet segments with basic metadata."""
        manifest_path = Path("artifacts/cpesh_manifest.jsonl")
        if not manifest_path.exists():
            return {"segments": [], "count": 0}

        segments: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                segment_path = entry.get("path")
                size_bytes = None
                if segment_path and Path(segment_path).exists():
                    try:
                        size_bytes = os.path.getsize(segment_path)
                    except OSError:
                        size_bytes = None
                segments.append(
                    {
                        "segment_id": entry.get("segment_id"),
                        "path": segment_path,
                        "rows": entry.get("rows"),
                        "created_utc": entry.get("created_utc"),
                        "bytes": size_bytes,
                    }
                )

        return {"segments": segments, "count": len(segments)}

    @app.get("/cache/stats")
    def cache_stats() -> Dict[str, Any]:
        """Return CPESH cache statistics."""
        ctx = get_context()

        if not ctx.cpesh_cache:
            return {
                "entries": 0,
                "oldest_created_at": None,
                "newest_last_accessed": None,
                "p50_access_age": None,
                "top_docs_by_access": []
            }
        
        entries = list(ctx.cpesh_cache.values())
        
        # Extract timestamps
        created_ats = []
        last_accesseds = []
        access_counts = []
        
        for entry in entries:
            cpesh_data = entry.get('cpesh', {})
            if cpesh_data:
                created_at = cpesh_data.get('created_at')
                last_accessed = cpesh_data.get('last_accessed')
                access_count = entry.get('access_count', 0)
                
                if created_at:
                    created_ats.append(created_at)
                if last_accessed:
                    last_accesseds.append(last_accessed)
                access_counts.append(access_count)
        
        # Calculate statistics
        oldest_created = min(created_ats) if created_ats else None
        newest_accessed = max(last_accesseds) if last_accesseds else None
        
        # Calculate p50 access age (median age in hours)
        p50_age = None
        if last_accesseds:
            from datetime import datetime
            now = datetime.now()
            ages = []
            for ts in last_accesseds:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    age_hours = (now - dt.replace(tzinfo=None)).total_seconds() / 3600
                    ages.append(age_hours)
                except:
                    pass
            
            if ages:
                ages.sort()
                mid = len(ages) // 2
                p50_age = ages[mid] if len(ages) % 2 != 0 else (ages[mid-1] + ages[mid]) / 2
        
        # Top docs by access count
        doc_access = {}
        for entry in entries:
            doc_id = entry.get('doc_id')
            access_count = entry.get('access_count', 0)
            if doc_id:
                doc_access[doc_id] = doc_access.get(doc_id, 0) + access_count
        
        top_docs = sorted(doc_access.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "entries": len(entries),
            "oldest_created_at": oldest_created,
            "newest_last_accessed": newest_accessed,
            "p50_access_age": round(p50_age, 2) if p50_age else None,
            "top_docs_by_access": top_docs
        }

# --- SLO snapshot store ---
SLO_PATH = os.getenv("LNSP_SLO_PATH", "artifacts/metrics_slo.json")

@app.post("/metrics/slo")
def slo_ingest(snapshot: dict):
    """
    Accept a metrics snapshot from an external eval harness.
    Example fields (add what you have): {
      "timestamp_utc": "...",
      "hit_at_1": 0.47, "hit_at_3": 0.57,
      "p50_ms": 42.1, "p95_ms": 310.0,
      "notes": "nprobe_default=24, cpesh gate 0.85/0.55"
    }
    """
    try:
        snapshot.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        os.makedirs(os.path.dirname(SLO_PATH), exist_ok=True)
        with open(SLO_PATH, "w") as f:
            json.dump(snapshot, f)
        return {"ok": True, "path": SLO_PATH}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/metrics/slo")
def slo_get():
    if not os.path.exists(SLO_PATH):
        return {"ok": True, "present": False, "snapshot": None}
    with open(SLO_PATH) as f:
        snap = json.load(f)
    snap["present"] = True
    snap["ok"] = True
    return snap

    @app.get("/metrics/gating")
    def gating_metrics() -> Dict[str, Any]:
        """Return CPESH gating usage metrics."""
        return get_gating_metrics()

    @app.post("/graph/search")
    def graph_search(req: SearchRequest) -> Dict[str, Any]:
        """Execute a GraphRAG query when the feature flag is enabled."""
        if not _graph_feature_enabled():
            raise HTTPException(status_code=501, detail="GraphRAG disabled (set LNSP_GRAPHRAG_ENABLED=1)")

        lane = req.lane or "L2_GRAPH"
        try:
            from ..adapters.lightrag import graphrag_runner as gr

            config_path = Path(os.getenv("LNSP_GRAPHRAG_CONFIG", "configs/lightrag.yml"))
            cfg = gr._load_config(config_path)
            LightRAG = gr._load_lightrag()
            query_item = gr.QueryItem(lane=lane, text=req.q)
            result = gr._run_query(LightRAG, cfg, query_item)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"GraphRAG runtime error: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=500, detail=f"GraphRAG error: {exc}") from exc

        return {"lane": lane, "query": req.q, "result": result}

    @app.post("/graph/hop")
    def graph_hop(payload: GraphHopRequest) -> Dict[str, Any]:
        """Return neighboring nodes from the local knowledge-graph edges."""
        if not _graph_feature_enabled():
            raise HTTPException(status_code=501, detail="GraphRAG disabled (set LNSP_GRAPHRAG_ENABLED=1)")

        edges_path = Path(os.getenv("LNSP_GRAPH_EDGES", "artifacts/kg/edges.jsonl"))
        if not edges_path.exists():
            raise HTTPException(status_code=404, detail=f"Edges file not found at {edges_path}")

        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        with edges_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    edge = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = edge.get("subj") or edge.get("src")
                dst = edge.get("obj") or edge.get("dst")
                if not src or not dst:
                    continue
                payload_edge = {
                    "source": src,
                    "target": dst,
                    "predicate": edge.get("pred") or edge.get("rel"),
                    "confidence": edge.get("confidence"),
                }
                adjacency.setdefault(src, []).append(payload_edge)
                # treat edges as undirected for exploration convenience
                reverse = {
                    "source": dst,
                    "target": src,
                    "predicate": edge.get("pred") or edge.get("rel"),
                    "confidence": edge.get("confidence"),
                }
                adjacency.setdefault(dst, []).append(reverse)

        start = payload.node_id
        visited: Dict[str, int] = {start: 0}
        queue = deque([(start, 0)])
        results: List[Dict[str, Any]] = []

        while queue and len(results) < payload.top_k:
            node, depth = queue.popleft()
            if depth >= payload.max_hops:
                continue
            for edge in adjacency.get(node, []):
                neighbor = edge["target"]
                next_depth = depth + 1
                prev = visited.get(neighbor)
                if prev is not None and prev <= next_depth:
                    continue
                visited[neighbor] = next_depth
                results.append({
                    "source": edge["source"],
                    "target": neighbor,
                    "predicate": edge.get("predicate"),
                    "confidence": edge.get("confidence"),
                    "hops": next_depth,
                })
                if len(results) >= payload.top_k:
                    break
                queue.append((neighbor, next_depth))

        return {
            "start": start,
            "max_hops": payload.max_hops,
            "results": results,
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
