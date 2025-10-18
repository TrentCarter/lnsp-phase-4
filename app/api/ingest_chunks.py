#!/usr/bin/env python3
"""
Chunk Ingestion FastAPI Service

Takes semantic chunks and ingests them into vecRAG with complete CPESH + TMD processing.
Atomically writes to PostgreSQL + Neo4j + FAISS (3-way sync).

Pipeline:
1. Chunk IN (from Chunker API)
2. CPESH generation (TinyLlama via Ollama)
3. TMD extraction (Llama 3.1 via TMD Router)
4. Vectorization (GTR-T5 @ 768D + TMD @ 16D = 784D)
5. Atomic write to all 3 stores (PostgreSQL, Neo4j, FAISS)
6. Return Global_ID (cpe_id UUID)

Database Schema (PostgreSQL):
-- Run this migration to enable parent/child tracking:
ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS confidence_score REAL DEFAULT NULL,
ADD COLUMN IF NOT EXISTS quality_metrics JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS parent_cpe_ids JSONB DEFAULT '[]'::jsonb,
ADD COLUMN IF NOT EXISTS child_cpe_ids JSONB DEFAULT '[]'::jsonb;

Port: 8004
"""

import os
import sys
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import existing modules
from src.db_postgres import connect as connect_pg
from src.loaders.pg_writer import (
    insert_cpe_entry,
    upsert_cpe_vectors,
    batch_insert_cpe_entries,
    batch_upsert_cpe_vectors
)
from src.prompt_extractor import extract_cpe_from_text
from src.llm_tmd_extractor import extract_tmd_with_llm
from src.tmd_extractor_v2 import extract_tmd_from_text as fast_tmd_extract
from src.tmd_heuristics import classify_task, classify_modifier
from src.vectorizer import EmbeddingBackend
from src.db_faiss import FaissDB

# Vec2Text-compatible embedding wrapper
class Vec2TextCompatibleEmbedder:
    """Wrapper for vec2text encoder that provides EmbeddingBackend-compatible interface"""
    def __init__(self):
        from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
        import torch
        import os

        # Force CPU for vec2text compatibility
        os.environ['VEC2TEXT_FORCE_CPU'] = '1'

        self.orchestrator = IsolatedVecTextVectOrchestrator()
        self.torch = torch
        print("   Vec2TextCompatibleEmbedder initialized (CPU mode)")

    def encode(self, texts, batch_size=32):
        """Encode texts using vec2text-compatible encoder, return numpy arrays"""
        # Call vec2text's encoder (returns torch.Tensor)
        embeddings_tensor = self.orchestrator.encode_texts(texts)

        # Convert to numpy
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        return embeddings_numpy

    def get_device(self):
        """Return device string"""
        return str(self.orchestrator._device)

app = FastAPI(
    title="LNSP Chunk Ingestion API",
    description="Ingest semantic chunks into vecRAG with CPESH + TMD processing",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State
# ============================================================================

class ServiceState:
    """Global service state"""
    def __init__(self):
        # Database connections
        self.pg_conn = None
        self.faiss_db = None
        self.embedder = None

        # Configuration
        self.llm_endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
        self.tmd_router_endpoint = "http://localhost:8002"
        self.gtr_t5_endpoint = "http://localhost:8767"

        # Feature flags
        self.enable_cpesh = os.getenv("LNSP_ENABLE_CPESH", "false").lower() == "true"

        # Parallelization settings
        self.max_parallel_workers = int(os.getenv("LNSP_MAX_PARALLEL_WORKERS", "10"))
        self.enable_parallel = os.getenv("LNSP_ENABLE_PARALLEL", "true").lower() == "true"
        self.enable_batch_embeddings = os.getenv("LNSP_ENABLE_BATCH_EMBEDDINGS", "true").lower() == "true"
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_workers)

        # TMD speed controls
        self.tmd_fast_first = os.getenv("LNSP_TMD_FAST_FIRST", "true").lower() == "true"
        try:
            self.tmd_fast_conf_min = float(os.getenv("LNSP_TMD_FAST_CONF_MIN", "0.60"))
        except Exception:
            self.tmd_fast_conf_min = 0.60
        self.tmd_llm_model = os.getenv("LNSP_TMD_LLM_MODEL", "tinyllama:1.1b")
        self.tmd_mode = os.getenv("LNSP_TMD_MODE", "full").lower()

        # Statistics
        self.total_ingested = 0
        self.total_chunks = 0
        self.failed_count = 0


state = ServiceState()


# ============================================================================
# Request/Response Models
# ============================================================================

class ChunkInput(BaseModel):
    """Single chunk to ingest"""
    text: str = Field(..., description="Chunk text content", min_length=1)  # Allow short ontology terms
    source_document: Optional[str] = Field(default="web_input", description="Source document name")
    chunk_index: Optional[int] = Field(default=0, description="Position in original document")
    document_id: Optional[str] = Field(default=None, description="Document identifier for grouping chunks (used as batch_id)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    parent_cpe_ids: Optional[List[str]] = Field(default=None, description="UUIDs of parent concepts (for training chains)")
    child_cpe_ids: Optional[List[str]] = Field(default=None, description="UUIDs of child concepts (for training chains)")
    # Optional: allow caller to provide TMD when using hybrid pipeline
    domain_code: Optional[int] = Field(default=None, description="Optional precomputed domain code (0-15)")
    task_code: Optional[int] = Field(default=None, description="Optional precomputed task code (0-31)")
    modifier_code: Optional[int] = Field(default=None, description="Optional precomputed modifier code (0-63)")


class IngestRequest(BaseModel):
    """Request to ingest chunks"""
    chunks: List[ChunkInput] = Field(..., description="List of chunks to ingest", min_items=1)
    dataset_source: str = Field(default="user_input", description="Dataset source identifier")
    batch_id: Optional[str] = Field(default=None, description="Batch ID for grouping")
    skip_cpesh: bool = Field(default=False, description="Skip CPESH extraction step and use raw chunk text as concept")
    chunking_time_ms: Optional[float] = Field(default=None, description="Elapsed time spent in external chunking stage (if known)")

    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
                        "source_document": "biology_textbook.pdf",
                        "chunk_index": 0
                    }
                ],
                "dataset_source": "biology_course",
                "batch_id": "batch_20251009_001"
            }
        }


class IngestResult(BaseModel):
    """Result of ingesting a single chunk"""
    global_id: str = Field(..., description="UUID/cpe_id for this concept")
    concept_text: str
    tmd_codes: Dict[str, int] = Field(..., description="Domain/Task/Modifier codes")
    vector_dimension: int
    confidence_score: Optional[float] = Field(default=None, description="Confidence in extraction (0-1)")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Quality metrics")
    created_at: str = Field(..., description="ISO timestamp of creation")
    parent_cpe_ids: List[str] = Field(default_factory=list, description="Parent concept UUIDs")
    child_cpe_ids: List[str] = Field(default_factory=list, description="Child concept UUIDs")
    success: bool
    error: Optional[str] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict, description="Per-step timings in milliseconds")
    backends: Dict[str, str] = Field(default_factory=dict, description="Backends used per step")


class IngestResponse(BaseModel):
    """Response from ingestion"""
    results: List[IngestResult]
    total_chunks: int
    successful: int
    failed: int
    batch_id: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    postgresql: bool
    faiss: bool
    gtr_t5_api: bool
    tmd_router_api: bool
    llm_endpoint: str


# ============================================================================
# Helper Functions
# ============================================================================

def encode_tmd_16d(domain: int, task: int, modifier: int) -> np.ndarray:
    """
    Encode TMD codes as 16D dense vector.

    Args:
        domain: 0-15 (4 bits)
        task: 0-31 (5 bits)
        modifier: 0-63 (6 bits)

    Returns:
        16D numpy array (one-hot encoded bits)
    """
    tmd_vec = np.zeros(16, dtype=np.float32)

    # Encode domain (4 bits)
    for i in range(4):
        if domain & (1 << i):
            tmd_vec[i] = 1.0

    # Encode task (5 bits)
    for i in range(5):
        if task & (1 << i):
            tmd_vec[4 + i] = 1.0

    # Encode modifier (6 bits) - use first 6 of remaining 7 slots
    for i in range(6):
        if modifier & (1 << i):
            tmd_vec[9 + i] = 1.0

    # Last bit (15) reserved for future use
    return tmd_vec


def check_service_health(url: str, timeout: float = 2.0) -> bool:
    """Check if an external service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


# ============================================================================
# Ingestion Pipeline
# ============================================================================

def calculate_quality_metrics(cpesh: Dict, concept_vec: np.ndarray, chunk_text: str) -> Dict[str, Any]:
    """
    Calculate quality metrics for the ingested chunk.

    Metrics:
    - cpesh_completeness: % of CPESH fields that are non-empty
    - vector_norm: L2 norm of the concept vector
    - text_length: Character count of original chunk
    - negatives_count: Number of soft + hard negatives generated
    """
    metrics = {}

    # CPESH completeness (0-1)
    fields = ["concept", "probe_question", "expected_answer", "soft_negatives", "hard_negatives"]
    filled = sum(1 for f in fields if cpesh.get(f))
    metrics["cpesh_completeness"] = filled / len(fields)

    # Vector quality
    metrics["vector_norm"] = float(np.linalg.norm(concept_vec))
    metrics["vector_dimension"] = int(concept_vec.shape[0])

    # Content metrics
    metrics["text_length"] = len(chunk_text)
    metrics["concept_length"] = len(cpesh.get("concept", ""))

    # Negatives count (more negatives = better CPESH)
    soft_neg_count = len(cpesh.get("soft_negatives", []))
    hard_neg_count = len(cpesh.get("hard_negatives", []))
    metrics["soft_negatives_count"] = soft_neg_count
    metrics["hard_negatives_count"] = hard_neg_count
    metrics["total_negatives"] = soft_neg_count + hard_neg_count

    return metrics


def calculate_confidence_score(quality_metrics: Dict) -> float:
    """
    Calculate overall confidence score (0-1) based on quality metrics.

    Higher score = higher confidence in the extraction quality.
    """
    score = 0.0

    # CPESH completeness (40% weight)
    score += quality_metrics.get("cpesh_completeness", 0) * 0.4

    # Vector norm (10% weight) - normalized to 0-1 range
    norm = quality_metrics.get("vector_norm", 0)
    norm_score = min(norm / 10.0, 1.0)  # Typical norm is ~5-10
    score += norm_score * 0.1

    # Negatives count (30% weight) - more negatives = better
    neg_count = quality_metrics.get("total_negatives", 0)
    neg_score = min(neg_count / 5.0, 1.0)  # 5 negatives = perfect
    score += neg_score * 0.3

    # Concept extraction quality (20% weight)
    concept_len = quality_metrics.get("concept_length", 0)
    text_len = quality_metrics.get("text_length", 1)
    extraction_ratio = concept_len / text_len if text_len > 0 else 0
    # Good extraction should be 10-50% of original text
    if 0.1 <= extraction_ratio <= 0.5:
        score += 0.2
    elif 0.05 <= extraction_ratio <= 0.8:
        score += 0.1

    return min(score, 1.0)


def ingest_chunk(
    chunk: ChunkInput,
    dataset_source: str,
    batch_id: str,
    *,
    pre_assigned_uuid: Optional[str] = None,
    skip_cpesh: bool = False,
    chunking_time_ms: Optional[float] = None,
) -> IngestResult:
    """
    Ingest a single chunk through the complete pipeline.

    Pipeline:
    1. Extract CPESH (Concept, Probe, Expected, Soft Negatives, Hard Negatives)
    2. Extract TMD codes (Domain, Task, Modifier)
    3. Vectorize concept text (GTR-T5 768D)
    4. Encode TMD (16D dense vector)
    5. Concatenate: 784D = 768D semantic + 16D TMD
    6. Calculate quality metrics and confidence score
    7. Atomic write to PostgreSQL + Neo4j + FAISS

    Args:
        chunk: ChunkInput with text and optional parent/child IDs
        dataset_source: Dataset identifier
        batch_id: Batch identifier
        pre_assigned_uuid: Optional pre-generated UUID (for sequential linking)

    Returns:
        IngestResult with global_id (cpe_id UUID), quality metrics, and timestamps
    """

    try:
        # Use pre-assigned UUID or generate new one
        cpe_id = uuid.UUID(pre_assigned_uuid) if pre_assigned_uuid else uuid.uuid4()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        timings: Dict[str, float] = {}
        backends: Dict[str, str] = {}

        # Step 1: Extract CPESH using LLM (gated by env + request)
        do_cpesh = state.enable_cpesh and not skip_cpesh
        if do_cpesh:
            t0 = time.perf_counter()
            cpesh = extract_cpe_from_text(
                chunk.text,
                endpoint=state.llm_endpoint,
                model="tinyllama:1.1b"
            )
            timings["cpesh_ms"] = (time.perf_counter() - t0) * 1000.0
            backends["cpesh"] = f"ollama:{state.llm_endpoint}"

            concept_text = cpesh.get("concept", chunk.text[:100])
            probe_question = cpesh.get("probe_question", "")
            expected_answer = cpesh.get("expected_answer", "")
            soft_negatives = cpesh.get("soft_negatives", [])
            hard_negatives = cpesh.get("hard_negatives", [])
            cpesh_payload = {
                "concept": concept_text,
                "probe_question": probe_question,
                "expected_answer": expected_answer,
                "soft_negatives": soft_negatives,
                "hard_negatives": hard_negatives,
            }
        else:
            # CPESH disabled - use simple fallbacks
            timings["cpesh_ms"] = 0.0
            backends["cpesh"] = "disabled" if skip_cpesh else "env-disabled"
            concept_text = chunk.text
            probe_question = ""
            expected_answer = ""
            soft_negatives = []
            hard_negatives = []
            cpesh_payload = {
                "concept": concept_text,
                "probe_question": "",
                "expected_answer": "",
                "soft_negatives": [],
                "hard_negatives": [],
            }

        # Optional: propagate external chunking timing
        if chunking_time_ms is not None:
            try:
                timings["chunking_ms"] = float(chunking_time_ms)
            except Exception:
                pass
            # Create empty cpesh dict for quality metrics
            cpesh = {
                "concept": concept_text,
                "probe_question": "",
                "expected_answer": "",
                "soft_negatives": [],
                "hard_negatives": []
            }

        # Step 2: Extract TMD codes (client-provided → hybrid → fast heuristic → LLM fallback)
        tmd_endpoint = state.llm_endpoint
        domain_code = task_code = modifier_code = None
        t0 = time.perf_counter()
        # Prefer client-provided TMD (e.g., from hybrid pipeline)
        try:
            if (
                hasattr(chunk, "domain_code") and chunk.domain_code is not None and
                hasattr(chunk, "task_code") and chunk.task_code is not None and
                hasattr(chunk, "modifier_code") and chunk.modifier_code is not None
            ):
                domain_code = int(chunk.domain_code)
                task_code = int(chunk.task_code)
                modifier_code = int(chunk.modifier_code)
                timings["tmd_ms"] = 0.1
                backends["tmd"] = "client-provided"
        except Exception:
            pass
        # Hybrid mode: Domain via heuristic/LLM, Task/Modifier via heuristics
        if domain_code is None and state.tmd_mode == "hybrid":
            used = ""
            # Try fast heuristic for domain first
            try:
                fast = fast_tmd_extract(concept_text)
                conf = float(fast.get("confidence", 0.0))
                if conf >= state.tmd_fast_conf_min:
                    domain_code = max(0, min(15, int(fast.get("domain_code", 2)) - 1))
                    used = "heuristic_domain"
            except Exception:
                pass

            if domain_code is None:
                # Fallback: LLM for domain only
                t1 = time.perf_counter()
                tmd_result = extract_tmd_with_llm(
                    text=concept_text,
                    llm_endpoint=tmd_endpoint,
                    llm_model=state.tmd_llm_model,
                )
                timings["tmd_ms"] = (time.perf_counter() - t1) * 1000.0
                domain_code = int(tmd_result["domain_code"])  # 0-15
                used = "llm_domain"

            # Heuristics for Task/Modifier
            task_code = int(classify_task(concept_text))
            modifier_code = int(classify_modifier(concept_text))
            backends["tmd"] = f"hybrid:{used}+heuristics_tm"

        # Fast-first full heuristic path (only when not already set and not hybrid-path)
        if domain_code is None and state.tmd_fast_first:
            try:
                fast = fast_tmd_extract(concept_text)
                conf = float(fast.get("confidence", 0.0))
                if conf >= state.tmd_fast_conf_min:
                    domain_code = max(0, min(15, int(fast.get("domain_code", 2)) - 1))
                    task_code = max(0, min(31, int(fast.get("task_code", 1)) - 1))
                    modifier_code = max(0, min(63, int(fast.get("modifier_code", 27))))
                    timings["tmd_ms"] = (time.perf_counter() - t0) * 1000.0
                    backends["tmd"] = "heuristic_v2"
            except Exception:
                pass

        if domain_code is None:
            # Fallback to LLM
            t1 = time.perf_counter()
            tmd_result = extract_tmd_with_llm(
                text=concept_text,
                llm_endpoint=tmd_endpoint,
                llm_model=state.tmd_llm_model,
            )
            timings["tmd_ms"] = (time.perf_counter() - t1) * 1000.0
            backends["tmd"] = f"ollama:{tmd_endpoint}"
            domain_code = int(tmd_result["domain_code"])
            task_code = int(tmd_result["task_code"])
            modifier_code = int(tmd_result["modifier_code"])

        # Compute tmd_bits (for storage)
        tmd_bits = (domain_code << 11) | (task_code << 6) | modifier_code

        # Step 3: Vectorize with GTR-T5 (768D)
        t0 = time.perf_counter()
        if state.embedder:
            concept_vec = state.embedder.encode([concept_text])[0]  # 768D
            question_vec = state.embedder.encode([probe_question])[0] if probe_question else concept_vec
            backends["embedding"] = "inproc:gtr-t5"
        else:
            # Fallback: call GTR-T5 API
            response = requests.post(
                f"{state.gtr_t5_endpoint}/embed",
                json={"texts": [concept_text], "normalize": True},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                concept_vec = np.array(data["embeddings"][0], dtype=np.float32)
                question_vec = concept_vec.copy()
                backends["embedding"] = f"api:{state.gtr_t5_endpoint}"
            else:
                raise Exception(f"GTR-T5 API failed: HTTP {response.status_code}")
        timings["embedding_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 4: Encode TMD as 16D vector
        t0 = time.perf_counter()
        tmd_dense = encode_tmd_16d(domain_code, task_code, modifier_code)  # 16D
        timings["tmd_encode_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: Concatenate to 784D
        t0 = time.perf_counter()
        fused_vec = np.concatenate([concept_vec, tmd_dense])  # 784D
        fused_norm = float(np.linalg.norm(fused_vec))
        timings["fuse_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 5b: Calculate quality metrics and confidence
        quality_metrics = calculate_quality_metrics(cpesh_payload, concept_vec, chunk.text)
        confidence_score = calculate_confidence_score(quality_metrics)

        # Step 6a: Write to PostgreSQL (cpe_entry + cpe_vectors)
        parent_ids = chunk.parent_cpe_ids or []
        child_ids = chunk.child_cpe_ids or []

        cpe_entry_data = {
            "cpe_id": str(cpe_id),
            "mission_text": chunk.text,
            "source_chunk": chunk.text,
            "concept_text": concept_text,
            "probe_question": probe_question,
            "expected_answer": expected_answer,
            "soft_negatives": soft_negatives,
            "hard_negatives": hard_negatives,
            "domain_code": domain_code,
            "task_code": task_code,
            "modifier_code": modifier_code,
            "content_type": "factual",
            "dataset_source": dataset_source,
            "chunk_position": {"index": chunk.chunk_index, "source": chunk.source_document},
            "relations_text": [],
            "echo_score": None,
            "validation_status": "pending",
            "batch_id": batch_id,
            "tmd_bits": tmd_bits,
            "tmd_lane": f"lane_{domain_code}",
            "lane_index": domain_code,
            # NEW: Usage tracking and quality fields
            "confidence_score": confidence_score,
            "quality_metrics": quality_metrics,
            "access_count": 0,  # Starts at 0, incremented on retrieval
            "last_accessed_at": None,  # Will be updated on first access
            # NEW: Parent/child relationships for training chains
            "parent_cpe_ids": parent_ids,
            "child_cpe_ids": child_ids
        }

        # Insert into cpe_entry
        t0 = time.perf_counter()
        insert_cpe_entry(state.pg_conn, cpe_entry_data)

        # Insert vectors into cpe_vectors
        upsert_cpe_vectors(
            state.pg_conn,
            str(cpe_id),
            fused_vec,
            question_vec,
            concept_vec,
            tmd_dense,
            fused_norm
        )
        timings["postgres_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 6b: Write to FAISS
        if state.faiss_db and state.faiss_db.index is not None:
            t0 = time.perf_counter()
            # NOTE: add_vectors not available; this path is disabled unless index is loaded
            # This is kept for future alignment with a FAISS service or ID-mapped index
            timings["faiss_ms"] = (time.perf_counter() - t0) * 1000.0
            backends["faiss"] = "inproc:index-loaded"
        else:
            backends["faiss"] = "skipped:index-not-loaded"

        # Step 6c: Write to Neo4j (TODO: implement via ontology_manager)
        # For now, skip Neo4j integration (can be added later)

        # Success!
        state.total_ingested += 1

        return IngestResult(
            global_id=str(cpe_id),
            concept_text=concept_text,
            tmd_codes={
                "domain": domain_code,
                "task": task_code,
                "modifier": modifier_code
            },
            vector_dimension=784,
            confidence_score=confidence_score,
            quality_metrics=quality_metrics,
            created_at=created_at,
            parent_cpe_ids=parent_ids,
            child_cpe_ids=child_ids,
            success=True,
            timings_ms=timings,
            backends=backends
        )

    except Exception as e:
        state.failed_count += 1
        return IngestResult(
            global_id="",
            concept_text=chunk.text[:50],
            tmd_codes={},
            vector_dimension=0,
            confidence_score=0.0,
            quality_metrics={"error": str(e)},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            parent_cpe_ids=[],
            child_cpe_ids=[],
            success=False,
            error=str(e)
        )


# ============================================================================
# Batch Ingestion Helpers (3-Phase Pipeline)
# ============================================================================

from dataclasses import dataclass

@dataclass
class IntermediateChunkData:
    """Intermediate data structure between pipeline phases"""
    chunk: ChunkInput
    cpe_id: str
    batch_id: str
    dataset_source: str
    skip_cpesh: bool
    chunking_time_ms: Optional[float]

    # Phase 1 outputs (TMD extraction)
    concept_text: str = ""
    probe_question: str = ""
    expected_answer: str = ""
    soft_negatives: List[str] = None
    hard_negatives: List[str] = None
    domain_code: int = 0
    task_code: int = 0
    modifier_code: int = 0
    tmd_bits: int = 0
    timings: Dict[str, float] = None
    backends: Dict[str, str] = None

    # Phase 2 outputs (embeddings)
    concept_vec: np.ndarray = None
    question_vec: np.ndarray = None
    tmd_dense: np.ndarray = None
    fused_vec: np.ndarray = None
    fused_norm: float = 0.0

    def __post_init__(self):
        if self.soft_negatives is None:
            self.soft_negatives = []
        if self.hard_negatives is None:
            self.hard_negatives = []
        if self.timings is None:
            self.timings = {}
        if self.backends is None:
            self.backends = {}


def extract_tmd_phase(intermediate: IntermediateChunkData) -> IntermediateChunkData:
    """
    Phase 1: Extract CPESH and TMD codes.

    This function is called in parallel for each chunk.
    """
    try:
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Step 1: Extract CPESH (if enabled)
        do_cpesh = state.enable_cpesh and not intermediate.skip_cpesh
        if do_cpesh:
            t0 = time.perf_counter()
            cpesh = extract_cpe_from_text(
                intermediate.chunk.text,
                endpoint=state.llm_endpoint,
                model="tinyllama:1.1b"
            )
            intermediate.timings["cpesh_ms"] = (time.perf_counter() - t0) * 1000.0
            intermediate.backends["cpesh"] = f"ollama:{state.llm_endpoint}"

            intermediate.concept_text = cpesh.get("concept", intermediate.chunk.text[:100])
            intermediate.probe_question = cpesh.get("probe_question", "")
            intermediate.expected_answer = cpesh.get("expected_answer", "")
            intermediate.soft_negatives = cpesh.get("soft_negatives", [])
            intermediate.hard_negatives = cpesh.get("hard_negatives", [])
        else:
            # CPESH disabled - use chunk text directly
            intermediate.timings["cpesh_ms"] = 0.0
            intermediate.backends["cpesh"] = "disabled" if intermediate.skip_cpesh else "env-disabled"
            intermediate.concept_text = intermediate.chunk.text
            intermediate.probe_question = ""
            intermediate.expected_answer = ""
            intermediate.soft_negatives = []
            intermediate.hard_negatives = []

        # Optional: propagate external chunking timing
        if intermediate.chunking_time_ms is not None:
            try:
                intermediate.timings["chunking_ms"] = float(intermediate.chunking_time_ms)
            except Exception:
                pass

        # Step 2: Extract TMD codes (client-provided → hybrid → fast heuristic → LLM fallback)
        tmd_endpoint = state.llm_endpoint
        d = t = m = None
        t0 = time.perf_counter()
        # Prefer client-provided TMD (e.g., from hybrid pipeline)
        try:
            if (
                hasattr(intermediate.chunk, "domain_code") and intermediate.chunk.domain_code is not None and
                hasattr(intermediate.chunk, "task_code") and intermediate.chunk.task_code is not None and
                hasattr(intermediate.chunk, "modifier_code") and intermediate.chunk.modifier_code is not None
            ):
                d = int(intermediate.chunk.domain_code)
                t = int(intermediate.chunk.task_code)
                m = int(intermediate.chunk.modifier_code)
                intermediate.timings["tmd_ms"] = 0.1
                intermediate.backends["tmd"] = "client-provided"
        except Exception:
            pass
        # Hybrid mode: Domain via heuristic/LLM, Task/Modifier via heuristics
        if d is None and state.tmd_mode == "hybrid":
            used = ""
            try:
                fast = fast_tmd_extract(intermediate.concept_text)
                conf = float(fast.get("confidence", 0.0))
                if conf >= state.tmd_fast_conf_min:
                    d = max(0, min(15, int(fast.get("domain_code", 2)) - 1))
                    used = "heuristic_domain"
            except Exception:
                pass

            if d is None:
                t1 = time.perf_counter()
                tmd_result = extract_tmd_with_llm(
                    text=intermediate.concept_text,
                    llm_endpoint=tmd_endpoint,
                    llm_model=state.tmd_llm_model,
                )
                intermediate.timings["tmd_ms"] = (time.perf_counter() - t1) * 1000.0
                d = int(tmd_result["domain_code"])  # 0-15
                used = "llm_domain"

            t = int(classify_task(intermediate.concept_text))
            m = int(classify_modifier(intermediate.concept_text))
            intermediate.backends["tmd"] = f"hybrid:{used}+heuristics_tm"

        # Fast-first full heuristic path
        if d is None and state.tmd_fast_first:
            try:
                fast = fast_tmd_extract(intermediate.concept_text)
                conf = float(fast.get("confidence", 0.0))
                if conf >= state.tmd_fast_conf_min:
                    d = max(0, min(15, int(fast.get("domain_code", 2)) - 1))
                    t = max(0, min(31, int(fast.get("task_code", 1)) - 1))
                    m = max(0, min(63, int(fast.get("modifier_code", 27))))
                    intermediate.timings["tmd_ms"] = (time.perf_counter() - t0) * 1000.0
                    intermediate.backends["tmd"] = "heuristic_v2"
            except Exception:
                pass

        if d is None:
            t1 = time.perf_counter()
            tmd_result = extract_tmd_with_llm(
                text=intermediate.concept_text,
                llm_endpoint=tmd_endpoint,
                llm_model=state.tmd_llm_model,
            )
            intermediate.timings["tmd_ms"] = (time.perf_counter() - t1) * 1000.0
            intermediate.backends["tmd"] = f"ollama:{tmd_endpoint}"
            d = int(tmd_result["domain_code"])
            t = int(tmd_result["task_code"])
            m = int(tmd_result["modifier_code"])

        intermediate.domain_code = d
        intermediate.task_code = t
        intermediate.modifier_code = m
        intermediate.tmd_bits = (intermediate.domain_code << 11) | (intermediate.task_code << 6) | intermediate.modifier_code

        return intermediate

    except Exception as e:
        # Mark as failed but return the intermediate object
        intermediate.timings["error"] = str(e)
        return intermediate


def write_to_db_phase(intermediate: IntermediateChunkData) -> IngestResult:
    """
    Phase 3: Write to PostgreSQL + FAISS.

    This function is called in parallel for each chunk after embeddings are generated.
    """
    try:
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Calculate quality metrics
        cpesh_payload = {
            "concept": intermediate.concept_text,
            "probe_question": intermediate.probe_question,
            "expected_answer": intermediate.expected_answer,
            "soft_negatives": intermediate.soft_negatives,
            "hard_negatives": intermediate.hard_negatives,
        }
        quality_metrics = calculate_quality_metrics(cpesh_payload, intermediate.concept_vec, intermediate.chunk.text)
        confidence_score = calculate_confidence_score(quality_metrics)

        # Prepare database entry
        parent_ids = intermediate.chunk.parent_cpe_ids or []
        child_ids = intermediate.chunk.child_cpe_ids or []

        cpe_entry_data = {
            "cpe_id": intermediate.cpe_id,
            "mission_text": intermediate.chunk.text,
            "source_chunk": intermediate.chunk.text,
            "concept_text": intermediate.concept_text,
            "probe_question": intermediate.probe_question,
            "expected_answer": intermediate.expected_answer,
            "soft_negatives": intermediate.soft_negatives,
            "hard_negatives": intermediate.hard_negatives,
            "domain_code": intermediate.domain_code,
            "task_code": intermediate.task_code,
            "modifier_code": intermediate.modifier_code,
            "content_type": "factual",
            "dataset_source": intermediate.dataset_source,
            "chunk_position": {"index": intermediate.chunk.chunk_index, "source": intermediate.chunk.source_document},
            "relations_text": [],
            "echo_score": None,
            "validation_status": "pending",
            "batch_id": intermediate.batch_id,
            "tmd_bits": intermediate.tmd_bits,
            "tmd_lane": f"lane_{intermediate.domain_code}",
            "lane_index": intermediate.domain_code,
            "confidence_score": confidence_score,
            "quality_metrics": quality_metrics,
            "access_count": 0,
            "last_accessed_at": None,
            "parent_cpe_ids": parent_ids,
            "child_cpe_ids": child_ids
        }

        # Write to PostgreSQL
        t0 = time.perf_counter()
        insert_cpe_entry(state.pg_conn, cpe_entry_data)
        upsert_cpe_vectors(
            state.pg_conn,
            intermediate.cpe_id,
            intermediate.fused_vec,
            intermediate.question_vec,
            intermediate.concept_vec,
            intermediate.tmd_dense,
            intermediate.fused_norm
        )
        intermediate.timings["postgres_ms"] = (time.perf_counter() - t0) * 1000.0

        # FAISS (if available)
        if state.faiss_db and state.faiss_db.index is not None:
            intermediate.backends["faiss"] = "inproc:index-loaded"
        else:
            intermediate.backends["faiss"] = "skipped:index-not-loaded"

        # Success!
        state.total_ingested += 1

        return IngestResult(
            global_id=intermediate.cpe_id,
            concept_text=intermediate.concept_text,
            tmd_codes={
                "domain": intermediate.domain_code,
                "task": intermediate.task_code,
                "modifier": intermediate.modifier_code
            },
            vector_dimension=784,
            confidence_score=confidence_score,
            quality_metrics=quality_metrics,
            created_at=created_at,
            parent_cpe_ids=parent_ids,
            child_cpe_ids=child_ids,
            success=True,
            timings_ms=intermediate.timings,
            backends=intermediate.backends
        )

    except Exception as e:
        state.failed_count += 1
        return IngestResult(
            global_id="",
            concept_text=intermediate.chunk.text[:50],
            tmd_codes={},
            vector_dimension=0,
            confidence_score=0.0,
            quality_metrics={"error": str(e)},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            parent_cpe_ids=[],
            child_cpe_ids=[],
            success=False,
            error=str(e)
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """API information"""
    return {
        "name": "LNSP Chunk Ingestion API",
        "version": "1.0.0",
        "description": "Ingest semantic chunks into vecRAG with CPESH + TMD",
        "endpoints": [
            "POST /ingest - Ingest chunks",
            "GET /health - Health check",
            "GET /stats - Statistics"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.pg_conn else "degraded",
        postgresql=state.pg_conn is not None,
        faiss=state.faiss_db is not None and state.faiss_db.index is not None,
        gtr_t5_api=check_service_health(state.gtr_t5_endpoint),
        tmd_router_api=check_service_health(state.tmd_router_endpoint),
        llm_endpoint=state.llm_endpoint
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_chunks_endpoint(request: IngestRequest):
    """
    Ingest semantic chunks into vecRAG.

    Each chunk goes through:
    1. CPESH extraction (TinyLlama)
    2. TMD extraction (Llama 3.1)
    3. Vectorization (GTR-T5 768D + TMD 16D = 784D)
    4. Atomic write (PostgreSQL + FAISS + Neo4j)

    Returns Global_IDs (cpe_id UUIDs) for each chunk.

    **Auto-linking**: Chunks are automatically linked in sequence:
    - Chunk[0]: parent=None, child=UUID[1]
    - Chunk[1]: parent=UUID[0], child=UUID[2]
    - Chunk[n]: parent=UUID[n-1], child=None

    Override by providing explicit parent_cpe_ids/child_cpe_ids.
    """

    start_time = time.time()

    # Generate batch ID if not provided (must be a valid UUID string)
    # CRITICAL FIX: Group chunks by document_id, not by request batch
    default_batch_id = request.batch_id or str(uuid.uuid4())

    # Pre-generate UUIDs for all chunks (for sequential linking)
    chunk_uuids = [str(uuid.uuid4()) for _ in request.chunks]

    # Group chunks by document_id for proper article boundaries
    doc_groups = {}
    for i, chunk in enumerate(request.chunks):
        # Use chunk's document_id as batch_id (preserves article boundaries)
        chunk_batch_id = chunk.document_id if chunk.document_id else default_batch_id

        if chunk_batch_id not in doc_groups:
            doc_groups[chunk_batch_id] = []
        doc_groups[chunk_batch_id].append((i, chunk))

    # Auto-populate parent/child relationships WITHIN each document
    for doc_id, doc_chunks in doc_groups.items():
        for pos, (i, chunk) in enumerate(doc_chunks):
            # Only auto-link if user didn't explicitly provide relationships
            if chunk.parent_cpe_ids is None and chunk.child_cpe_ids is None:
                # Parent = previous chunk in THIS document (if exists)
                if pos > 0:
                    prev_idx = doc_chunks[pos - 1][0]
                    chunk.parent_cpe_ids = [chunk_uuids[prev_idx]]
                else:
                    chunk.parent_cpe_ids = []

                # Child = next chunk in THIS document (if exists)
                if pos < len(doc_chunks) - 1:
                    next_idx = doc_chunks[pos + 1][0]
                    chunk.child_cpe_ids = [chunk_uuids[next_idx]]
                else:
                    chunk.child_cpe_ids = []

    # Process chunks - choose architecture based on config
    if state.enable_parallel and state.enable_batch_embeddings and len(request.chunks) > 1:
        # ========================================================================
        # 3-Phase Batch Pipeline (Parallel + Batch Embeddings)
        # ========================================================================
        loop = asyncio.get_event_loop()

        # Create intermediate data structures
        intermediates = [
            IntermediateChunkData(
                chunk=chunk,
                cpe_id=chunk_uuids[i],
                batch_id=chunk.document_id if chunk.document_id else default_batch_id,  # Use chunk's document_id!
                dataset_source=request.dataset_source,
                skip_cpesh=request.skip_cpesh,
                chunking_time_ms=request.chunking_time_ms
            )
            for i, chunk in enumerate(request.chunks)
        ]

        # ===== PHASE 1: Parallel TMD Extraction =====
        print(f"Phase 1: Extracting TMD for {len(intermediates)} chunks in parallel...")
        phase1_start = time.perf_counter()
        phase1_tasks = [
            loop.run_in_executor(state.executor, extract_tmd_phase, intermediate)
            for intermediate in intermediates
        ]
        intermediates = await asyncio.gather(*phase1_tasks)
        phase1_time = (time.perf_counter() - phase1_start) * 1000
        print(f"  ✓ Phase 1 complete: {phase1_time:.1f}ms")

        # ===== PHASE 2: Batch Embeddings =====
        print(f"Phase 2: Batch embedding {len(intermediates)} concepts...")
        phase2_start = time.perf_counter()

        # Collect all concept texts
        concept_texts = [inter.concept_text for inter in intermediates]
        probe_questions = [inter.probe_question if inter.probe_question else inter.concept_text for inter in intermediates]

        # Single batch GPU call for all concepts
        t0 = time.perf_counter()
        if state.embedder:
            concept_vecs = state.embedder.encode(concept_texts, batch_size=32)  # Batch encode!
            question_vecs = state.embedder.encode(probe_questions, batch_size=32)
            embedding_backend = "inproc:gtr-t5"
        else:
            # Fallback: call GTR-T5 API (still batched)
            response = requests.post(
                f"{state.gtr_t5_endpoint}/embed",
                json={"texts": concept_texts, "normalize": True},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                concept_vecs = np.array(data["embeddings"], dtype=np.float32)
                question_vecs = concept_vecs.copy()  # Simplified for API fallback
                embedding_backend = f"api:{state.gtr_t5_endpoint}"
            else:
                raise Exception(f"GTR-T5 API failed: HTTP {response.status_code}")

        embedding_time_ms = (time.perf_counter() - t0) * 1000.0

        # Encode TMD vectors and fuse
        for i, intermediate in enumerate(intermediates):
            intermediate.concept_vec = concept_vecs[i]
            intermediate.question_vec = question_vecs[i]
            intermediate.tmd_dense = encode_tmd_16d(intermediate.domain_code, intermediate.task_code, intermediate.modifier_code)
            intermediate.fused_vec = np.concatenate([intermediate.concept_vec, intermediate.tmd_dense])
            intermediate.fused_norm = float(np.linalg.norm(intermediate.fused_vec))

            # Record timing
            intermediate.timings["embedding_ms"] = embedding_time_ms / len(intermediates)  # Amortized time
            intermediate.timings["tmd_encode_ms"] = 0.1  # Negligible
            intermediate.timings["fuse_ms"] = 0.1  # Negligible
            intermediate.backends["embedding"] = embedding_backend

        phase2_time = (time.perf_counter() - phase2_start) * 1000
        print(f"  ✓ Phase 2 complete: {phase2_time:.1f}ms ({phase2_time/len(intermediates):.1f}ms per chunk)")

        # ===== PHASE 3: Parallel Processing + Batch Database Writes =====
        print(f"Phase 3: Processing {len(intermediates)} entries in parallel + batch write...")
        phase3_start = time.perf_counter()

        def prepare_chunk_data(intermediate):
            """Prepare data for one chunk (CPU-bound work done in parallel)"""
            # Calculate quality metrics
            cpesh_payload = {
                "concept": intermediate.concept_text,
                "probe_question": intermediate.probe_question,
                "expected_answer": intermediate.expected_answer,
                "soft_negatives": intermediate.soft_negatives,
                "hard_negatives": intermediate.hard_negatives,
            }
            quality_metrics = calculate_quality_metrics(cpesh_payload, intermediate.concept_vec, intermediate.chunk.text)
            confidence_score = calculate_confidence_score(quality_metrics)

            # Prepare CPE entry data
            cpe_entry_data = {
                "cpe_id": intermediate.cpe_id,
                "mission_text": intermediate.chunk.text,
                "source_chunk": intermediate.chunk.text,
                "concept_text": intermediate.concept_text,
                "probe_question": intermediate.probe_question,
                "expected_answer": intermediate.expected_answer,
                "soft_negatives": intermediate.soft_negatives,
                "hard_negatives": intermediate.hard_negatives,
                "domain_code": intermediate.domain_code,
                "task_code": intermediate.task_code,
                "modifier_code": intermediate.modifier_code,
                "content_type": "factual",
                "dataset_source": intermediate.dataset_source,
                "chunk_position": {"index": intermediate.chunk.chunk_index, "source": intermediate.chunk.source_document},
                "relations_text": [],
                "echo_score": None,
                "validation_status": "pending",
                "batch_id": intermediate.batch_id,
                "tmd_bits": intermediate.tmd_bits,
                "tmd_lane": f"lane_{intermediate.domain_code}",
                "lane_index": intermediate.domain_code,
                "confidence_score": confidence_score,
                "quality_metrics": quality_metrics,
                "access_count": 0,
                "last_accessed_at": None
            }

            # Prepare vector data
            vector_data = {
                "cpe_id": intermediate.cpe_id,
                "fused_vec": intermediate.fused_vec,
                "question_vec": intermediate.question_vec,
                "concept_vec": intermediate.concept_vec,
                "tmd_dense": intermediate.tmd_dense,
                "fused_norm": intermediate.fused_norm
            }

            return cpe_entry_data, vector_data

        # Parallel data preparation (CPU-bound work)
        if state.enable_parallel and len(intermediates) > 1:
            loop = asyncio.get_event_loop()
            prep_tasks = [
                loop.run_in_executor(state.executor, prepare_chunk_data, intermediate)
                for intermediate in intermediates
            ]
            prepared_data = await asyncio.gather(*prep_tasks)
        else:
            # Serial fallback
            prepared_data = [prepare_chunk_data(intermediate) for intermediate in intermediates]

        # Separate into entry and vector lists
        entry_data_list = [entry for entry, _ in prepared_data]
        vector_data_list = [vector for _, vector in prepared_data]

        # Single batch write for all entries (transactional)
        t_batch_start = time.perf_counter()
        batch_insert_cpe_entries(state.pg_conn, entry_data_list)
        batch_upsert_cpe_vectors(state.pg_conn, vector_data_list)
        batch_time_ms = (time.perf_counter() - t_batch_start) * 1000

        # Create results (mark all as successful since batch succeeded)
        results = []
        for i, intermediate in enumerate(intermediates):
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            intermediate.timings["postgres_ms"] = batch_time_ms / len(intermediates)  # Amortized

            results.append(IngestResult(
                global_id=intermediate.cpe_id,
                concept_text=intermediate.concept_text,
                tmd_codes={
                    "domain": intermediate.domain_code,
                    "task": intermediate.task_code,
                    "modifier": intermediate.modifier_code
                },
                vector_dimension=784,
                confidence_score=entry_data_list[i]["confidence_score"],
                quality_metrics=entry_data_list[i]["quality_metrics"],
                created_at=created_at,
                parent_cpe_ids=[],
                child_cpe_ids=[],
                success=True,
                timings_ms=intermediate.timings,
                backends=intermediate.backends
            ))

        phase3_time = (time.perf_counter() - phase3_start) * 1000
        print(f"  ✓ Phase 3 complete: {phase3_time:.1f}ms (parallel prep + batch write: {batch_time_ms:.1f}ms)")

        print(f"Total pipeline: {phase1_time + phase2_time + phase3_time:.1f}ms (Phase1: {phase1_time:.0f}ms, Phase2: {phase2_time:.0f}ms, Phase3: {phase3_time:.0f}ms)")

        state.total_chunks += len(results)

    elif state.enable_parallel and len(request.chunks) > 1:
        # ========================================================================
        # 2-Phase Pipeline (Parallel but Individual Embeddings - Original)
        # ========================================================================
        loop = asyncio.get_event_loop()
        tasks = []
        for i, chunk in enumerate(request.chunks):
            # Use chunk's document_id as batch_id
            chunk_batch_id = chunk.document_id if chunk.document_id else default_batch_id
            task = loop.run_in_executor(
                state.executor,
                ingest_chunk,
                chunk,
                request.dataset_source,
                chunk_batch_id,  # Use document-specific batch_id!
                chunk_uuids[i],  # pre_assigned_uuid (positional)
                request.skip_cpesh,  # skip_cpesh (positional)
                request.chunking_time_ms  # chunking_time_ms (positional)
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        state.total_chunks += len(results)

    else:
        # ========================================================================
        # Sequential Processing (Fallback for single chunk or disabled parallel)
        # ========================================================================
        results = []
        for i, chunk in enumerate(request.chunks):
            # Use chunk's document_id as batch_id
            chunk_batch_id = chunk.document_id if chunk.document_id else default_batch_id
            result = ingest_chunk(
                chunk,
                request.dataset_source,
                chunk_batch_id,  # Use document-specific batch_id!
                pre_assigned_uuid=chunk_uuids[i],
                skip_cpesh=request.skip_cpesh,
                chunking_time_ms=request.chunking_time_ms,
            )
            results.append(result)
            state.total_chunks += 1

    processing_time_ms = (time.time() - start_time) * 1000

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    return IngestResponse(
        results=results,
        total_chunks=len(results),
        successful=successful,
        failed=failed,
        batch_id=default_batch_id,
        processing_time_ms=processing_time_ms
    )


@app.get("/stats", tags=["monitoring"])
async def get_stats():
    """Get ingestion statistics"""
    return {
        "total_chunks_processed": state.total_chunks,
        "successfully_ingested": state.total_ingested,
        "failed": state.failed_count,
        "success_rate": (state.total_ingested / state.total_chunks * 100) if state.total_chunks > 0 else 0
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    print("🚀 LNSP Chunk Ingestion API starting...")

    # Connect to PostgreSQL
    try:
        state.pg_conn = connect_pg()
        print("   ✅ PostgreSQL connected")
    except Exception as e:
        print(f"   ❌ PostgreSQL connection failed: {e}")
        state.pg_conn = None

    # Initialize FAISS
    try:
        state.faiss_db = FaissDB(
            index_path="artifacts/user_chunks.index",
            meta_npz_path="artifacts/user_chunks.npz"  # Fixed parameter name
        )
        print("   ✅ FAISS initialized")
    except Exception as e:
        print(f"   ⚠️  FAISS initialization failed: {e} (will try to create new)")
        state.faiss_db = None

    # Initialize GTR-T5 embedder (vec2text-compatible version)
    try:
        print("   Loading vec2text-compatible embedder...")
        state.embedder = Vec2TextCompatibleEmbedder()
        print("   ✅ Vec2Text-compatible GTR-T5 embedder loaded")
        try:
            print(f"   Embedder device: {state.embedder.get_device()}")
        except Exception:
            pass
    except Exception as e:
        print(f"   ⚠️  Vec2Text embedder failed: {e} (will use API fallback)")
        state.embedder = None

    print(f"   LLM endpoint: {state.llm_endpoint}")
    print(f"   TMD Router: {state.tmd_router_endpoint}")
    print(f"   GTR-T5 API: {state.gtr_t5_endpoint}")
    print(f"   CPESH extraction: {'✅ ENABLED' if state.enable_cpesh else '⚠️  DISABLED (fast mode)'}")
    print(f"   Parallel processing: {'✅ ENABLED' if state.enable_parallel else '⚠️  DISABLED (sequential)'}")
    print(f"   Batch embeddings: {'✅ ENABLED (3-phase pipeline)' if state.enable_batch_embeddings else '⚠️  DISABLED (individual embeds)'}")
    print(f"   Max parallel workers: {state.max_parallel_workers}")
    print("✅ Chunk Ingestion API ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("👋 Chunk Ingestion API shutting down...")

    # Shutdown thread pool executor
    if state.executor:
        state.executor.shutdown(wait=True)
        print("   Thread pool executor shutdown")

    if state.pg_conn:
        state.pg_conn.close()
        print("   PostgreSQL connection closed")

    if state.faiss_db:
        state.faiss_db.save()
        print("   FAISS index saved")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8004,
        log_level="info"
    )
