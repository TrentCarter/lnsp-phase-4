#!/usr/bin/env python3
"""
Chunking API: FastAPI endpoint for semantic text chunking.

Provides RESTful API for chunking text into concept-based segments
for TMD-LS pipeline integration.

Endpoints:
    POST /chunk - Chunk text using specified mode
    GET /health - Health check
    GET /stats - Service statistics

Usage:
    uvicorn app.api.chunking:app --reload --port 8001

Example:
    curl -X POST http://localhost:8001/chunk \
      -H "Content-Type: application/json" \
      -d '{
        "text": "Your text here...",
        "mode": "semantic",
        "max_chunk_size": 320
      }'
"""

import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import chunkers
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.semantic_chunker import (
    SemanticChunker,
    PropositionChunker,
    HybridChunker,
    Chunk,
    analyze_chunks
)

# Also import simple chunking from chunker_v2
from src.chunker_v2 import create_chunks as simple_create_chunks

# Import LlamaIndex components for lightweight chunking
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChunkingMode(str, Enum):
    """Chunking strategy modes."""
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    PROPOSITION = "proposition"
    HYBRID = "hybrid"


class ChunkRequest(BaseModel):
    """Request model for /chunk endpoint."""
    text: str = Field(..., description="Input text to chunk", min_length=10)
    mode: ChunkingMode = Field(
        default=ChunkingMode.SEMANTIC,
        description="Chunking mode (semantic/proposition/hybrid)"
    )
    max_chunk_size: int = Field(
        default=320,
        description="Maximum words per chunk (semantic mode only)",
        ge=50,
        le=1000
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum characters per chunk",
        ge=10,
        le=2000
    )
    breakpoint_threshold: int = Field(
        default=95,
        description="Semantic boundary sensitivity (50-99). Lower=more chunks, Higher=fewer chunks",
        ge=50,
        le=99
    )
    llm_model: str = Field(
        default="tinyllama:1.1b",
        description="LLM model for proposition/hybrid modes (e.g., tinyllama:1.1b, llama3.1:8b)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to attach to chunks"
    )
    force_refine: bool = Field(
        default=False,
        description="Force proposition refinement (hybrid mode only)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Photosynthesis is the process by which plants convert light energy into chemical energy. This occurs in the chloroplasts of plant cells.",
                "mode": "semantic",
                "max_chunk_size": 320,
                "min_chunk_size": 500,
                "metadata": {"document_id": "doc_123", "source": "biology_textbook"}
            }
        }


class ChunkResponse(BaseModel):
    """Response model for /chunk endpoint."""
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunk objects")
    total_chunks: int = Field(..., description="Number of chunks created")
    chunking_mode: str = Field(..., description="Chunking mode used")
    statistics: Dict[str, Any] = Field(..., description="Chunk statistics")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "text": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                        "chunk_id": "a1b2c3d4e5f6g7h8",
                        "chunk_index": 0,
                        "word_count": 13,
                        "char_count": 86,
                        "chunking_mode": "semantic",
                        "metadata": {"document_id": "doc_123"}
                    }
                ],
                "total_chunks": 3,
                "chunking_mode": "semantic",
                "statistics": {
                    "mean_words": 45.2,
                    "min_words": 13,
                    "max_words": 87,
                    "chunking_modes": {"semantic": 3}
                },
                "processing_time_ms": 124.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    chunkers_loaded: Dict[str, bool] = Field(..., description="Chunker availability")
    version: str = Field(..., description="API version")


class StatsResponse(BaseModel):
    """Service statistics response."""
    total_requests: int = Field(..., description="Total chunking requests processed")
    total_chunks_created: int = Field(..., description="Total chunks created")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    chunking_mode_usage: Dict[str, int] = Field(..., description="Usage by mode")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="LNSP Chunking API",
    description="Semantic text chunking service for TMD-LS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
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
    """Global service state and cached models."""
    def __init__(self):
        # Cached models (loaded once at startup for speed)
        self.cached_embed_model = None  # HuggingFaceEmbedding (for semantic/hybrid)

        # Statistics
        self.total_requests = 0
        self.total_chunks_created = 0
        self.total_processing_time_ms = 0.0
        self.mode_usage = {
            ChunkingMode.SIMPLE: 0,
            ChunkingMode.SEMANTIC: 0,
            ChunkingMode.PROPOSITION: 0,
            ChunkingMode.HYBRID: 0
        }

        # Configuration from environment
        self.embed_model = os.getenv("LNSP_EMBEDDER_PATH", "sentence-transformers/gtr-t5-base")
        self.llm_endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
        self.llm_model = os.getenv("LNSP_LLM_MODEL", "tinyllama:1.1b")  # Default LLM


state = ServiceState()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize chunkers on startup."""
    logger.info("Starting Chunking API...")

    try:
        # Load embedding model ONCE (cached for all requests)
        logger.info(f"Loading embedding model {state.embed_model} (this takes ~2 seconds)...")
        state.cached_embed_model = HuggingFaceEmbedding(model_name=state.embed_model)
        logger.info("✓ Embedding model cached (used by semantic + hybrid modes)")

        logger.info("✅ Chunking API started")
        logger.info(f"   LLM endpoint: {state.llm_endpoint}")
        logger.info(f"   Default LLM: {state.llm_model} (switchable via GUI)")

    except Exception as e:
        logger.error(f"Failed to initialize chunkers: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Chunking API...")
    logger.info(f"Total requests processed: {state.total_requests}")
    logger.info(f"Total chunks created: {state.total_chunks_created}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """API root endpoint."""
    return {
        "service": "LNSP Chunking API",
        "version": "1.0.0",
        "endpoints": {
            "chunk": "POST /chunk",
            "health": "GET /health",
            "stats": "GET /stats",
            "docs": "GET /docs",
            "web_ui": "GET /web"
        }
    }


@app.get("/web", tags=["info"])
async def web_ui():
    """Serve the web UI for testing chunking."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "chunk_tester.html")

    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI not found. Make sure app/api/static/chunk_tester.html exists."
        )


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        chunkers_loaded={
            "semantic": state.cached_embed_model is not None,
            "proposition": True,  # Created dynamically per request
            "hybrid": state.cached_embed_model is not None
        },
        version="1.0.0"
    )


@app.get("/stats", response_model=StatsResponse, tags=["monitoring"])
async def get_stats():
    """Get service statistics."""
    avg_time = (
        state.total_processing_time_ms / state.total_requests
        if state.total_requests > 0
        else 0.0
    )

    return StatsResponse(
        total_requests=state.total_requests,
        total_chunks_created=state.total_chunks_created,
        average_processing_time_ms=round(avg_time, 2),
        chunking_mode_usage={
            mode.value: count
            for mode, count in state.mode_usage.items()
        }
    )


@app.post("/chunk", response_model=ChunkResponse, tags=["chunking"])
async def chunk_text(request: ChunkRequest):
    """
    Chunk text into semantic segments.

    Supports three chunking modes:
    - **semantic**: Fast embedding-based semantic boundary detection
    - **proposition**: High-quality LLM-extracted atomic propositions
    - **hybrid**: Semantic splitting + selective proposition refinement

    Returns chunks with metadata for TMD-LS pipeline integration.
    """
    start_time = time.time()

    try:
        # Create chunker with requested parameters
        # (We create fresh instances to allow custom settings per request)
        if request.mode == ChunkingMode.SIMPLE:
            # Simple mode uses word-count based chunking
            min_words = max(10, request.min_chunk_size // 5)  # Convert chars to approx words
            chunks_raw = simple_create_chunks(
                text=request.text,
                min_words=min_words,
                max_words=request.max_chunk_size
            )
            # Convert dict chunks to Chunk objects
            from dataclasses import dataclass
            chunks = []
            for c in chunks_raw:
                chunk_obj = type('Chunk', (), {
                    'text': c['text'],
                    'chunk_id': c['chunk_id'],
                    'chunk_index': c['chunk_index'],
                    'word_count': c['word_count'],
                    'char_count': len(c['text']),
                    'chunking_mode': c.get('chunking_mode', 'simple'),
                    'metadata': request.metadata or {},
                    'to_dict': lambda self: {
                        'text': self.text,
                        'chunk_id': self.chunk_id,
                        'chunk_index': self.chunk_index,
                        'word_count': self.word_count,
                        'char_count': self.char_count,
                        'chunking_mode': self.chunking_mode,
                        'metadata': self.metadata
                    }
                })()
                chunks.append(chunk_obj)
            chunker = None  # No chunker object for simple mode

        elif request.mode == ChunkingMode.SEMANTIC:
            # Use cached embedding model (FAST - no reload)
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=request.breakpoint_threshold,
                embed_model=state.cached_embed_model
            )

            # Create document and split
            doc = Document(text=request.text, metadata=request.metadata or {})
            nodes = splitter.get_nodes_from_documents([doc])

            # Convert to Chunk objects
            chunks = []
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()

                # Filter tiny chunks
                if len(chunk_text) < request.min_chunk_size:
                    continue

                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=hashlib.md5(f"{chunk_text}{idx}".encode()).hexdigest()[:16],
                    chunk_index=idx,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    chunking_mode="semantic",
                    metadata={
                        **(request.metadata or {}),
                        "embedding_model": state.embed_model,
                        "buffer_size": 1,
                        "breakpoint_threshold": request.breakpoint_threshold
                    }
                )
                chunks.append(chunk)

            chunker = None  # No chunker object needed

        elif request.mode == ChunkingMode.PROPOSITION:
            # Determine endpoint based on model (different models run on different ports)
            model_endpoints = {
                "tinyllama:1.1b": "http://localhost:11435",
                "llama3.1:8b": "http://localhost:11434",
                "phi3:mini": "http://localhost:11436",
                "granite3-moe:1b": "http://localhost:11437"
            }
            llm_endpoint = model_endpoints.get(request.llm_model, state.llm_endpoint)

            # Create PropositionChunker with requested LLM model
            proposition_chunker = PropositionChunker(
                llm_endpoint=llm_endpoint,
                llm_model=request.llm_model
            )
            chunks = proposition_chunker.chunk(
                text=request.text,
                metadata=request.metadata
            )
            chunker = None  # No chunker object needed

        elif request.mode == ChunkingMode.HYBRID:
            # Fast hybrid: use cached model for semantic, then refine with requested LLM
            # Stage 1: Semantic splitting (FAST - uses cached model)
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=request.breakpoint_threshold,
                embed_model=state.cached_embed_model
            )

            doc = Document(text=request.text, metadata=request.metadata or {})
            nodes = splitter.get_nodes_from_documents([doc])

            # Convert to semantic chunks
            semantic_chunks = []
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()
                if len(chunk_text) < request.min_chunk_size:
                    continue

                semantic_chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=hashlib.md5(f"{chunk_text}{idx}".encode()).hexdigest()[:16],
                    chunk_index=idx,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    chunking_mode="semantic",
                    metadata={**(request.metadata or {})}
                ))

            # Stage 2: Selective proposition refinement (for large chunks)
            # Determine endpoint based on model
            model_endpoints = {
                "tinyllama:1.1b": "http://localhost:11435",
                "llama3.1:8b": "http://localhost:11434",
                "phi3:mini": "http://localhost:11436",
                "granite3-moe:1b": "http://localhost:11437"
            }
            llm_endpoint = model_endpoints.get(request.llm_model, state.llm_endpoint)

            # Create PropositionChunker with requested LLM model
            proposition_chunker = PropositionChunker(
                llm_endpoint=llm_endpoint,
                llm_model=request.llm_model
            )

            chunks = []
            refine_threshold = 150  # Words
            for chunk in semantic_chunks:
                should_refine = (
                    request.force_refine or
                    chunk.word_count > refine_threshold
                )

                if should_refine:
                    # Extract propositions using requested LLM model
                    propositions = proposition_chunker.chunk(chunk.text, chunk.metadata)
                    if propositions:
                        for prop in propositions:
                            prop.metadata["refined_from"] = chunk.chunk_id
                            prop.metadata["llm_model"] = request.llm_model
                            prop.chunking_mode = "hybrid"
                        chunks.extend(propositions)
                    else:
                        chunk.chunking_mode = "hybrid"
                        chunks.append(chunk)
                else:
                    chunk.chunking_mode = "hybrid"
                    chunks.append(chunk)

            chunker = None  # No chunker object needed

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chunking mode: {request.mode}"
            )

        # All chunking is now done inline above (no need for this block)

        # Calculate statistics
        stats = analyze_chunks(chunks)

        # Update service state
        state.total_requests += 1
        state.total_chunks_created += len(chunks)
        state.mode_usage[request.mode] += 1

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        state.total_processing_time_ms += processing_time_ms

        # Build response
        return ChunkResponse(
            chunks=[chunk.to_dict() for chunk in chunks],
            total_chunks=len(chunks),
            chunking_mode=request.mode.value,
            statistics=stats,
            processing_time_ms=round(processing_time_ms, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunking failed: {str(e)}"
        )


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("CHUNKING_API_PORT", "8001"))
    host = os.getenv("CHUNKING_API_HOST", "127.0.0.1")

    logger.info(f"Starting Chunking API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
