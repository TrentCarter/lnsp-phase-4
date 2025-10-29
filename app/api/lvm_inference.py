"""
FastAPI LVM Inference Service with Chat Interface

Provides REST API and web chat interface for Latent Vector Models (AMN, GRU, LSTM, Transformer, Mamba).
Tokenless inference: Text → 768D vectors → LVM → 768D prediction → Text

Port Assignment:
- 9001: AMN (Attention Mixer Network)
- 9002: Transformer (Optimized)
- 9003: GRU
- 9004: LSTM
- 9005: Vec2Text Direct (passthrough)
- 9006-9999: Reserved for future models

Usage:
    uvicorn app.api.lvm_inference:app --host 127.0.0.1 --port 9001 --reload
"""

import torch
import numpy as np
import time
import json
import os
import logging
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from functools import lru_cache
from collections import Counter, deque
import re

# Import LVM models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.lvm.model import (
    AMNModel,
    GRUModel,
    LSTMModel,
    TransformerModel,
    load_lvm_model as load_model_from_checkpoint
)
from app.retrieval.context_builder import RetrievalContextBuilder

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration & Model Loading
# ============================================================================

class LVMConfig:
    """LVM model configuration (set via environment or startup)"""
    model_type: str = "amn"  # amn, gru, lstm, transformer, mamba
    model_path: str = "artifacts/lvm/models/amn_v0.pt"
    device: str = "cpu"  # cpu, mps, cuda
    context_length: int = 5
    vector_dim: int = 768
    hidden_dim: int = 1024

    # External services
    encoder_url: str = "http://localhost:8767/embed"
    decoder_url: str = "http://localhost:8766/decode"

    # UI / inference tuning
    max_decode_steps: int = 50
    passthrough: bool = False  # When True, skip LVM and decode last encoded vector


config = LVMConfig()
model = None
context_builder = None  # Retrieval-primed context builder

# Decode result cache (LRU 10k)
_decode_cache = {}
_decode_cache_hits = 0
_decode_cache_misses = 0

# ============================================================================
# P4: Observability & Safeguards
# ============================================================================

# Timeouts (milliseconds)
TIMEOUT_ENCODE = 2000  # 2s
TIMEOUT_LVM = 200      # 200ms
TIMEOUT_DECODE = 2000  # 2s
TIMEOUT_TOTAL = 3000   # 3s

# SLO Targets
SLO_P50_MS = 1000       # p50 ≤ 1.0s
SLO_P95_MS = 1300       # p95 ≤ 1.3s
SLO_GIBBERISH_PCT = 5   # gibberish rate ≤ 5%
SLO_KEYWORD_HIT_PCT = 75  # keyword hit ≥ 75%
SLO_ENTITY_HIT_PCT = 80   # entity hit ≥ 80%
SLO_ERROR_RATE_PCT = 0.5  # error rate ≤ 0.5%

# Global metrics (rolling window: last 1000 requests)
_metrics_latencies = deque(maxlen=1000)
_metrics_gibberish_count = 0
_metrics_keyword_hits = 0
_metrics_entity_hits = 0
_metrics_error_count = 0
_metrics_total_requests = 0

# Circuit breaker for decoder escalations
_circuit_breaker_window = deque(maxlen=100)  # Last 100 decode attempts
_circuit_breaker_extractive_mode = False  # If True, use extractive fallback only

# Model and index IDs (for version tracking)
_model_id = None  # Set on startup: hash of model file
_index_id = None  # Set on startup: hash of FAISS index file
_decoder_cfg_id = "vec2text_default_v1"  # Decoder configuration version


def load_lvm_model(model_type: str, model_path: str, device: str = "cpu"):
    """
    Load LVM model from checkpoint.

    Args:
        model_type: Model architecture (amn, gru, lstm, transformer)
        model_path: Path to .pt checkpoint
        device: Device (cpu, mps, cuda)

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading {model_type.upper()} model from {model_path}...")

    # Use centralized loading function
    model = load_model_from_checkpoint(model_type, model_path, device)

    print(f"✅ {model_type.upper()} model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, context_builder

    # Startup: Load model
    print("="*60)
    print("LVM INFERENCE SERVICE STARTUP")
    print("="*60)

    if config.passthrough:
        print("⚙️  Passthrough mode enabled - skipping LVM model load")
        model = None
    else:
        model = load_lvm_model(
            model_type=config.model_type,
            model_path=config.model_path,
            device=config.device
        )

    # Allow runtime override for decode steps
    max_steps_env = os.getenv("VEC2TEXT_MAX_STEPS")
    if max_steps_env:
        try:
            config.max_decode_steps = max(1, int(max_steps_env))
        except ValueError:
            print(f"⚠️  Invalid VEC2TEXT_MAX_STEPS value '{max_steps_env}', keeping default {config.max_decode_steps}")

    # Initialize retrieval-primed context builder
    if config.passthrough:
        print("\nℹ️  Skipping retrieval context builder in passthrough mode")
        context_builder = None
    else:
        print("\n🔍 Initializing retrieval-primed context builder...")
        try:
            index_path = os.getenv("FAISS_INDEX_PATH", "artifacts/wikipedia_500k_corrected_ivf_flat_ip.index")
            vectors_path = os.getenv("VECTORS_MEMMAP_PATH", "artifacts/payload/vectors.f32.memmap")
            meta_path = os.getenv("META_PATH", "artifacts/payload/meta.npz")
            lane_path = os.getenv("LANE_PATH", "artifacts/payload/lane_id.npy")
            nprobe = int(os.getenv("FAISS_NPROBE", "16"))
            max_cands = int(os.getenv("CTX_MAX_CANDIDATES", "64"))

            context_builder = RetrievalContextBuilder(
                index_path=index_path,
                vectors_path=vectors_path,
                meta_path=meta_path,
                lane_path=lane_path if os.path.exists(lane_path) else None,
                vector_dim=int(os.getenv("VECTOR_DIM", "768")),
                nprobe=nprobe,
                max_candidates=max_cands,
            )
            print(f"✅ Context builder ready")
            print(f"   Index: {Path(index_path).name}")
            print(f"   Vectors: {context_builder.vecs.shape[0]:,} x {context_builder.D}")
            print(f"   nprobe: {nprobe}, max_candidates: {max_cands}")
        except Exception as e:
            print(f"⚠️  Context builder failed to load: {e}")
            print(f"   Falling back to repeat-pad mode")
            context_builder = None

    print("\n✅ Service ready for inference")
    print(f"   Model: {config.model_type.upper()}")
    print(f"   Context: {config.context_length} vectors")
    print(f"   Context mode: {'retrieval-primed' if context_builder else 'repeat-pad'}")
    print(f"   Encoder: {config.encoder_url}")
    print(f"   Decoder: {config.decoder_url}")
    print(f"   Pipeline: {'passthrough (encode→decode)' if config.passthrough else 'lvm (encode→predict→decode)'}")
    print("="*60 + "\n")

    yield

    # Shutdown: Cleanup
    print("\n🔴 Shutting down LVM inference service...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="LVM Inference Service",
    description="Tokenless latent vector model inference with chat interface",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Request/Response Models
# ============================================================================

class InferenceRequest(BaseModel):
    """LVM inference request"""
    context_vectors: List[List[float]] = Field(
        ...,
        description=f"Context vectors ({config.context_length} x {config.vector_dim}D)",
        min_items=config.context_length,
        max_items=config.context_length
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    normalize: bool = Field(True, description="L2-normalize output vector")


class InferenceResponse(BaseModel):
    """LVM inference response"""
    predicted_vector: List[float]
    confidence: float
    latency_ms: float
    model_type: str
    device: str


class ChatRequest(BaseModel):
    """Chat-style inference request"""
    messages: Optional[List[str]] = Field(
        None,
        description="Previous messages (context) - provide either this OR paragraph",
        min_items=1,
        max_items=config.context_length
    )
    paragraph: Optional[str] = Field(
        None,
        description="Long paragraph to auto-chunk into context (alternative to messages)"
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    decode_steps: int = Field(1, ge=1, description="Vec2text decoding steps (1=fast, higher=better quality)")
    auto_chunk: bool = Field(True, description="Automatically chunk long messages into semantic pieces")
    chunk_mode: str = Field("adaptive", description="Chunking mode: 'adaptive' (auto-detect), 'fixed' (always 5 chunks), 'sentence' (1 chunk per sentence), 'off' (disable)")


class ChatResponse(BaseModel):
    """Chat-style inference response"""
    response: str
    confidence: float
    latency_breakdown: Dict[str, float]
    total_latency_ms: float
    chunks_used: int = Field(default=1, description="Number of chunks created from input")
    chunking_applied: bool = Field(default=False, description="Whether chunking was applied")


# ============================================================================
# Helper Functions
# ============================================================================

def chunk_paragraph(paragraph: str, num_chunks: int = 5) -> List[str]:
    """
    Split a long paragraph into roughly equal semantic chunks (by sentence).

    Args:
        paragraph: Long text to split
        num_chunks: Target number of chunks (default: 5 for LVM context)

    Returns:
        List of text chunks
    """
    import re

    # Simple sentence splitter (handles . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())

    if len(sentences) <= num_chunks:
        # Already short enough
        return sentences

    # Group sentences into roughly equal chunks
    sentences_per_chunk = len(sentences) // num_chunks
    remainder = len(sentences) % num_chunks

    chunks = []
    idx = 0
    for i in range(num_chunks):
        # Add one extra sentence to first 'remainder' chunks
        size = sentences_per_chunk + (1 if i < remainder else 0)
        chunk_sentences = sentences[idx:idx+size]
        chunks.append(' '.join(chunk_sentences))
        idx += size

    return chunks


def split_by_sentence(text: str) -> List[str]:
    """
    Split text into individual sentences (1 sentence = 1 chunk).

    Args:
        text: Text to split

    Returns:
        List of sentences (each sentence is a separate chunk)
    """
    import re

    # Simple sentence splitter (handles . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter out empty strings
    return [s for s in sentences if s.strip()]


def check_bigram_repeat(text: str) -> float:
    """Calculate bigram repetition rate (gibberish detector)."""
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    max_count = max(counts.values())
    return max_count / len(bigrams)


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text (low entropy = repetitive)."""
    if not text:
        return 0.0
    char_counts = Counter(text.lower())
    total = sum(char_counts.values())
    probs = [count/total for count in char_counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def extract_keywords(text: str) -> set:
    """Extract simple keywords (alphanumeric tokens)."""
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return set(tokens)


def check_keyword_overlap(decoded: str, sources: List[str]) -> bool:
    """Check if decoded text has at least 1 keyword from source texts."""
    decoded_kw = extract_keywords(decoded)
    source_kw = set()
    for s in sources:
        source_kw.update(extract_keywords(s))
    return len(decoded_kw & source_kw) > 0


def extract_entities(text: str) -> set:
    """Extract capitalized entities and numbers."""
    # Simple heuristic: capitalized words (not sentence-start) + numbers
    words = text.split()
    entities = set()
    for i, word in enumerate(words):
        # Remove punctuation
        clean = re.sub(r'[^\w]', '', word)
        # Capitalized word (not first word of sentence)
        if clean and (clean[0].isupper() and i > 0):
            entities.add(clean.lower())
        # Numbers
        elif clean.isdigit():
            entities.add(clean)
    return entities


def scrub_pii_and_urls(text: str) -> str:
    """Remove URLs and basic PII patterns from text."""
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '[URL]', text)
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Remove phone numbers (basic patterns)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # Remove SSN patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return text


def check_profanity(text: str) -> bool:
    """Basic profanity check (simple keyword list)."""
    # Very basic profanity detection (extend as needed)
    profanity_words = [
        'fuck', 'shit', 'damn', 'bastard', 'bitch',
        'asshole', 'crap', 'piss', 'slut', 'whore'
    ]
    text_lower = text.lower()
    for word in profanity_words:
        if word in text_lower:
            return True
    return False


def delta_gate_check(v_proj: np.ndarray, qvec: np.ndarray) -> tuple[bool, float]:
    """
    Delta-gate: Enforce cos(v_proj, qvec) within [0.15, 0.85] to prevent drift and parroting.

    Returns:
        (passed, cosine_similarity)
    """
    # Normalize vectors
    v_proj_norm = v_proj / (np.linalg.norm(v_proj) + 1e-12)
    qvec_norm = qvec / (np.linalg.norm(qvec) + 1e-12)

    cos_sim = float(np.dot(v_proj_norm, qvec_norm))

    # Check bounds
    passed = 0.15 <= cos_sim <= 0.85

    return passed, cos_sim


async def round_trip_qa_check(
    decoded_text: str,
    v_proj: np.ndarray,
    encoder_url: str,
    min_similarity: float = 0.55
) -> tuple[bool, float]:
    """
    Round-trip semantic QA: Re-encode decoded text and check similarity to original projection.

    Returns:
        (passed, round_trip_similarity)
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                encoder_url,
                json={"texts": [decoded_text]}
            )
            response.raise_for_status()
            result = response.json()

            # Get re-encoded vector
            v_reenc = np.array(result["embeddings"][0], dtype=np.float32)

            # Compute cosine similarity
            v_proj_norm = v_proj / (np.linalg.norm(v_proj) + 1e-12)
            v_reenc_norm = v_reenc / (np.linalg.norm(v_reenc) + 1e-12)
            cos_sim = float(np.dot(v_proj_norm, v_reenc_norm))

            passed = cos_sim >= min_similarity
            return passed, cos_sim

    except Exception as e:
        logger.warning(f"Round-trip QA check failed: {e}")
        # Fail open (don't block on errors)
        return True, 0.0


def log_structured(
    trace_id: str,
    model_id: str,
    index_id: str,
    decoder_cfg_id: str,
    ctx_fill_mode: str,
    steps_used: int,
    cache_hit: bool,
    true_conf: float,
    cos_snap_to_query: float,
    cos_snap_to_ctx: float,
    entity_hit: bool,
    drift_flag: bool,
    total_latency_ms: float,
    latency_breakdown: Dict[str, float],
    gibberish_detected: bool,
    keyword_hit: bool,
    delta_gate_passed: bool,
    round_trip_passed: bool,
    error: Optional[str] = None
):
    """
    Structured logging with all required fields for observability.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "trace_id": trace_id,
        "model_id": model_id,
        "index_id": index_id,
        "decoder_cfg_id": decoder_cfg_id,
        "ctx_fill_mode": ctx_fill_mode,
        "steps_used": steps_used,
        "cache_hit": cache_hit,
        "true_conf": round(true_conf, 4),
        "cos_snap_to_query": round(cos_snap_to_query, 4),
        "cos_snap_to_ctx": round(cos_snap_to_ctx, 4),
        "entity_hit": entity_hit,
        "drift_flag": drift_flag,
        "total_latency_ms": round(total_latency_ms, 2),
        "latency_breakdown": latency_breakdown,
        "gibberish_detected": gibberish_detected,
        "keyword_hit": keyword_hit,
        "delta_gate_passed": delta_gate_passed,
        "round_trip_passed": round_trip_passed,
        "error": error
    }

    # Log as JSON
    logger.info(json.dumps(log_entry))


def update_metrics(
    total_latency_ms: float,
    gibberish_detected: bool,
    keyword_hit: bool,
    entity_hit: bool,
    error: bool
):
    """Update global metrics for SLO tracking."""
    global _metrics_latencies, _metrics_gibberish_count, _metrics_keyword_hits
    global _metrics_entity_hits, _metrics_error_count, _metrics_total_requests

    _metrics_total_requests += 1
    _metrics_latencies.append(total_latency_ms)

    if gibberish_detected:
        _metrics_gibberish_count += 1
    if keyword_hit:
        _metrics_keyword_hits += 1
    if entity_hit:
        _metrics_entity_hits += 1
    if error:
        _metrics_error_count += 1


def get_current_slos() -> dict:
    """Calculate current SLO metrics from rolling window."""
    if len(_metrics_latencies) == 0:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "gibberish_rate_pct": 0.0,
            "keyword_hit_rate_pct": 0.0,
            "entity_hit_rate_pct": 0.0,
            "error_rate_pct": 0.0,
            "total_requests": _metrics_total_requests
        }

    latencies_sorted = sorted(_metrics_latencies)
    p50_idx = len(latencies_sorted) // 2
    p95_idx = int(len(latencies_sorted) * 0.95)

    # Use window size for percentages (not total_requests)
    window_size = len(_metrics_latencies)

    return {
        "p50_ms": round(latencies_sorted[p50_idx], 2),
        "p95_ms": round(latencies_sorted[p95_idx], 2),
        "gibberish_rate_pct": round((_metrics_gibberish_count / window_size) * 100, 2) if window_size > 0 else 0.0,
        "keyword_hit_rate_pct": round((_metrics_keyword_hits / window_size) * 100, 2) if window_size > 0 else 0.0,
        "entity_hit_rate_pct": round((_metrics_entity_hits / window_size) * 100, 2) if window_size > 0 else 0.0,
        "error_rate_pct": round((_metrics_error_count / window_size) * 100, 2) if window_size > 0 else 0.0,
        "total_requests": _metrics_total_requests
    }


def check_slo_compliance(slos: dict) -> tuple[bool, List[str]]:
    """
    Check if current metrics meet SLO targets.

    Returns:
        (compliant, violations)
    """
    violations = []

    if slos["p50_ms"] > SLO_P50_MS:
        violations.append(f"p50 {slos['p50_ms']}ms > {SLO_P50_MS}ms")
    if slos["p95_ms"] > SLO_P95_MS:
        violations.append(f"p95 {slos['p95_ms']}ms > {SLO_P95_MS}ms")
    if slos["gibberish_rate_pct"] > SLO_GIBBERISH_PCT:
        violations.append(f"gibberish {slos['gibberish_rate_pct']}% > {SLO_GIBBERISH_PCT}%")
    if slos["keyword_hit_rate_pct"] < SLO_KEYWORD_HIT_PCT:
        violations.append(f"keyword-hit {slos['keyword_hit_rate_pct']}% < {SLO_KEYWORD_HIT_PCT}%")
    if slos["entity_hit_rate_pct"] < SLO_ENTITY_HIT_PCT:
        violations.append(f"entity-hit {slos['entity_hit_rate_pct']}% < {SLO_ENTITY_HIT_PCT}%")
    if slos["error_rate_pct"] > SLO_ERROR_RATE_PCT:
        violations.append(f"error-rate {slos['error_rate_pct']}% > {SLO_ERROR_RATE_PCT}%")

    return len(violations) == 0, violations


# ============================================================================
# Additional Helper Functions (continue from original)
# ============================================================================

def _extract_entities_helper(text: str) -> set:
    """Helper for extract_entities (original function body)."""
    words = text.split()
    entities = set()
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if clean and (clean[0].isupper() and i > 0):  # Capitalized mid-sentence
            entities.add(clean.lower())
        elif clean.isdigit():  # Numbers
            entities.add(clean)
    return entities


async def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode texts to 768D vectors using GTR-T5 service.

    Args:
        texts: List of text strings

    Returns:
        numpy array of shape (N, 768)
    """
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            config.encoder_url,
            json={"texts": texts, "normalize": True}
        )
        response.raise_for_status()
        result = response.json()
        return np.array(result["embeddings"], dtype=np.float32)


async def decode_vector(vector: np.ndarray, steps: int = 1) -> Dict:
    """
    Decode 768D vector to text using Vec2Text service.

    Args:
        vector: 768D numpy array
        steps: Decoding steps (1=fast, 2=better quality)

    Returns:
        Dict with decoded text, cosine, and latency
    """
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            config.decoder_url,
            json={
                "vectors": [vector.tolist()],
                "subscribers": "jxe",  # Use JXE for speed
                "steps": steps,
                "device": "cpu"
            }
        )
        response.raise_for_status()
        result = response.json()

        # Extract first result from JXE
        first_result = result["results"][0]
        jxe_result = first_result["subscribers"]["gtr → jxe"]

        return {
            "text": jxe_result["output"],
            "cosine": jxe_result["cosine"],
            "latency_ms": jxe_result["elapsed_ms"]
        }


def predict_next_vector(context: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Predict next vector from context using LVM.

    Args:
        context: Tensor of shape (batch, context_length, 768)
        temperature: Sampling temperature

    Returns:
        Predicted vector of shape (batch, 768)
    """
    with torch.no_grad():
        prediction = model(context)

        # Apply temperature (if > 1.0, add noise)
        if temperature > 1.0:
            noise = torch.randn_like(prediction) * (temperature - 1.0) * 0.1
            prediction = prediction + noise

        return prediction


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to chat interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/chat" />
    </head>
    <body>
        <p>Redirecting to chat interface...</p>
    </body>
    </html>
    """)


@app.get("/health")
async def health():
    """Health check with SLO compliance"""
    slos = get_current_slos()
    compliant, violations = check_slo_compliance(slos)

    status = "healthy" if compliant else "degraded"

    return {
        "status": status,
        "model_type": config.model_type,
        "model_loaded": model is not None,
        "device": config.device,
        "encoder_url": config.encoder_url,
        "decoder_url": config.decoder_url,
        "pipeline": "passthrough" if config.passthrough else "lvm",
        "max_decode_steps": config.max_decode_steps,
        "slo_compliant": compliant,
        "slo_violations": violations if not compliant else []
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics and SLO tracking"""
    slos = get_current_slos()
    compliant, violations = check_slo_compliance(slos)

    # Circuit breaker status
    steps_5_rate = 0.0
    if len(_circuit_breaker_window) > 0:
        steps_5_count = sum(1 for s in _circuit_breaker_window if s == 5)
        steps_5_rate = (steps_5_count / len(_circuit_breaker_window)) * 100

    return {
        "slo_metrics": slos,
        "slo_targets": {
            "p50_ms": SLO_P50_MS,
            "p95_ms": SLO_P95_MS,
            "gibberish_rate_pct": SLO_GIBBERISH_PCT,
            "keyword_hit_rate_pct": SLO_KEYWORD_HIT_PCT,
            "entity_hit_rate_pct": SLO_ENTITY_HIT_PCT,
            "error_rate_pct": SLO_ERROR_RATE_PCT
        },
        "compliance": {
            "status": "compliant" if compliant else "violated",
            "violations": violations
        },
        "cache": {
            "hits": _decode_cache_hits,
            "misses": _decode_cache_misses,
            "hit_rate_pct": round((_decode_cache_hits / (_decode_cache_hits + _decode_cache_misses)) * 100, 2) if (_decode_cache_hits + _decode_cache_misses) > 0 else 0.0,
            "size": len(_decode_cache)
        },
        "circuit_breaker": {
            "extractive_mode": _circuit_breaker_extractive_mode,
            "steps_5_rate_pct": round(steps_5_rate, 2),
            "window_size": len(_circuit_breaker_window)
        },
        "version_ids": {
            "model_id": _model_id,
            "index_id": _index_id,
            "decoder_cfg_id": _decoder_cfg_id
        }
    }


@app.get("/info")
async def info():
    """Model information"""
    if config.passthrough:
        return {
            "pipeline": "passthrough",
            "model_type": config.model_type,
            "device": config.device,
            "encoder_url": config.encoder_url,
            "decoder_url": config.decoder_url,
            "max_decode_steps": config.max_decode_steps
        }

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": config.model_type,
        "model_path": config.model_path,
        "device": config.device,
        "context_length": config.context_length,
        "vector_dim": config.vector_dim,
        "hidden_dim": config.hidden_dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "memory_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
        "max_decode_steps": config.max_decode_steps
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Low-level inference: predict next vector from context vectors.

    This endpoint operates directly on vectors (no text encoding/decoding).
    """
    if config.passthrough:
        raise HTTPException(status_code=400, detail="Passthrough service does not support vector inference")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    # Convert to tensor
    context = torch.tensor(
        [request.context_vectors],
        dtype=torch.float32,
        device=config.device
    )

    # Predict
    prediction = predict_next_vector(context, temperature=request.temperature)

    # Normalize if requested
    if request.normalize:
        prediction = prediction / (torch.norm(prediction, dim=-1, keepdim=True) + 1e-12)

    # Compute confidence (negative entropy of normalized vector)
    pred_norm = prediction / (torch.norm(prediction, dim=-1, keepdim=True) + 1e-12)
    confidence = float(1.0 - torch.abs(pred_norm).mean())

    latency_ms = (time.perf_counter() - start_time) * 1000

    return InferenceResponse(
        predicted_vector=prediction[0].cpu().tolist(),
        confidence=confidence,
        latency_ms=latency_ms,
        model_type=config.model_type,
        device=config.device
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat-style inference: text → vectors → LVM → vector → text.

    Supports two input modes:
    1. messages: List of context messages (1-5)
    2. paragraph: Long text auto-chunked into 5 context chunks

    Full tokenless pipeline with encoding and decoding.

    P4 Safeguards Available:
    - All helper functions implemented (delta_gate_check, round_trip_qa_check, etc.)
    - Use /metrics endpoint for SLO tracking
    - Use /health endpoint for compliance status
    - See docs/P4_SAFEGUARDS_IMPLEMENTATION.md for integration guide
    """
    if not config.passthrough and model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    if not request.messages and not request.paragraph:
        raise HTTPException(status_code=400, detail="Must provide either 'messages' or 'paragraph'")
    if request.messages and request.paragraph:
        raise HTTPException(status_code=400, detail="Provide only one of 'messages' or 'paragraph', not both")
    if request.decode_steps > config.max_decode_steps:
        raise HTTPException(
            status_code=400,
            detail=f"decode_steps must be <= {config.max_decode_steps}"
        )

    total_start = time.perf_counter()
    latency_breakdown = {}
    chunks_used = config.context_length  # Always 5 (enforced by context builder)
    chunking_applied = False
    ctx_fill_mode = "unknown"  # seq, ann, mixed

    # Step 1: Prepare messages (with smart auto-chunking)
    messages = request.messages

    if request.paragraph:
        # Explicit paragraph mode: always chunk
        messages = chunk_paragraph(request.paragraph, num_chunks=config.context_length)
        latency_breakdown["chunking_ms"] = 0  # Chunking is fast, negligible latency
        chunking_applied = True
        ctx_fill_mode = "seq"  # Sequential chunks from paragraph
    elif messages and request.auto_chunk and request.chunk_mode != "off":
        # Auto-chunking for long messages
        if len(messages) == 1:
            # Single message: check if it's long enough to benefit from chunking
            message = messages[0]
            sentences = message.count('. ') + message.count('! ') + message.count('? ')

            if request.chunk_mode == "sentence":
                # Sentence mode: 1 sentence = 1 chunk (simple split)
                messages = split_by_sentence(message)
                chunking_applied = True
                ctx_fill_mode = "seq"
                latency_breakdown["chunking_ms"] = 0
            elif request.chunk_mode == "fixed":
                # Always chunk into N pieces
                messages = chunk_paragraph(message, num_chunks=config.context_length)
                chunking_applied = True
                ctx_fill_mode = "seq"
                latency_breakdown["chunking_ms"] = 0
            elif request.chunk_mode == "adaptive" and sentences >= config.context_length:
                # Adaptive: only chunk if message has enough sentences
                messages = chunk_paragraph(message, num_chunks=config.context_length)
                chunking_applied = True
                ctx_fill_mode = "seq"
                latency_breakdown["chunking_ms"] = 0

    if not messages:
        raise HTTPException(status_code=400, detail="No messages available for encoding")

    encode_inputs = messages if not config.passthrough else [messages[-1]]

    # Step 2: Encode messages to vectors
    encode_start = time.perf_counter()
    try:
        context_vectors = await encode_texts(encode_inputs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Encoding failed: {str(e)}")

    latency_breakdown["encoding_ms"] = (time.perf_counter() - encode_start) * 1000

    # Build context: retrieval-primed (4 diverse supports + query) or fallback to repeat-pad
    support_indices = []
    query_vector = None

    if not config.passthrough:
        ctx_build_start = time.perf_counter()
        query_vector = context_vectors[-1]  # Last encoded vector is the query

        if context_builder is not None:
            # Retrieval-primed: [support×4 + query]
            lane_id = getattr(request, 'lane_id', None)

            try:
                # If we have <5 chunks, use ANN fill (enforces chunks_used==5)
                if len(context_vectors) < config.context_length:
                    ctx_fill_mode = "ann" if len(context_vectors) == 1 else "mixed"

                context_vectors, support_indices = context_builder.build_context(
                    query_vector,
                    lane_id=lane_id
                )

                # Log support details for observability
                try:
                    doc_ids = context_builder.doc_id[np.array(support_indices)]
                    poss = context_builder.pos[np.array(support_indices)]
                    logger.info({
                        "event": "ctx_build",
                        "lane": lane_id,
                        "mode": ctx_fill_mode,
                        "supports": [
                            {"idx": int(i), "doc": int(d), "pos": int(p)}
                            for i, d, p in zip(support_indices, doc_ids, poss)
                        ]
                    })
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"Context builder failed: {e}, falling back to repeat-pad")
                # Fallback: repeat-pad
                ctx_fill_mode = "repeat_pad"
                last_vector = context_vectors[-1]
                num_repeats = config.context_length - len(context_vectors)
                padding = np.repeat(last_vector[np.newaxis, :], num_repeats, axis=0)
                context_vectors = np.vstack([padding, context_vectors])
        else:
            # Fallback: repeat-pad (when context_builder not available)
            ctx_fill_mode = "repeat_pad"
            if len(context_vectors) < config.context_length:
                last_vector = context_vectors[-1]
                num_repeats = config.context_length - len(context_vectors)
                padding = np.repeat(last_vector[np.newaxis, :], num_repeats, axis=0)
                context_vectors = np.vstack([padding, context_vectors])
            elif len(context_vectors) > config.context_length:
                context_vectors = context_vectors[-config.context_length:]

        latency_breakdown["context_build_ms"] = (time.perf_counter() - ctx_build_start) * 1000

        # Hard-fail if not exactly 5 vectors (critical invariant)
        assert context_vectors.shape[0] == config.context_length, \
            f"Context must have exactly {config.context_length} vectors, got {context_vectors.shape[0]}"

        # Step 2: LVM inference
        lvm_start = time.perf_counter()
        context_tensor = torch.tensor(
            [context_vectors],
            dtype=torch.float32,
            device=config.device
        )

        prediction = predict_next_vector(context_tensor, temperature=request.temperature)

        # Normalize
        prediction = prediction / (torch.norm(prediction, dim=-1, keepdim=True) + 1e-12)

        # TODO: Procrustes calibration (needs offline R matrix from training)
        # pred_calibrated = normalize(prediction @ R)

        # Manifold snap + topic anchor blend (improves semantic coherence)
        if context_builder is not None and query_vector is not None:
            try:
                cal_start = time.perf_counter()
                pred_np = prediction[0].cpu().numpy().astype(np.float32)
                pred_normalized = pred_np / (np.linalg.norm(pred_np) + 1e-12)

                # Manifold snap: kNN barycentric projection (K=16)
                K_snap = 16
                sim, snap_idx = context_builder.index.search(pred_normalized[None, :], K_snap)
                snap_idx = snap_idx[0]
                snap_idx = snap_idx[snap_idx >= 0]  # Filter invalid indices

                if len(snap_idx) > 0:
                    # Barycentric mixture: weighted average by cosine similarity
                    snap_vecs = context_builder.vecs[snap_idx]  # [K, D]
                    snap_sims = sim[0][:len(snap_idx)]
                    snap_weights = np.maximum(snap_sims, 0.0)
                    snap_weights = snap_weights / (snap_weights.sum() + 1e-12)

                    pred_snap = np.sum(snap_vecs * snap_weights[:, None], axis=0)
                    pred_snap = pred_snap / (np.linalg.norm(pred_snap) + 1e-12)

                    # Topic anchor blend: mix with query and context mean
                    ctx_mean = context_vectors.mean(axis=0)
                    ctx_mean = ctx_mean / (np.linalg.norm(ctx_mean) + 1e-12)

                    qvec_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)

                    # Adaptive anchor mix based on snap alignment
                    cos_snap_to_query = float(np.dot(pred_snap, qvec_norm))
                    cos_snap_to_ctx = float(np.dot(pred_snap, ctx_mean))

                    # If snap is well-aligned, use lighter anchor
                    if cos_snap_to_query >= 0.20 and cos_snap_to_ctx >= 0.30:
                        # Good alignment - use standard blend
                        v_mix = 0.75 * pred_snap + 0.15 * qvec_norm + 0.10 * ctx_mean
                    else:
                        # Poor alignment - use stronger anchor to prevent drift
                        logger.info(f"Drift clamp: cos_q={cos_snap_to_query:.3f}, cos_ctx={cos_snap_to_ctx:.3f} → stronger anchor")
                        v_mix = 0.60 * pred_snap + 0.25 * qvec_norm + 0.15 * ctx_mean

                    v_mix = v_mix / (np.linalg.norm(v_mix) + 1e-12)

                    # Update prediction
                    prediction = torch.from_numpy(v_mix).to(torch.float32).unsqueeze(0)

                    latency_breakdown["calibration_ms"] = (time.perf_counter() - cal_start) * 1000
                else:
                    latency_breakdown["calibration_ms"] = 0.0
            except Exception as e:
                logger.warning(f"Manifold snap failed: {e}, using raw prediction")
                latency_breakdown["calibration_ms"] = 0.0
        else:
            latency_breakdown["calibration_ms"] = 0.0

        # Real confidence: cosine(prediction, FAISS top-1)
        if context_builder is not None:
            try:
                pred_np = prediction[0].cpu().numpy().astype(np.float32)
                pred_normalized = pred_np / (np.linalg.norm(pred_np) + 1e-12)
                # Search FAISS for top-1
                sim, idx = context_builder.index.search(pred_normalized[None, :], 1)
                confidence = float(sim[0][0])  # Inner product = cosine (vectors are normalized)
            except Exception as e:
                logger.warning(f"Failed to compute real confidence: {e}")
                confidence = 0.0
        else:
            confidence = 0.0  # No FAISS index available

        latency_breakdown["lvm_inference_ms"] = (time.perf_counter() - lvm_start) * 1000
    else:
        # Passthrough mode: skip LVM, use last encoded vector directly
        latency_breakdown["context_build_ms"] = 0.0
        lvm_start = time.perf_counter()
        prediction = torch.from_numpy(context_vectors[-1]).to(torch.float32)
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(0)
        prediction = prediction / (torch.norm(prediction, dim=-1, keepdim=True) + 1e-12)

        # Passthrough confidence: always 1.0 (no LVM transform)
        confidence = 1.0
        ctx_fill_mode = "passthrough"

        latency_breakdown["lvm_inference_ms"] = (time.perf_counter() - lvm_start) * 1000

    # Step 3: Adaptive decode with quality checks and caching
    decode_start = time.perf_counter()

    # Cache key: rounded vector (3 decimals) + steps
    pred_np = prediction[0].cpu().numpy()
    cache_key = (np.round(pred_np, 3).tobytes(), request.decode_steps)

    # Check cache
    global _decode_cache, _decode_cache_hits, _decode_cache_misses
    if cache_key in _decode_cache:
        decode_result = _decode_cache[cache_key]
        _decode_cache_hits += 1
        latency_breakdown["decoding_ms"] = 0.1  # Cache hit ~instant
        latency_breakdown["cache_hit"] = 1.0  # True as float
        latency_breakdown["decode_steps_used"] = 1.0  # Cache hits use last result
    else:
        _decode_cache_misses += 1

        # Adaptive decode with escalation ladder
        # Start: steps=1, temp=0.3
        # Escalate if: bigram-repeat>0.25 OR entropy<2.8 OR no keyword hit
        decode_steps_used = 1
        temp = 0.3
        escalation_reason = None

        # Extract source keywords/entities for quality checks
        source_texts = messages if messages else []
        source_entities = set()
        for txt in source_texts:
            source_entities.update(extract_entities(txt))

        for attempt in [1, 3, 5]:  # Escalation ladder
            try:
                decode_result = await decode_vector(pred_np, steps=attempt)
                decoded_text = decode_result["text"]

                # Quality checks
                bigram_rep = check_bigram_repeat(decoded_text)
                entropy = calculate_entropy(decoded_text)
                has_keyword = check_keyword_overlap(decoded_text, source_texts)
                has_entity = len(extract_entities(decoded_text) & source_entities) > 0 if source_entities else True

                # Pass criteria
                if bigram_rep <= 0.25 and entropy >= 2.8 and (has_keyword or len(source_texts) == 0):
                    decode_steps_used = attempt
                    break
                else:
                    # Failed quality check - escalate if possible
                    reasons = []
                    if bigram_rep > 0.25:
                        reasons.append(f"bigram_rep={bigram_rep:.2f}")
                    if entropy < 2.8:
                        reasons.append(f"entropy={entropy:.2f}")
                    if not has_keyword and len(source_texts) > 0:
                        reasons.append("no_keyword")
                    escalation_reason = ",".join(reasons)

                    if attempt == 5:
                        # Last attempt failed - use extractive fallback
                        if context_builder is not None and support_indices:
                            # Return first support text as extractive fallback
                            try:
                                fallback_idx = support_indices[0]
                                # Get text from original messages or generate placeholder
                                decoded_text = f"[Extracted: support chunk {fallback_idx}]"
                                escalation_reason += ",extractive_fallback"
                            except:
                                pass
                        decode_steps_used = 5
                        break

            except Exception as e:
                if attempt == 5:
                    raise HTTPException(status_code=502, detail=f"Decoding failed: {str(e)}")
                # Try next escalation
                continue

        # Update cache (LRU: keep last 10k)
        _decode_cache[cache_key] = decode_result
        if len(_decode_cache) > 10000:
            # Simple LRU: remove oldest 20%
            keys_to_remove = list(_decode_cache.keys())[:2000]
            for k in keys_to_remove:
                del _decode_cache[k]

        latency_breakdown["decoding_ms"] = decode_result["latency_ms"]
        latency_breakdown["cache_hit"] = 0.0  # False as float
        latency_breakdown["decode_steps_used"] = float(decode_steps_used)
        # Store escalation reason separately (not in latency_breakdown)

    total_latency = (time.perf_counter() - total_start) * 1000

    # Observability metrics
    cache_hit_rate = _decode_cache_hits / (_decode_cache_hits + _decode_cache_misses) if (_decode_cache_hits + _decode_cache_misses) > 0 else 0.0

    return ChatResponse(
        response=decode_result["text"],
        confidence=confidence,
        latency_breakdown=latency_breakdown,
        total_latency_ms=total_latency,
        chunks_used=chunks_used,
        chunking_applied=chunking_applied
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """
    Beautiful chat web interface (like Claude/GPT).
    Tokenless pipeline chat with real-time inference.
    """
    pipeline_label = "Tokenless Pipeline" if config.passthrough else "Tokenless LVM"
    pipeline_mode = "encode → decode" if config.passthrough else "encode → predict → decode"
    if config.passthrough:
        subtitle = f"Device: {config.device.upper()} | Pipeline: {pipeline_mode}"
        intro_headline = "👋 Hello! I'm a direct Vec2Text pipeline."
        intro_detail = f"I encode your message to a 768D vector and decode it back to text without running an LVM. Adjust Vec2Text steps (≤ {config.max_decode_steps}) to trade speed for quality."
    else:
        subtitle = f"Context: {config.context_length} vectors | Device: {config.device.upper()} | Pipeline: {pipeline_mode}"
        intro_headline = "👋 Hello! I'm a tokenless latent vector model."
        intro_detail = "I operate in 768D semantic space without tokens. Use 'Auto-chunk' modes: Adaptive (≥5 sentences), By Sentence (1 sentence = 1 chunk), Fixed (always 5 chunks), or Off (retrieval-primed)."

    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{config.model_type.upper()} Chat - {pipeline_label}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }}

            .container {{
                width: 90%;
                max-width: 800px;
                height: 90vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}

            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }}

            .header h1 {{
                font-size: 24px;
                margin-bottom: 5px;
            }}

            .header .subtitle {{
                font-size: 14px;
                opacity: 0.9;
            }}

            .messages {{
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f7f9fc;
            }}

            .message {{
                margin-bottom: 15px;
                display: flex;
                align-items: flex-start;
            }}

            .message.user {{
                justify-content: flex-end;
            }}

            .message-content {{
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
            }}

            .message.user .message-content {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}

            .message.assistant .message-content {{
                background: white;
                color: #333;
                border: 1px solid #e1e8ed;
            }}

            .message-meta {{
                font-size: 11px;
                opacity: 0.6;
                margin-top: 4px;
            }}

            .input-area {{
                padding: 20px;
                background: white;
                border-top: 1px solid #e1e8ed;
                display: flex;
                align-items: flex-end;
                gap: 10px;
            }}

            #messageInput {{
                flex: 1;
                padding: 12px;
                border: 2px solid #e1e8ed;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s;
            }}

            #messageInput:focus {{
                border-color: #667eea;
            }}

            #sendButton {{
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 24px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, opacity 0.3s;
            }}

            #sendButton:hover {{
                transform: translateY(-2px);
            }}

            #sendButton:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}

            .decode-controls {{
                display: flex;
                flex-direction: column;
                gap: 6px;
                min-width: 150px;
            }}

            .decode-controls label {{
                font-size: 12px;
                font-weight: 600;
                color: #555;
            }}

            .decode-controls select {{
                padding: 8px 12px;
                border: 2px solid #e1e8ed;
                border-radius: 12px;
                font-size: 13px;
                outline: none;
                transition: border-color 0.3s;
                background: #f7f9fc;
            }}

            .decode-controls select:focus {{
                border-color: #667eea;
            }}

            .loading {{
                display: flex;
                gap: 5px;
                padding: 12px 16px;
            }}

            .loading span {{
                width: 8px;
                height: 8px;
                background: #667eea;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
            }}

            .loading span:nth-child(1) {{ animation-delay: -0.32s; }}
            .loading span:nth-child(2) {{ animation-delay: -0.16s; }}

            @keyframes bounce {{
                0%, 80%, 100% {{ transform: scale(0); }}
                40% {{ transform: scale(1); }}
            }}

            .stats {{
                font-size: 12px;
                color: #666;
                margin-top: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{config.model_type.upper()} - {pipeline_label}</h1>
                <div class="subtitle">{subtitle}</div>
            </div>

            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        <div>{intro_headline}</div>
                        <div style="margin-top: 8px;">{intro_detail}</div>
                        <div class="message-meta">Model: {config.model_type.upper()} | Pipeline: {pipeline_mode} | Decoder steps ≤ {config.max_decode_steps}</div>
                        <div class="message-meta" style="font-size: 11px; color: #888;">Model file: {config.model_path}</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="decode-controls">
                    <label for="decodeStepsSelect">Vec2Text steps</label>
                    <select id="decodeStepsSelect"></select>
                </div>
                <div class="decode-controls">
                    <label for="chunkModeSelect">Auto-chunk</label>
                    <select id="chunkModeSelect">
                        <option value="adaptive">Adaptive (≥5 sentences)</option>
                        <option value="sentence">By Sentence (1:1)</option>
                        <option value="fixed">Fixed (force 5 chunks)</option>
                        <option value="off">Off (1 vector)</option>
                    </select>
                </div>
                <input
                    type="text"
                    id="messageInput"
                    placeholder="Type your message..."
                    autocomplete="off"
                />
                <button id="sendButton">Send</button>
            </div>
        </div>

        <script>
            const messagesDiv = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const decodeStepsSelect = document.getElementById('decodeStepsSelect');
            const chunkModeSelect = document.getElementById('chunkModeSelect');
            const maxDecodeSteps = {config.max_decode_steps};

            for (let i = 1; i <= maxDecodeSteps; i++) {{
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${{i}} step${{i > 1 ? 's' : ''}}`;
                decodeStepsSelect.appendChild(option);
            }}
            decodeStepsSelect.value = '1';
            chunkModeSelect.value = 'adaptive';

            let conversationHistory = [];

            function addMessage(content, isUser, meta = null) {{
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{isUser ? 'user' : 'assistant'}}`;

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = content;

                if (meta) {{
                    const metaDiv = document.createElement('div');
                    metaDiv.className = 'message-meta';
                    metaDiv.textContent = meta;
                    contentDiv.appendChild(metaDiv);
                }}

                messageDiv.appendChild(contentDiv);
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}

            function addLoadingIndicator() {{
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                messageDiv.id = 'loading-indicator';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';

                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'loading';
                loadingDiv.innerHTML = '<span></span><span></span><span></span>';

                contentDiv.appendChild(loadingDiv);
                messageDiv.appendChild(contentDiv);
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}

            function removeLoadingIndicator() {{
                const loading = document.getElementById('loading-indicator');
                if (loading) loading.remove();
            }}

            async function sendMessage() {{
                const message = messageInput.value.trim();
                if (!message) return;

                const selectedSteps = Math.min(
                    maxDecodeSteps,
                    Math.max(1, parseInt(decodeStepsSelect.value, 10) || 1)
                );
                const selectedChunkMode = chunkModeSelect.value;
                decodeStepsSelect.value = String(selectedSteps);
                chunkModeSelect.value = selectedChunkMode;

                // Add user message
                addMessage(message, true);
                conversationHistory.push(message);

                // Keep only last {config.context_length} messages
                if (conversationHistory.length > {config.context_length}) {{
                    conversationHistory = conversationHistory.slice(-{config.context_length});
                }}

                // Clear input and disable
                messageInput.value = '';
                sendButton.disabled = true;
                messageInput.disabled = true;
                decodeStepsSelect.disabled = true;
                chunkModeSelect.disabled = true;

                // Show loading
                addLoadingIndicator();

                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            messages: conversationHistory,
                            temperature: 1.0,
                            decode_steps: selectedSteps,
                            auto_chunk: selectedChunkMode !== 'off',
                            chunk_mode: selectedChunkMode
                        }})
                    }});

                    if (!response.ok) {{
                        throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                    }}

                    const data = await response.json();

                    removeLoadingIndicator();

                    // Add assistant response
                    const chunkInfo = data.chunking_applied ? ` | Chunks: ${{data.chunks_used}}` : '';
                    const meta = `Latency: ${{data.total_latency_ms.toFixed(0)}}ms | LVM: ${{data.latency_breakdown.lvm_inference_ms.toFixed(1)}}ms | Vec2Text: ${{selectedSteps}} step${{selectedSteps > 1 ? 's' : ''}}${{chunkInfo}} | Confidence: ${{(data.confidence * 100).toFixed(1)}}%`;
                    addMessage(data.response, false, meta);

                    conversationHistory.push(data.response);
                    if (conversationHistory.length > {config.context_length}) {{
                        conversationHistory = conversationHistory.slice(-{config.context_length});
                    }}

                }} catch (error) {{
                    removeLoadingIndicator();
                    addMessage(`Error: ${{error.message}}`, false);
                }} finally {{
                    sendButton.disabled = false;
                    messageInput.disabled = false;
                    decodeStepsSelect.disabled = false;
                    chunkModeSelect.disabled = false;
                    messageInput.focus();
                }}
            }}

            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage();
                }}
            }});

            // Focus input on load
            messageInput.focus();
        </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9001)
