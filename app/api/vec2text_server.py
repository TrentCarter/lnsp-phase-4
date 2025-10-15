#!/usr/bin/env python3
"""
Vec2Text FastAPI Server - In-Memory Version
Keeps vec2text models (JXE + IELab) warm in memory for fast decoding
"""

import asyncio
import os
import sys
import time
import signal
import contextlib
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Vec2TextProcessor for in-memory decoding
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Import Vec2TextProcessor for in-memory decoding
from app.vect_text_vect.vec2text_processor import Vec2TextProcessor, Vec2TextConfig
from app.vect_text_vect.procrustes_adapter import ProcrustesAdapter

app = FastAPI(
    title="Vec2Text Decoding Server",
    description="Always-on vec2text service for LNSP (JXE + IELab decoders)",
    version="2.0.0"
)

# Global vec2text processors (kept warm in memory)
vec2text_processors = {}

# Optional Procrustes adapter for aligning external embeddings
PROCRUSTES_PATH = os.getenv("VEC2TEXT_PROCRUSTES_PATH")
_procrustes_adapter: ProcrustesAdapter | None = None

# Supported decoders for in-memory processing
decoder_configs = {
    'jxe': {
        'teacher_model': 'sentence-transformers/gtr-t5-base',
        'device': 'cpu',
        'random_seed': 42  # Consistent seed for deterministic decoding
    },
    'ielab': {
        'teacher_model': 'sentence-transformers/gtr-t5-base',
        'device': 'cpu',
        'random_seed': 42  # Same seed for consistency
    }
}

# Legacy orchestrator for encoding (when needed)
_orchestrator: IsolatedVecTextVectOrchestrator | None = None
_orch_lock = asyncio.Lock()

async def _ensure_orchestrator(steps: int = 1) -> IsolatedVecTextVectOrchestrator:
    """Lazy-load the vec2text orchestrator once and update its step count."""
    global _orchestrator
    if _orchestrator is None:
        async with _orch_lock:
            if _orchestrator is None:
                _orchestrator = IsolatedVecTextVectOrchestrator(
                    steps=steps, vec2text_backend="isolated"
                )
    if _orchestrator is not None:
        _orchestrator.steps = steps
    return _orchestrator


class Vec2TextRequest(BaseModel):
    """Request for vec2text decoding"""
    vectors: List[List[float]] = Field(
        ..., description="List of embedding vectors to decode (typically 768D)"
    )
    subscribers: str = Field(
        default="jxe,ielab", description="Comma-separated decoders: jxe,ielab"
    )
    steps: int = Field(default=1, description="Number of decoding steps (1-20)")
    device: str = Field(default="cpu", description="Device: cpu, mps, cuda")
    apply_adapter: bool = Field(
        default=True,
        description="Apply Procrustes adapter before decoding when available",
    )


class Vec2TextResponse(BaseModel):
    """Response with decoded texts"""
    results: List[dict]
    count: int


class TextToVecRequest(BaseModel):
    """Request for text encoding (via GTR-T5) then decoding"""
    texts: List[str] = Field(..., description="Texts to encode then decode")
    subscribers: str = Field(
        default="jxe,ielab", description="Comma-separated decoders"
    )
    steps: int = Field(default=1, description="Number of decoding steps")
    apply_adapter: bool = Field(
        default=False,
        description="Apply adapter to the encoded vectors before decoding",
    )


class EmbedRequest(BaseModel):
    """Request for encoding texts to embeddings"""
    texts: List[str] = Field(..., description="Texts to encode to embeddings")
    normalize: bool = Field(default=True, description="Normalize embeddings to unit length")


class EmbedResponse(BaseModel):
    """Response with embeddings"""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimension: int = Field(..., description="Dimension of each embedding")
    count: int = Field(..., description="Number of embeddings")
    encoder: str = Field(..., description="Encoder model used")


class SelfTestRequest(BaseModel):
    """Request payload for the self-test endpoint."""

    text: str = Field(
        default="Photosynthesis converts light energy to chemical energy in plants.",
        description="Sample text used to produce reference embeddings",
    )
    subscribers: str = Field(
        default="jxe,ielab",
        description="Comma-separated decoders to probe",
    )
    steps: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of decoding steps for the probe",
    )
    apply_adapter: bool = Field(
        default=True,
        description="Test the adapter path when configured",
    )
    vector: Optional[List[float]] = Field(
        default=None,
        description="Optional external vector (pre-adaptation) to validate",
    )


VALID_DECODERS = {"jxe", "ielab"}
VALID_DEVICES = {"cpu", "mps", "cuda"}


@contextlib.contextmanager
def timeout(duration):
    """Timeout context manager to prevent hanging operations"""
    def timeout_handler(signum, frame):
        _ = signum  # Mark as used
        _ = frame   # Mark as used
        raise TimeoutError(f"Operation timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def _parse_subscribers(raw: str) -> List[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        return ["jxe", "ielab"]
    invalid = [name for name in values if name not in VALID_DECODERS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown decoders: {invalid}")
    # Preserve order but de-duplicate while respecting first occurrence
    seen: Dict[str, None] = {}
    for name in values:
        if name not in seen:
            seen[name] = None
    return list(seen.keys())


def _validate_device(raw: str) -> str:
    device = (raw or "cpu").lower()
    if device not in VALID_DEVICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device '{raw}'. Expected one of {sorted(VALID_DEVICES)}",
        )
    return device


def _ensure_orchestrator(steps: int) -> IsolatedVecTextVectOrchestrator:
    """Get or create an orchestrator instance for encoding"""
    # For now, create a simple orchestrator for encoding
    # This ensures we use the same encoder as the decoders
    try:
        return IsolatedVecTextVectOrchestrator(steps=steps, debug=False)
    except Exception as e:
        print(f"❌ Failed to create orchestrator: {e}")
        raise


def _decode_single_vector_in_memory(
    vector: np.ndarray,
    subscribers: List[str],
    device_override: str,
    steps: int,
    apply_adapter: bool,
) -> Dict[str, dict]:
    """Run the requested decoders for a single embedding using in-memory processors."""
    _ = device_override  # Device is set at processor creation time

    results: Dict[str, dict] = {}

    # Optionally align incoming vectors to the decoder space
    transformed_vector = vector.astype(np.float32, copy=False)
    if apply_adapter and _procrustes_adapter is not None:
        try:
            transformed_vector = _procrustes_adapter.adapt([transformed_vector])[0]
        except Exception as exc:
            results["adapter"] = {
                "status": "error",
                "error": f"Adapter failed: {exc}",
                "elapsed_ms": 0,
            }
            transformed_vector = vector.astype(np.float32, copy=False)

    # Prepare tensor once per vector
    vector_tensor = torch.from_numpy(transformed_vector).unsqueeze(0)

    for decoder_name in subscribers:
        if decoder_name not in vec2text_processors:
            results[f"gtr → {decoder_name}"] = {
                "status": "error",
                "error": f"Decoder {decoder_name} not available (not loaded at startup)",
                "elapsed_ms": 0,
            }
            continue

        processor = vec2text_processors[decoder_name]
        start = time.time()

        try:
            # Use the in-memory processor for decoding
            print(f"🔍 Decoding with {decoder_name}, steps={steps}")

            # Add timeout to prevent hanging (30 seconds max)
            with timeout(30):
                decoded_info = processor.decode_embeddings(
                    vector_tensor,
                    num_iterations=steps,
                    beam_width=1,  # Default beam width
                    prompts=[""]  # Empty prompt
                )
            print(f"✅ {decoder_name} decoding completed")

            elapsed_ms = round((time.time() - start) * 1000, 2)

            if decoded_info and len(decoded_info) > 0:
                decoded_text = decoded_info[0].get("final_text", "")
                print(f"📝 {decoder_name} output: {decoded_text[:100]}...")

                # Calculate cosine similarity
                try:
                    decoded_vec = processor._embed_text(decoded_text)
                    cosine = torch.cosine_similarity(
                        vector_tensor.squeeze(0), decoded_vec.squeeze(0), dim=0
                    ).item()
                except Exception as e:
                    print(f"⚠️ Cosine calculation failed for {decoder_name}: {e}")
                    cosine = 0.0

                results[f"gtr → {decoder_name}"] = {
                    "status": "success",
                    "output": decoded_text,
                    "cosine": round(float(cosine), 4),
                    "elapsed_ms": elapsed_ms,
                }
            else:
                print(f"❌ {decoder_name} returned no results")
                results[f"gtr → {decoder_name}"] = {
                    "status": "error",
                    "error": "Decoder returned no results",
                    "elapsed_ms": elapsed_ms,
                }

        except TimeoutError as e:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            print(f"⏰ {decoder_name} timed out: {e}")
            results[f"gtr → {decoder_name}"] = {
                "status": "error",
                "error": f"Decoder timed out: {e}",
                "elapsed_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            print(f"❌ {decoder_name} failed: {exc}")
            results[f"gtr → {decoder_name}"] = {
                "status": "error",
                "error": f"Decoder failed: {exc}",
                "elapsed_ms": elapsed_ms,
            }

    return results


async def load_vec2text_processors():
    """Load vec2text processors at startup and keep them warm in memory"""
    global vec2text_processors

    print("Loading vec2text processors...")

    for decoder_name, config in decoder_configs.items():
        try:
            vec2text_config = Vec2TextConfig(
                teacher_model=config['teacher_model'],
                device=config['device'],
                random_seed=config['random_seed'],  # Use consistent seed
                debug=False
            )
            processor = Vec2TextProcessor(vec2text_config)
            vec2text_processors[decoder_name] = processor
            print(f"✅ {decoder_name.upper()} processor loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {decoder_name.upper()} processor: {e}")
            raise

    print(f"✅ All vec2text processors loaded ({len(vec2text_processors)} total)")


async def cleanup_vec2text_processors():
    """Cleanup processors on shutdown"""
    global vec2text_processors
    global _procrustes_adapter
    vec2text_processors.clear()
    _procrustes_adapter = None
    print("Vec2text processors unloaded")


def load_procrustes_adapter() -> None:
    """Load an optional Procrustes adapter if configured."""
    global _procrustes_adapter
    if not PROCRUSTES_PATH:
        _procrustes_adapter = None
        return
    try:
        adapter_path = Path(PROCRUSTES_PATH)
        _procrustes_adapter = ProcrustesAdapter.load(adapter_path)
        print(f"✅ Procrustes adapter loaded from {adapter_path}")
    except FileNotFoundError:
        _procrustes_adapter = None
        print(f"⚠️  Procrustes adapter path not found: {PROCRUSTES_PATH}")
    except Exception as exc:
        _procrustes_adapter = None
        print(f"⚠️  Failed to load Procrustes adapter: {exc}")


@app.on_event("startup")
async def startup():
    """Load vec2text processors at startup and keep them warm in memory."""

    print("Vec2Text server starting...")
    try:
        await load_vec2text_processors()
        load_procrustes_adapter()
        print("✅ Vec2Text server ready (in-memory mode)")
    except Exception as exc:
        print(f"❌ Failed to start Vec2Text server: {exc}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await cleanup_vec2text_processors()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not vec2text_processors:
        raise HTTPException(status_code=503, detail="Vec2Text processors not loaded")

    loaded_decoders = list(vec2text_processors.keys())
    return {
        "status": "healthy",
        "decoders": loaded_decoders,
        "dimensions": 768,
        "mode": "in_memory"
    }


@app.get("/config")
async def config_snapshot():
    """Return the runtime configuration for quick diagnostics."""
    decoder_info = {
        name: {
            "teacher_model": cfg.get("teacher_model"),
            "device": cfg.get("device"),
            "random_seed": cfg.get("random_seed"),
        }
        for name, cfg in decoder_configs.items()
    }
    adapter_block: Dict[str, object] | None
    if _procrustes_adapter is not None:
        adapter_block = {
            "path": PROCRUSTES_PATH,
            "dimension": _procrustes_adapter.dimension,
        }
    else:
        adapter_block = None
    return {
        "decoders": decoder_info,
        "adapter": adapter_block,
        "adapter_loaded": adapter_block is not None,
        "vector_dimension": 768,
        "default_steps": 1,
    }


@app.post("/decode", response_model=Vec2TextResponse)
async def decode_vectors(request: Vec2TextRequest):
    """Decode vectors to text using vec2text"""
    if not request.vectors:
        raise HTTPException(status_code=400, detail="No vectors provided")

    if request.steps < 1 or request.steps > 20:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 20")

    subscribers = _parse_subscribers(request.subscribers)
    device = _validate_device(request.device)

    # Use in-memory processors instead of subprocess calls
    results = []

    for i, vector in enumerate(request.vectors):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Vector at index {i} must be 1-dimensional, got shape {arr.shape}",
            )

        vector_entry = {
            "index": i,
            "subscribers": _decode_single_vector_in_memory(
                arr,
                subscribers,
                device,
                request.steps,
                request.apply_adapter,
            ),
        }
        results.append(vector_entry)

    return Vec2TextResponse(results=results, count=len(results))


@app.post("/selftest")
async def self_test(request: SelfTestRequest):
    """Run a quick end-to-end probe to validate decoder health."""

    subscribers = _parse_subscribers(request.subscribers)
    if request.steps < 1 or request.steps > 20:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 20")

    orchestrator = _ensure_orchestrator(request.steps)
    try:
        teacher_vector = orchestrator.encode_texts([request.text]).cpu().numpy()[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to encode text: {exc}")

    teacher_cycle = _decode_single_vector_in_memory(
        teacher_vector,
        subscribers,
        "cpu",
        request.steps,
        apply_adapter=False,
    )

    adapter_cycle: Dict[str, dict] | None = None
    adapter_source = None
    if request.apply_adapter and _procrustes_adapter is not None:
        if request.vector is not None:
            adapter_source = "request"
            candidate = np.asarray(request.vector, dtype=np.float32)
        else:
            adapter_source = "teacher"
            candidate = teacher_vector
        if candidate.ndim != 1:
            raise HTTPException(status_code=400, detail="adapter vector must be 1-D")
        if candidate.shape[0] != teacher_vector.shape[0]:
            raise HTTPException(
                status_code=400,
                detail=f"adapter vector dimension mismatch: expected {teacher_vector.shape[0]}",
            )
        adapter_cycle = _decode_single_vector_in_memory(
            candidate,
            subscribers,
            "cpu",
            request.steps,
            apply_adapter=True,
        )

    return {
        "text": request.text,
        "steps": request.steps,
        "subscribers": subscribers,
        "adapter_available": _procrustes_adapter is not None,
        "adapter_cycle": adapter_cycle,
        "adapter_source": adapter_source,
        "teacher_cycle": teacher_cycle,
    }


@app.post("/encode")
async def encode_texts(request: EmbedRequest):
    """Encode texts to vec2text-compatible embeddings (768D)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Use the same orchestrator as the decoder for consistency
        orchestrator = _ensure_orchestrator(steps=1)
        embeddings_tensor = orchestrator.encode_texts(request.texts)

        # Convert to numpy and normalize (same as decoder expects)
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        # Normalize to unit length (vec2text standard)
        if request.normalize:
            norms = np.linalg.norm(embeddings_numpy, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings_numpy = embeddings_numpy / norms

        # Convert to list format
        embeddings_list = embeddings_numpy.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=embeddings_numpy.shape[1],
            count=len(embeddings_list),
            encoder="vec2text-gtr-t5-base"
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text encoding failed: {exc}")


@app.post("/encode/single")
async def encode_single(text: str):
    """Encode a single text to vec2text-compatible embedding (convenience endpoint)"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        # Use the same orchestrator as the decoder for consistency
        orchestrator = _ensure_orchestrator(steps=1)
        embeddings_tensor = orchestrator.encode_texts([text])
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        # Normalize to unit length
        norms = np.linalg.norm(embeddings_numpy, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_numpy = embeddings_numpy / norms

        return {
            "embedding": embeddings_numpy[0].tolist(),
            "dimension": embeddings_numpy.shape[1],
            "encoder": "vec2text-gtr-t5-base"
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text encoding failed: {exc}")


@app.post("/encode-decode")
async def encode_then_decode(request: TextToVecRequest):
    """Encode texts to vectors (GTR-T5) then decode back (round-trip test)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if request.steps < 1 or request.steps > 20:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 20")

    subscribers = _parse_subscribers(request.subscribers)

    # Use in-memory processors for encoding and decoding
    try:
        # Use the same orchestrator as the decoder for consistency
        orchestrator = _ensure_orchestrator(request.steps)
        encoded = orchestrator.encode_texts(request.texts).cpu().numpy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text encoding failed: {exc}")

    results = []
    for idx, text in enumerate(request.texts):
        vector_entry = {
            "index": idx,
            "input_text": text,
            "subscribers": _decode_single_vector_in_memory(
                encoded[idx],
                subscribers,
                "cpu",
                request.steps,
                request.apply_adapter,
            ),
        }
        results.append(vector_entry)

    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8766,
        log_level="info"
    )
