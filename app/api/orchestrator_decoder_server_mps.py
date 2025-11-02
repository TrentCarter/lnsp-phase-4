#!/usr/bin/env python3
"""
Orchestrator-based Vec2Text Decoder Service (Port 7004) - MPS OPTIMIZED
Fast in-memory decoding using pre-loaded Vec2Text models on MPS device

This service uses Vec2TextProcessor for decoding, keeping models warm in memory.
ONLY use vectors from port 7003. DO NOT use vectors from port 8767.
"""

import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Vec2TextProcessor for in-memory decoding
from app.vect_text_vect.vec2text_processor import Vec2TextProcessor, Vec2TextConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vec2text processors (kept warm in memory)
vec2text_processors: Dict[str, Vec2TextProcessor] = {}

# Decoder configurations (MPS for Apple Silicon)
decoder_configs = {
    'jxe': {
        'teacher_model': 'sentence-transformers/gtr-t5-base',
        'device': 'mps',  # MPS for Apple Silicon
        'random_seed': 42
    },
    'ielab': {
        'teacher_model': 'sentence-transformers/gtr-t5-base',
        'device': 'mps',  # MPS for Apple Silicon (Note: IELab is CPU-only, but we'll try)
        'random_seed': 42
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup - Pre-load Vec2Text models
    logger.info("🚀 Starting Vec2Text Decoder (Port 7004 - MPS)...")
    try:
        await load_vec2text_processors()
        logger.info("✅ Orchestrator decoder ready (in-memory mode, MPS device)")
    except Exception as exc:
        logger.error(f"❌ Failed to start decoder: {exc}")
        raise

    yield

    # Shutdown - Cleanup
    await cleanup_vec2text_processors()


app = FastAPI(
    title="Orchestrator Decoder Service (MPS)",
    version="2.0.0",
    lifespan=lifespan
)


class DecodeRequest(BaseModel):
    vectors: List[List[float]]
    subscriber: str = "ielab"  # or "jxe"
    steps: Optional[int] = 1  # Default to 1 for speed
    original_texts: Optional[List[str]] = None  # For metadata


class DecodeResponse(BaseModel):
    results: List[str]
    subscriber: str
    steps: int


async def load_vec2text_processors():
    """Load vec2text processors at startup and keep them warm in memory (MPS)"""
    global vec2text_processors

    logger.info("Loading vec2text processors (MPS)...")

    for decoder_name, config in decoder_configs.items():
        try:
            vec2text_config = Vec2TextConfig(
                teacher_model=config['teacher_model'],
                device=config['device'],
                random_seed=config['random_seed'],
                debug=False
            )
            processor = Vec2TextProcessor(vec2text_config)
            vec2text_processors[decoder_name] = processor
            logger.info(f"✅ {decoder_name.upper()} processor loaded successfully (MPS)")
        except Exception as e:
            logger.error(f"❌ Failed to load {decoder_name.upper()} processor: {e}")
            # If MPS fails for IELab, fall back to CPU
            if decoder_name == 'ielab' and 'mps' in str(e).lower():
                logger.warning(f"⚠️  IELab may not support MPS, falling back to CPU...")
                try:
                    vec2text_config = Vec2TextConfig(
                        teacher_model=config['teacher_model'],
                        device='cpu',
                        random_seed=config['random_seed'],
                        debug=False
                    )
                    processor = Vec2TextProcessor(vec2text_config)
                    vec2text_processors[decoder_name] = processor
                    logger.info(f"✅ {decoder_name.upper()} processor loaded successfully (CPU fallback)")
                except Exception as e2:
                    logger.error(f"❌ CPU fallback also failed: {e2}")
                    raise
            else:
                raise

    logger.info(f"✅ All vec2text processors loaded ({len(vec2text_processors)} total)")


async def cleanup_vec2text_processors():
    """Cleanup processors on shutdown"""
    global vec2text_processors
    vec2text_processors.clear()
    logger.info("Vec2text processors unloaded")


def decode_single_vector_in_memory(
    vector: np.ndarray,
    subscriber: str,
    steps: int
) -> str:
    """Decode a single vector using in-memory processor on MPS (FAST - no subprocess)"""

    if subscriber not in vec2text_processors:
        raise ValueError(f"Decoder {subscriber} not available")

    processor = vec2text_processors[subscriber]

    # Prepare tensor
    vector_tensor = torch.from_numpy(vector.astype(np.float32)).unsqueeze(0)

    try:
        # Use in-memory processor for decoding (THIS IS FAST!)
        decoded_info = processor.decode_embeddings(
            vector_tensor,
            num_iterations=steps,
            beam_width=1,
            prompts=[""]
        )

        if decoded_info and len(decoded_info) > 0:
            decoded_text = decoded_info[0].get("final_text", "")
            return decoded_text
        else:
            raise ValueError("Decoder returned no results")

    except Exception as exc:
        raise RuntimeError(f"Decoding failed: {exc}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    if not vec2text_processors:
        return {"status": "initializing"}

    return {
        "status": "ok",
        "service": "orchestrator_decoder_mps",
        "port": 7004,
        "device": "mps",
        "mode": "in_memory",
        "decoder": "Vec2Text (ielab/jxe)",
        "compatible_encoder": "port 7003",
        "incompatible_encoders": ["port 8767"],
        "subscribers": list(vec2text_processors.keys()),
        "loaded": len(vec2text_processors) > 0
    }


@app.post("/decode", response_model=DecodeResponse)
async def decode(request: DecodeRequest):
    """
    Decode 768D vectors to text using in-memory Vec2Text processors on MPS (FAST!)

    These vectors MUST come from port 7003 encoder.
    DO NOT use vectors from port 8767 - they are incompatible.
    """
    if not vec2text_processors:
        raise HTTPException(status_code=503, detail="Vec2Text processors not initialized")

    if not request.vectors:
        raise HTTPException(status_code=400, detail="No vectors provided")

    if request.subscriber not in ["ielab", "jxe"]:
        raise HTTPException(status_code=400, detail="Subscriber must be 'ielab' or 'jxe'")

    try:
        start_time = time.time()
        logger.info(f"Decoding {len(request.vectors)} vectors with {request.subscriber} (steps={request.steps}, MPS)...")

        # Decode all vectors
        decoded_texts = []
        for i, vector in enumerate(request.vectors):
            arr = np.asarray(vector, dtype=np.float32)
            if arr.ndim != 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Vector at index {i} must be 1-dimensional, got shape {arr.shape}"
                )

            # Decode using in-memory processor (FAST - no subprocess!)
            decoded_text = decode_single_vector_in_memory(
                arr,
                request.subscriber,
                request.steps
            )
            decoded_texts.append(decoded_text)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"✓ Decoded {len(decoded_texts)} texts in {elapsed_ms}ms (MPS)")

        return DecodeResponse(
            results=decoded_texts,
            subscriber=request.subscriber,
            steps=request.steps
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decoding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7004)
