#!/usr/bin/env python3
"""
Orchestrator-based GTR-T5 Encoder Service (Port 7001) - OPTIMIZED
Fast in-memory encoding using pre-loaded GTR-T5 model (no subprocess spawning)

CRITICAL: Uses Vec2TextProcessor's orchestrator encoder for compatibility with port 7002 decoder
DO NOT use port 8767 with this decoder - they are INCOMPATIBLE.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Vec2TextProcessor to use its orchestrator encoder (for compatibility)
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator Encoder Service", version="2.0.0")

# Global orchestrator instance (stays warm in memory)
orchestrator: Optional[IsolatedVecTextVectOrchestrator] = None


class EncodeRequest(BaseModel):
    texts: List[str]


class EncodeResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]


@app.on_event("startup")
async def load_model():
    """Load Vec2Text orchestrator encoder on startup and keep it warm in memory"""
    global orchestrator
    logger.info("🚀 Starting GTR-T5 Encoder (Port 7001)...")
    logger.info("Loading Vec2Text orchestrator encoder (for compatibility with port 7002)...")

    # Use the same orchestrator encoder that the decoder uses
    # This ensures 100% compatibility between encoder and decoder
    orchestrator = IsolatedVecTextVectOrchestrator(
        steps=1,  # Steps only matter for decoding, not encoding
        debug=False,
        vec2text_backend="isolated"
    )

    logger.info("✅ Orchestrator encoder ready (in-memory mode, vec2text-compatible)")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global orchestrator
    orchestrator = None
    logger.info("Orchestrator encoder unloaded")


@app.get("/health")
async def health():
    """Health check endpoint"""
    if orchestrator is None:
        return {"status": "initializing"}

    return {
        "status": "ok",
        "service": "orchestrator_encoder",
        "port": 7001,
        "mode": "in_memory",  # NEW: indicates optimization
        "encoder": "GTR-T5 (vec2text orchestrator - compatible with port 7002)",
        "compatible_decoder": "port 7002",
        "incompatible_decoders": ["port 8766"],
        "output_dimensions": 768,
        "model": "vec2text-gtr-t5-base"
    }


@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    """
    Encode texts to 768D vectors using in-memory Vec2Text orchestrator (FAST!)

    These vectors are COMPATIBLE with the decoder on port 7002.
    DO NOT use these vectors with port 8766 decoder - they are incompatible.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        start_time = time.time()
        logger.info(f"Encoding {len(request.texts)} texts...")

        # Use orchestrator's encoder (same as decoder uses)
        # This ensures vectors are compatible with port 7002 decoder
        embeddings_tensor = orchestrator.encode_texts(request.texts)

        # Convert to numpy and normalize (vec2text standard)
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        # Normalize to unit length
        norms = np.linalg.norm(embeddings_numpy, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings_numpy = embeddings_numpy / norms

        # Convert to list format
        embeddings_list = embeddings_numpy.tolist()

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"✓ Encoded to shape {embeddings_numpy.shape} in {elapsed_ms}ms")

        return EncodeResponse(
            embeddings=embeddings_list,
            shape=list(embeddings_numpy.shape)
        )

    except Exception as e:
        logger.error(f"Encoding failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7001)
