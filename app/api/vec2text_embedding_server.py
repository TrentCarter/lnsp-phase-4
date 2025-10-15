#!/usr/bin/env python3
"""
Vec2Text-Compatible GTR-T5 Embedding FastAPI Server
Keeps vec2text's GTR-T5 encoder warm in memory for vec2text-compatible embeddings
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

app = FastAPI(
    title="Vec2Text-Compatible GTR-T5 Embedding Server",
    description="Always-on vec2text-compatible GTR-T5 embedding service (Port 8767)",
    version="1.0.0"
)

# Global model instance (stays warm)
vec2text_orchestrator: Optional[IsolatedVecTextVectOrchestrator] = None


class EmbedRequest(BaseModel):
    """Request for embedding generation"""
    texts: List[str] = Field(..., description="List of texts to embed")
    normalize: bool = Field(default=True, description="Normalize vectors to unit length (always True for vec2text)")


class EmbedResponse(BaseModel):
    """Response with embeddings"""
    embeddings: List[List[float]]
    dimension: int
    count: int
    encoder: str = "vec2text-gtr-t5-base"


@app.on_event("startup")
async def load_model():
    """Load vec2text's GTR-T5 encoder on startup"""
    global vec2text_orchestrator
    print("Loading vec2text GTR-T5 encoder...")

    # Force CPU for vec2text compatibility
    os.environ['VEC2TEXT_FORCE_CPU'] = '1'

    vec2text_orchestrator = IsolatedVecTextVectOrchestrator()
    print("✅ Vec2text GTR-T5 encoder loaded and ready (CPU mode)")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global vec2text_orchestrator
    vec2text_orchestrator = None
    print("Vec2text GTR-T5 encoder unloaded")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if vec2text_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": "vec2text-gtr-t5-base",
        "dimension": 768,
        "encoder": "vec2text-compatible",
        "device": str(vec2text_orchestrator._device)
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate vec2text-compatible embeddings for texts"""
    if vec2text_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Generate embeddings using vec2text's encoder
        embeddings_tensor = vec2text_orchestrator.encode_texts(request.texts)

        # Convert to numpy
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        # Normalize (vec2text encoder already normalizes, but ensure)
        if request.normalize:
            norms = np.linalg.norm(embeddings_numpy, axis=1, keepdims=True)
            embeddings_numpy = embeddings_numpy / norms

        # Convert to list
        embeddings_list = embeddings_numpy.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=embeddings_numpy.shape[1],
            count=len(embeddings_list),
            encoder="vec2text-gtr-t5-base"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/embed/single")
async def embed_single(text: str):
    """Generate vec2text-compatible embedding for a single text (convenience endpoint)"""
    if vec2text_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        embeddings_tensor = vec2text_orchestrator.encode_texts([text])
        embeddings_numpy = embeddings_tensor.cpu().detach().numpy()

        # Normalize
        norms = np.linalg.norm(embeddings_numpy, axis=1, keepdims=True)
        embeddings_numpy = embeddings_numpy / norms

        return {
            "embedding": embeddings_numpy[0].tolist(),
            "dimension": embeddings_numpy.shape[1],
            "encoder": "vec2text-gtr-t5-base"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8767,
        log_level="info"
    )
