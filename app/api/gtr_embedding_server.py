#!/usr/bin/env python3
"""
GTR-T5 Embedding FastAPI Server
Keeps GTR-T5 model warm in memory for fast embedding generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.vectorizer import EmbeddingBackend

app = FastAPI(
    title="GTR-T5 Embedding Server",
    description="Always-on GTR-T5 embedding service for LNSP",
    version="1.0.0"
)

# Global model instance (stays warm)
embedding_backend: Optional[EmbeddingBackend] = None


class EmbedRequest(BaseModel):
    """Request for embedding generation"""
    texts: List[str] = Field(..., description="List of texts to embed")
    normalize: bool = Field(default=True, description="Normalize vectors to unit length")


class EmbedResponse(BaseModel):
    """Response with embeddings"""
    embeddings: List[List[float]]
    dimension: int
    count: int


@app.on_event("startup")
async def load_model():
    """Load GTR-T5 model on startup"""
    global embedding_backend
    print("Loading GTR-T5 model...")
    embedding_backend = EmbeddingBackend()
    print("✅ GTR-T5 model loaded and ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global embedding_backend
    embedding_backend = None
    print("GTR-T5 model unloaded")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if embedding_backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "gtr-t5-base", "dimension": 768}


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for texts"""
    if embedding_backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Generate embeddings
        embeddings = embedding_backend.encode(request.texts)

        # Normalize if requested
        if request.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        # Convert to list
        embeddings_list = embeddings.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=embeddings.shape[1],
            count=len(embeddings_list)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.post("/embed/single")
async def embed_single(text: str):
    """Generate embedding for a single text (convenience endpoint)"""
    if embedding_backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        embeddings = embedding_backend.encode([text])

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return {
            "embedding": embeddings[0].tolist(),
            "dimension": embeddings.shape[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info"
    )
