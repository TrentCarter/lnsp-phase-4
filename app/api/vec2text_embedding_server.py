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


class DecodeRequest(BaseModel):
    """Request for decoding vectors to text"""
    vectors: List[List[float]] = Field(..., description="List of 768D vectors to decode")
    steps: int = Field(default=1, ge=1, le=20, description="Number of vec2text decoding steps")


class DecodeResponse(BaseModel):
    """Response with decoded texts"""
    results: List[dict]
    count: int


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


@app.post("/decode", response_model=DecodeResponse)
async def decode_vectors(request: DecodeRequest):
    """Decode 768D vectors back to text using vec2text"""
    if vec2text_orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.vectors:
        raise HTTPException(status_code=400, detail="No vectors provided")

    try:
        import torch
        import torch.nn.functional as F
        from vec2text.api import load_pretrained_corrector, invert_embeddings

        # Load vec2text corrector (JXE model) - force CPU for compatibility
        import os
        os.environ['PYTORCH_MPS_DISABLE'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        corrector = load_pretrained_corrector("gtr-base")

        # Force corrector models to CPU
        device = torch.device("cpu")
        if hasattr(corrector, 'model'):
            corrector.model = corrector.model.to(device)
        if hasattr(corrector, 'inversion_trainer') and hasattr(corrector.inversion_trainer, 'model'):
            corrector.inversion_trainer.model = corrector.inversion_trainer.model.to(device)

        results = []
        for idx, vector in enumerate(request.vectors):
            # Convert to tensor and normalize (on CPU)
            vector_tensor = torch.tensor([vector], dtype=torch.float32, device=device)
            vector_tensor = F.normalize(vector_tensor, dim=-1)

            # Decode using vec2text
            with torch.no_grad():
                decoded_texts = invert_embeddings(
                    embeddings=vector_tensor,
                    corrector=corrector,
                    num_steps=request.steps,
                    sequence_beam_width=1
                )

            output_text = decoded_texts[0] if decoded_texts else ""

            # Calculate cosine similarity with original vector
            try:
                decoded_vec = vec2text_orchestrator.encode_texts([output_text])
                decoded_vec = F.normalize(decoded_vec, dim=-1).to(device)
                cosine = F.cosine_similarity(vector_tensor, decoded_vec, dim=1).item()
            except:
                cosine = 0.0

            results.append({
                "index": idx,
                "output": output_text,
                "cosine": round(cosine, 4),
                "status": "success"
            })

        return DecodeResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Vector decoding failed: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8767,
        log_level="info"
    )
