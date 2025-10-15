#!/usr/bin/env python3
"""
LVM (Latent Vector Model) Inference FastAPI Service

Provides tokenless vector-to-vector prediction using Mamba-style architecture.
Input: 768D vectors + 16D TMD codes → Output: 768D predicted vector
"""

import os
import sys
import time
from typing import List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.lvm.models import LatentMamba

app = FastAPI(
    title="LVM Inference API",
    description="Latent Vector Model inference for tokenless sequence prediction",
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
# Global Model State
# ============================================================================

class ModelState:
    """Global model state"""
    model: Optional[LatentMamba] = None
    device: str = "cpu"
    model_path: Optional[str] = None
    d_input: int = 784  # 768 semantic + 16 TMD
    d_hidden: int = 512
    n_layers: int = 2
    loaded: bool = False


state = ModelState()


# ============================================================================
# Request/Response Models
# ============================================================================

class InferRequest(BaseModel):
    """Request for inference"""
    vector_sequence: List[List[float]] = Field(
        ...,
        description="Sequence of 768D vectors (semantic embeddings)"
    )
    tmd_codes: List[int] = Field(
        ...,
        description="TMD codes [domain, task, modifier] (16D one-hot)",
        min_items=3,
        max_items=3
    )
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    use_mock: bool = Field(default=False, description="Use mock prediction (for testing)")


class InferResponse(BaseModel):
    """Response with predicted vector"""
    predicted_vector: List[float]
    confidence: float
    latency_ms: int
    model_version: str
    is_mock: bool


class BatchInferRequest(BaseModel):
    """Request for batch inference"""
    sequences: List[List[List[float]]] = Field(..., description="Batch of vector sequences")
    tmd_codes_batch: List[List[int]] = Field(..., description="Batch of TMD codes")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    use_mock: bool = Field(default=False, description="Use mock prediction")


class BatchInferResponse(BaseModel):
    """Response with batch predictions"""
    predictions: List[List[float]]
    count: int
    total_latency_ms: int
    avg_latency_ms: float
    is_mock: bool


class ModelInfoResponse(BaseModel):
    """Model metadata"""
    loaded: bool
    model_path: Optional[str]
    d_input: int
    d_hidden: int
    n_layers: int
    device: str
    num_parameters: Optional[int]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    mock_mode_available: bool


# ============================================================================
# Helper Functions
# ============================================================================

def tmd_codes_to_vector(domain: int, task: int, modifier: int) -> np.ndarray:
    """
    Convert TMD codes to 16D one-hot vector.

    Args:
        domain: 0-15 (4 bits)
        task: 0-31 (5 bits)
        modifier: 0-63 (6 bits)

    Returns:
        16D numpy array (one-hot encoded)
    """
    # Create 16D vector
    tmd_vec = np.zeros(16, dtype=np.float32)

    # Encode domain (4 bits)
    for i in range(4):
        if domain & (1 << i):
            tmd_vec[i] = 1.0

    # Encode task (5 bits)
    for i in range(5):
        if task & (1 << i):
            tmd_vec[4 + i] = 1.0

    # Encode modifier (6 bits) - only use first 6 of remaining 7 slots
    for i in range(6):
        if modifier & (1 << i):
            tmd_vec[9 + i] = 1.0

    return tmd_vec


def prepare_input(vector_sequence: List[List[float]], tmd_codes: List[int]) -> np.ndarray:
    """
    Prepare input tensor by concatenating semantic vectors with TMD codes.

    Args:
        vector_sequence: List of 768D vectors
        tmd_codes: [domain, task, modifier]

    Returns:
        numpy array of shape [seq_len, 784] (768 semantic + 16 TMD)
    """
    # Convert to numpy
    semantic_vecs = np.array(vector_sequence, dtype=np.float32)  # [seq_len, 768]

    # Get TMD vector
    domain, task, modifier = tmd_codes
    tmd_vec = tmd_codes_to_vector(domain, task, modifier)  # [16]

    # Broadcast TMD to all sequence positions
    tmd_broadcast = np.tile(tmd_vec, (len(vector_sequence), 1))  # [seq_len, 16]

    # Concatenate
    full_input = np.concatenate([semantic_vecs, tmd_broadcast], axis=1)  # [seq_len, 784]

    return full_input


def mock_inference(vector_sequence: List[List[float]], tmd_codes: List[int]) -> np.ndarray:
    """
    Mock inference for testing (before model is trained).

    Returns a plausible 768D vector based on:
    - Average of input sequence
    - Small random perturbation
    - TMD-based bias

    Args:
        vector_sequence: List of 768D vectors
        tmd_codes: [domain, task, modifier]

    Returns:
        768D numpy array (predicted vector)
    """
    # Average input vectors
    avg_vec = np.mean(vector_sequence, axis=0)

    # Add small random perturbation
    noise = np.random.normal(0, 0.1, size=768).astype(np.float32)

    # Add TMD-based bias (very small)
    domain, task, modifier = tmd_codes
    tmd_bias = np.sin(np.arange(768) * (domain + 1) / 100.0).astype(np.float32) * 0.05

    predicted = avg_vec + noise + tmd_bias

    # Normalize to unit length (typical for GTR-T5 embeddings)
    norm = np.linalg.norm(predicted)
    if norm > 0:
        predicted = predicted / norm

    return predicted


def real_inference(vector_sequence: List[List[float]], tmd_codes: List[int]) -> np.ndarray:
    """
    Real inference using trained LVM model.

    Args:
        vector_sequence: List of 768D vectors
        tmd_codes: [domain, task, modifier]

    Returns:
        768D numpy array (predicted vector)
    """
    if not state.loaded or state.model is None:
        raise ValueError("Model not loaded")

    # Prepare input
    input_arr = prepare_input(vector_sequence, tmd_codes)  # [seq_len, 784]

    # Convert to tensor
    input_tensor = torch.from_numpy(input_arr).unsqueeze(0)  # [1, seq_len, 784]
    input_tensor = input_tensor.to(state.device)

    # Inference
    state.model.eval()
    with torch.no_grad():
        output = state.model(input_tensor)  # [1, 784]

    # Extract semantic part (first 768 dims)
    predicted_784 = output[0].cpu().numpy()  # [784]
    predicted_768 = predicted_784[:768]  # [768]

    # Normalize
    norm = np.linalg.norm(predicted_768)
    if norm > 0:
        predicted_768 = predicted_768 / norm

    return predicted_768


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """API information"""
    return {
        "name": "LVM Inference API",
        "version": "1.0.0",
        "description": "Latent Vector Model inference for tokenless sequence prediction",
        "endpoints": [
            "POST /infer - Single sequence inference",
            "POST /infer/batch - Batch inference",
            "GET /model/info - Model metadata",
            "POST /model/load - Load model from path",
            "GET /health - Health check"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=state.loaded,
        device=state.device,
        mock_mode_available=True
    )


@app.post("/infer", response_model=InferResponse, tags=["inference"])
async def infer_endpoint(request: InferRequest):
    """
    Predict next vector from sequence.

    Input: Sequence of 768D vectors + TMD codes
    Output: Predicted 768D vector
    """
    try:
        start_time = time.time()

        # Validate input dimensions
        for vec in request.vector_sequence:
            if len(vec) != 768:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Expected 768D vectors, got {len(vec)}D"
                )

        # Validate TMD codes
        domain, task, modifier = request.tmd_codes
        if not (0 <= domain <= 15):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Domain code must be 0-15, got {domain}"
            )
        if not (0 <= task <= 31):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task code must be 0-31, got {task}"
            )
        if not (0 <= modifier <= 63):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Modifier code must be 0-63, got {modifier}"
            )

        # Choose inference method
        if request.use_mock or not state.loaded:
            predicted = mock_inference(request.vector_sequence, request.tmd_codes)
            is_mock = True
        else:
            predicted = real_inference(request.vector_sequence, request.tmd_codes)
            is_mock = False

        latency_ms = int((time.time() - start_time) * 1000)

        # Mock confidence (in real system, would be based on model uncertainty)
        confidence = 0.85 if not is_mock else 0.5

        return InferResponse(
            predicted_vector=predicted.tolist(),
            confidence=confidence,
            latency_ms=latency_ms,
            model_version="mamba-768-v1.0" if state.loaded else "mock",
            is_mock=is_mock
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/infer/batch", response_model=BatchInferResponse, tags=["inference"])
async def batch_infer_endpoint(request: BatchInferRequest):
    """Batch inference for multiple sequences"""
    try:
        if len(request.sequences) != len(request.tmd_codes_batch):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of sequences must match number of TMD code sets"
            )

        start_time = time.time()
        predictions = []

        for seq, tmd_codes in zip(request.sequences, request.tmd_codes_batch):
            if request.use_mock or not state.loaded:
                pred = mock_inference(seq, tmd_codes)
            else:
                pred = real_inference(seq, tmd_codes)
            predictions.append(pred.tolist())

        total_latency_ms = int((time.time() - start_time) * 1000)
        avg_latency_ms = total_latency_ms / len(predictions) if predictions else 0

        return BatchInferResponse(
            predictions=predictions,
            count=len(predictions),
            total_latency_ms=total_latency_ms,
            avg_latency_ms=avg_latency_ms,
            is_mock=request.use_mock or not state.loaded
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def model_info_endpoint():
    """Get model metadata"""
    num_params = None
    if state.model is not None:
        num_params = state.model.get_num_params()

    return ModelInfoResponse(
        loaded=state.loaded,
        model_path=state.model_path,
        d_input=state.d_input,
        d_hidden=state.d_hidden,
        n_layers=state.n_layers,
        device=state.device,
        num_parameters=num_params
    )


@app.post("/model/load", tags=["model"])
async def load_model_endpoint(model_path: str, device: str = "cpu"):
    """Load model from checkpoint"""
    try:
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {model_path}"
            )

        # Create model
        model = LatentMamba(
            d_input=state.d_input,
            d_hidden=state.d_hidden,
            n_layers=state.n_layers
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Update state
        state.model = model
        state.device = device
        state.model_path = model_path
        state.loaded = True

        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "num_parameters": model.get_num_params(),
            "device": device
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading failed: {str(e)}"
        )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    print("🚀 LVM Inference API starting...")

    # Try to auto-load model if path is set
    model_path = os.getenv("LNSP_LVM_MODEL_PATH")
    if model_path and os.path.exists(model_path):
        try:
            device = os.getenv("LNSP_LVM_DEVICE", "cpu")
            print(f"   Auto-loading model from: {model_path}")

            model = LatentMamba(
                d_input=state.d_input,
                d_hidden=state.d_hidden,
                n_layers=state.n_layers
            )
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()

            state.model = model
            state.device = device
            state.model_path = model_path
            state.loaded = True

            print(f"   ✅ Model loaded ({model.get_num_params():,} parameters)")
        except Exception as e:
            print(f"   ⚠️  Model loading failed: {e}")
            print("   ⚠️  Running in mock mode")
    else:
        print("   ⚠️  No model path set (LNSP_LVM_MODEL_PATH)")
        print("   ⚠️  Running in mock mode")

    print(f"   Device: {state.device}")
    print("✅ LVM Inference API ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("👋 LVM Inference API shutting down...")
    state.model = None
    state.loaded = False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003,
        log_level="info"
    )
