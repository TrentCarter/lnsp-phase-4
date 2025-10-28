"""
FastAPI LVM Inference Service with Chat Interface

Provides REST API and web chat interface for Latent Vector Models (AMN, GRU, LSTM, Transformer, Mamba).
Tokenless inference: Text → 768D vectors → LVM → 768D prediction → Text

Port Assignment:
- 9001: AMN (Attention Mixer Network)
- 9002: Transformer (Optimized)
- 9003: GRU
- 9004: LSTM
- 9005: Mamba (future)
- 9006-9999: Reserved for future models

Usage:
    uvicorn app.api.lvm_inference:app --host 127.0.0.1 --port 9001 --reload
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

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


config = LVMConfig()
model = None


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
    global model

    # Startup: Load model
    print("="*60)
    print("LVM INFERENCE SERVICE STARTUP")
    print("="*60)

    model = load_lvm_model(
        model_type=config.model_type,
        model_path=config.model_path,
        device=config.device
    )

    print("\n✅ Service ready for inference")
    print(f"   Model: {config.model_type.upper()}")
    print(f"   Context: {config.context_length} vectors")
    print(f"   Encoder: {config.encoder_url}")
    print(f"   Decoder: {config.decoder_url}")
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
    messages: List[str] = Field(
        ...,
        description="Previous messages (context)",
        min_items=1,
        max_items=config.context_length
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    decode_steps: int = Field(1, ge=1, le=10, description="Vec2text decoding steps (1=fast, 2=better quality)")


class ChatResponse(BaseModel):
    """Chat-style inference response"""
    response: str
    confidence: float
    latency_breakdown: Dict[str, float]
    total_latency_ms: float


# ============================================================================
# Helper Functions
# ============================================================================

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
    """Health check"""
    return {
        "status": "healthy",
        "model_type": config.model_type,
        "model_loaded": model is not None,
        "device": config.device,
        "encoder_url": config.encoder_url,
        "decoder_url": config.decoder_url
    }


@app.get("/info")
async def info():
    """Model information"""
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
        "memory_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Low-level inference: predict next vector from context vectors.

    This endpoint operates directly on vectors (no text encoding/decoding).
    """
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

    Takes previous messages as context, predicts next message.
    Full tokenless pipeline with encoding and decoding.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_start = time.perf_counter()
    latency_breakdown = {}

    # Step 1: Encode messages to vectors
    encode_start = time.perf_counter()
    try:
        context_vectors = await encode_texts(request.messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Encoding failed: {str(e)}")

    latency_breakdown["encoding_ms"] = (time.perf_counter() - encode_start) * 1000

    # Pad context if needed
    if len(context_vectors) < config.context_length:
        padding = np.zeros((config.context_length - len(context_vectors), config.vector_dim))
        context_vectors = np.vstack([padding, context_vectors])
    elif len(context_vectors) > config.context_length:
        context_vectors = context_vectors[-config.context_length:]

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

    # Confidence
    confidence = float(1.0 - torch.abs(prediction).mean())

    latency_breakdown["lvm_inference_ms"] = (time.perf_counter() - lvm_start) * 1000

    # Step 3: Decode vector to text
    decode_start = time.perf_counter()
    try:
        decode_result = await decode_vector(
            prediction[0].cpu().numpy(),
            steps=request.decode_steps
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Decoding failed: {str(e)}")

    latency_breakdown["decoding_ms"] = decode_result["latency_ms"]

    total_latency = (time.perf_counter() - total_start) * 1000

    return ChatResponse(
        response=decode_result["text"],
        confidence=confidence,
        latency_breakdown=latency_breakdown,
        total_latency_ms=total_latency
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """
    Beautiful chat web interface (like Claude/GPT).
    Tokenless LVM chat with real-time inference.
    """
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{config.model_type.upper()} Chat - Tokenless LVM</title>
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
                <h1>{config.model_type.upper()} - Tokenless LVM</h1>
                <div class="subtitle">Context: {config.context_length} messages | Device: {config.device.upper()}</div>
            </div>

            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        <div>👋 Hello! I'm a tokenless latent vector model.</div>
                        <div style="margin-top: 8px;">I operate in 768D semantic space without tokens. Type a message to chat!</div>
                        <div class="message-meta">Model: {config.model_type.upper()} | Ready for inference</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
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

                // Show loading
                addLoadingIndicator();

                try {{
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            messages: conversationHistory,
                            temperature: 1.0,
                            decode_steps: 1
                        }})
                    }});

                    if (!response.ok) {{
                        throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                    }}

                    const data = await response.json();

                    removeLoadingIndicator();

                    // Add assistant response
                    const meta = `Latency: ${{data.total_latency_ms.toFixed(0)}}ms | LVM: ${{data.latency_breakdown.lvm_inference_ms.toFixed(1)}}ms | Confidence: ${{(data.confidence * 100).toFixed(1)}}%`;
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
