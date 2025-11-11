#!/usr/bin/env python3
"""
Model Service Template — Ports 8051-8099
OpenAI-compatible FastAPI wrapper for individual Ollama models.

Each model gets its own FastAPI service on a dedicated port:
- 8051: qwen2.5-coder:7b
- 8052: llama3.1:8b
- 8053+: Dynamic allocation

Provides:
- /v1/chat/completions (OpenAI-compatible)
- /v1/completions (OpenAI-compatible)
- /health (health check)
- Automatic TTL extension on each request

Author: Trent Carter
Date: 2025-11-11
"""

import argparse
import httpx
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ============================================================================
# Data Models (OpenAI-Compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # system, user, assistant
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ============================================================================
# Model Service
# ============================================================================

class ModelService:
    """FastAPI service wrapper for Ollama model"""

    def __init__(self, model_id: str, port: int, ollama_url: str, pool_manager_url: str):
        self.model_id = model_id
        self.port = port
        self.ollama_url = ollama_url
        self.pool_manager_url = pool_manager_url
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.request_count = 0
        self.start_time = time.time()

    async def extend_ttl(self):
        """Notify pool manager of activity (extends TTL)"""
        try:
            await self.http_client.post(
                f"{self.pool_manager_url}/models/{self.model_id}/extend-ttl",
                json={"minutes": 15}
            )
        except Exception as e:
            print(f"⚠️  Could not extend TTL: {e}")

    async def chat_completion(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request via Ollama"""
        self.request_count += 1
        await self.extend_ttl()

        # Convert OpenAI format to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in req.messages
        ]

        ollama_req = {
            "model": self.model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
                "top_p": req.top_p,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty
            }
        }

        # Call Ollama API
        try:
            resp = await self.http_client.post(
                f"{self.ollama_url}/api/chat",
                json=ollama_req
            )
            resp.raise_for_status()
            ollama_resp = resp.json()
        except Exception as e:
            raise HTTPException(500, f"Ollama API error: {e}")

        # Convert Ollama response to OpenAI format
        message = ollama_resp.get("message", {})
        content = message.get("content", "")

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=self.model_id,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                "completion_tokens": ollama_resp.get("eval_count", 0),
                "total_tokens": (
                    ollama_resp.get("prompt_eval_count", 0) +
                    ollama_resp.get("eval_count", 0)
                )
            }
        )

    async def completion(self, req: CompletionRequest) -> CompletionResponse:
        """Handle completion request via Ollama"""
        self.request_count += 1
        await self.extend_ttl()

        ollama_req = {
            "model": self.model_id,
            "prompt": req.prompt,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
                "top_p": req.top_p,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty
            }
        }

        # Call Ollama API
        try:
            resp = await self.http_client.post(
                f"{self.ollama_url}/api/generate",
                json=ollama_req
            )
            resp.raise_for_status()
            ollama_resp = resp.json()
        except Exception as e:
            raise HTTPException(500, f"Ollama API error: {e}")

        # Convert Ollama response to OpenAI format
        text = ollama_resp.get("response", "")

        return CompletionResponse(
            id=f"cmpl-{int(time.time())}",
            created=int(time.time()),
            model=self.model_id,
            choices=[{
                "index": 0,
                "text": text,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": ollama_resp.get("prompt_eval_count", 0),
                "completion_tokens": ollama_resp.get("eval_count", 0),
                "total_tokens": (
                    ollama_resp.get("prompt_eval_count", 0) +
                    ollama_resp.get("eval_count", 0)
                )
            }
        )

    def get_health(self) -> dict:
        """Get service health status"""
        uptime_seconds = int(time.time() - self.start_time)
        return {
            "model": self.model_id,
            "status": "ready",
            "uptime_seconds": uptime_seconds,
            "requests_served": self.request_count,
            "port": self.port
        }


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app(model_id: str, port: int, ollama_url: str, pool_manager_url: str) -> FastAPI:
    """Create FastAPI app for model service"""

    app = FastAPI(
        title=f"Model Service: {model_id}",
        description=f"OpenAI-compatible API for {model_id}",
        version="1.0.0"
    )

    service = ModelService(model_id, port, ollama_url, pool_manager_url)

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return service.get_health()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint"""
        return await service.chat_completion(req)

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest):
        """OpenAI-compatible completions endpoint"""
        return await service.completion(req)

    return app


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model Service Template")
    parser.add_argument("--model-id", required=True, help="Model ID (e.g., qwen2.5-coder:7b)")
    parser.add_argument("--port", type=int, required=True, help="Port to run on")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--pool-manager-url", default="http://localhost:8050", help="Pool manager URL")

    args = parser.parse_args()

    print(f"🚀 Starting model service: {args.model_id}")
    print(f"📍 Port: {args.port}")
    print(f"🔗 Ollama: {args.ollama_url}")
    print(f"🔗 Pool Manager: {args.pool_manager_url}")

    app = create_app(args.model_id, args.port, args.ollama_url, args.pool_manager_url)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
