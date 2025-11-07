"""
Google Gemini Cloud Adapter (Port 8102)

Wrapper for Google Gemini API (Gemini 2.5 Pro, Gemini 2.5 Flash)
Provides OpenAI-compatible endpoints and auto-registers with Provider Router
"""

import uvicorn
import time
from typing import Optional
import google.generativeai as genai

import sys
sys.path.insert(0, '/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4')

from services.cloud_providers.common.base_adapter import BaseCloudAdapter
from services.cloud_providers.common.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatRole,
    Usage
)


class GeminiAdapter(BaseCloudAdapter):
    """Google Gemini cloud provider adapter"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize Gemini adapter

        Args:
            model: Gemini model name (default: "gemini-2.5-pro")
            api_base_url: Optional custom API base URL (not used for Gemini)
        """
        # Model-specific configuration
        model_configs = {
            "gemini-2.5-pro": {
                "context_window": 2000000,
                "cost_input": 0.000010,
                "cost_output": 0.000030,
                "capabilities": ["planning", "code_write", "multimodal", "long_context"]
            },
            "gemini-2.5-flash": {
                "context_window": 1000000,
                "cost_input": 0.000001,
                "cost_output": 0.000003,
                "capabilities": ["fast_tasks", "code_write", "multimodal", "long_context"]
            },
            "gemini-2.5-flash-lite": {
                "context_window": 1000000,
                "cost_input": 0.0000005,
                "cost_output": 0.0000015,
                "capabilities": ["fast_tasks", "classification", "extraction"]
            }
        }

        config = model_configs.get(model, model_configs["gemini-2.5-pro"])

        super().__init__(
            provider_name="gemini",
            service_name=f"Google Gemini {model} Adapter",
            model=model,
            port=8102,
            api_key_env_var="GEMINI_API_KEY",
            capabilities=config["capabilities"],
            context_window=config["context_window"],
            cost_per_input_token=config["cost_input"],
            cost_per_output_token=config["cost_output"],
            version="1.0.0",
            api_base_url=api_base_url
        )

        # Initialize Gemini client
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)

    async def _call_provider_api(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Call Gemini API

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Convert Pydantic request to Gemini format
        # Gemini uses "parts" instead of "content"
        history = []
        user_prompt = ""

        for msg in request.messages:
            if msg.role == ChatRole.SYSTEM:
                # Gemini doesn't have system role - prepend to first user message
                user_prompt = f"{msg.content}\n\n"
            elif msg.role == ChatRole.USER:
                user_prompt += msg.content
            elif msg.role == ChatRole.ASSISTANT:
                history.append({
                    "role": "model",
                    "parts": [msg.content]
                })

        # Generate configuration
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens or 8192,
            stop_sequences=request.stop or []
        )

        # Call Gemini API (synchronous for now)
        # Note: google-generativeai doesn't have proper async support yet
        import asyncio
        response = await asyncio.to_thread(
            self.client.generate_content,
            user_prompt,
            generation_config=generation_config
        )

        # Estimate token usage (Gemini doesn't always provide this)
        # Use heuristic: ~4 chars per token
        prompt_tokens = sum(len(msg.content) for msg in request.messages) // 4
        completion_tokens = len(response.text) // 4 if response.text else 0

        # Convert Gemini response to OpenAI-compatible format
        return ChatCompletionResponse(
            id=f"gemini-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=response.text or ""
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    async def startup(self):
        """Service startup"""
        await super().startup()
        print(f"✅ Gemini client initialized for model: {self.model}")


# Create service instance
service = GeminiAdapter(
    model="gemini-2.5-pro"  # Default model
)
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting Google Gemini Adapter (Port 8102)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("Gemini Adapter Ready")
    print(f"  URL: http://localhost:8102")
    print(f"  Model: {service.model}")
    print(f"  Capabilities: {', '.join(service.capabilities)}")
    print("  OpenAPI Docs: http://localhost:8102/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down Gemini Adapter...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.cloud_providers.gemini.gemini_adapter:app",
        host="127.0.0.1",
        port=8102,
        log_level="info"
    )
