"""
xAI Grok Cloud Adapter (Port 8103)

Wrapper for xAI Grok API
Provides OpenAI-compatible endpoints and auto-registers with Provider Router

Note: xAI Grok API uses OpenAI-compatible format, so we use the OpenAI client
"""

import uvicorn
import time
from typing import Optional
from openai import AsyncOpenAI

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


class GrokAdapter(BaseCloudAdapter):
    """xAI Grok cloud provider adapter"""

    def __init__(
        self,
        model: str = "grok-beta",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize Grok adapter

        Args:
            model: Grok model name (default: "grok-beta")
            api_base_url: xAI API base URL (default: "https://api.x.ai/v1")
        """
        # Model-specific configuration
        model_configs = {
            "grok-beta": {
                "context_window": 128000,
                "cost_input": 0.000005,  # Estimated, adjust based on actual pricing
                "cost_output": 0.000015,
                "capabilities": ["planning", "reasoning", "real_time", "function_calling"]
            },
            "grok-1": {
                "context_window": 128000,
                "cost_input": 0.000005,
                "cost_output": 0.000015,
                "capabilities": ["planning", "reasoning", "real_time"]
            }
        }

        config = model_configs.get(model, model_configs["grok-beta"])

        # Default xAI API base URL
        if not api_base_url:
            api_base_url = "https://api.x.ai/v1"

        super().__init__(
            provider_name="grok",
            service_name=f"xAI {model} Adapter",
            model=model,
            port=8103,
            api_key_env_var="GROK_API_KEY",
            capabilities=config["capabilities"],
            context_window=config["context_window"],
            cost_per_input_token=config["cost_input"],
            cost_per_output_token=config["cost_output"],
            version="1.0.0",
            api_base_url=api_base_url
        )

        # Initialize xAI client (OpenAI-compatible)
        # Note: xAI uses OpenAI-compatible API format
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base_url
        )

    async def _call_provider_api(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Call xAI Grok API

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Convert Pydantic request to xAI format
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        # Call xAI API (OpenAI-compatible)
        response = await self.client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=False,
            stop=request.stop
        )

        # Convert xAI response to our schema
        return ChatCompletionResponse(
            id=response.id,
            object="chat.completion",
            created=response.created,
            model=response.model,
            choices=[
                ChatChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=choice.message.content or ""
                    ),
                    finish_reason=choice.finish_reason or "stop"
                )
                for choice in response.choices
            ],
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            ),
            system_fingerprint=getattr(response, 'system_fingerprint', None)
        )

    async def startup(self):
        """Service startup"""
        await super().startup()
        print(f"✅ Grok client initialized for model: {self.model}")


# Create service instance
service = GrokAdapter(
    model="grok-beta"  # Default model
)
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting xAI Grok Adapter (Port 8103)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("Grok Adapter Ready")
    print(f"  URL: http://localhost:8103")
    print(f"  Model: {service.model}")
    print(f"  Capabilities: {', '.join(service.capabilities)}")
    print("  OpenAPI Docs: http://localhost:8103/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down Grok Adapter...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.cloud_providers.grok.grok_adapter:app",
        host="127.0.0.1",
        port=8103,
        log_level="info"
    )
