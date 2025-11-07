"""
Anthropic Cloud Adapter (Port 8101)

Wrapper for Anthropic API (Claude Sonnet 4.5, Claude Haiku 4.5)
Provides OpenAI-compatible endpoints and auto-registers with Provider Router
"""

import uvicorn
import time
from typing import Optional
from anthropic import AsyncAnthropic

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


class AnthropicAdapter(BaseCloudAdapter):
    """Anthropic cloud provider adapter"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize Anthropic adapter

        Args:
            model: Anthropic model name (default: "claude-sonnet-4-5-20250929")
            api_base_url: Optional custom API base URL
        """
        # Model-specific configuration
        model_configs = {
            "claude-sonnet-4-5-20250929": {
                "context_window": 200000,
                "cost_input": 0.000003,
                "cost_output": 0.000015,
                "capabilities": ["planning", "code_write", "reasoning", "long_context"]
            },
            "claude-haiku-4-5": {
                "context_window": 100000,
                "cost_input": 0.00000025,
                "cost_output": 0.00000125,
                "capabilities": ["classification", "extraction", "fast_tasks", "long_context"]
            }
        }

        config = model_configs.get(model, model_configs["claude-sonnet-4-5-20250929"])

        super().__init__(
            provider_name="anthropic",
            service_name=f"Anthropic {model} Adapter",
            model=model,
            port=8101,
            api_key_env_var="ANTHROPIC_API_KEY",
            capabilities=config["capabilities"],
            context_window=config["context_window"],
            cost_per_input_token=config["cost_input"],
            cost_per_output_token=config["cost_output"],
            version="1.0.0",
            api_base_url=api_base_url
        )

        # Initialize Anthropic client
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=api_base_url
        )

    async def _call_provider_api(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Call Anthropic API

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Convert Pydantic request to Anthropic format
        # Anthropic requires system message to be separate
        system_message = None
        messages = []

        for msg in request.messages:
            if msg.role == ChatRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        # Call Anthropic API
        response = await self.client.messages.create(
            model=request.model,
            system=system_message,
            messages=messages,
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop
        )

        # Convert Anthropic response to OpenAI-compatible format
        return ChatCompletionResponse(
            id=response.id,
            object="chat.completion",
            created=int(time.time()),
            model=response.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=ChatRole.ASSISTANT,
                        content=response.content[0].text if response.content else ""
                    ),
                    finish_reason=response.stop_reason or "stop"
                )
            ],
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        )

    async def startup(self):
        """Service startup"""
        await super().startup()
        print(f"✅ Anthropic client initialized for model: {self.model}")


# Create service instance
service = AnthropicAdapter(
    model="claude-sonnet-4-5-20250929"  # Default model
)
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting Anthropic Adapter (Port 8101)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("Anthropic Adapter Ready")
    print(f"  URL: http://localhost:8101")
    print(f"  Model: {service.model}")
    print(f"  Capabilities: {', '.join(service.capabilities)}")
    print("  OpenAPI Docs: http://localhost:8101/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown")
    print("\nShutting down Anthropic Adapter...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.cloud_providers.anthropic.anthropic_adapter:app",
        host="127.0.0.1",
        port=8101,
        log_level="info"
    )
