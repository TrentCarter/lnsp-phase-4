"""
OpenAI Cloud Adapter (Port 8100)

Wrapper for OpenAI API (GPT-4, GPT-5, Codex)
Provides OpenAI-compatible endpoints and auto-registers with Provider Router
"""

import uvicorn
import time
from typing import Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

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


class OpenAIAdapter(BaseCloudAdapter):
    """OpenAI cloud provider adapter"""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI adapter

        Args:
            model: OpenAI model name (default: "gpt-4-turbo")
            api_base_url: Optional custom API base URL
        """
        # Model-specific configuration
        model_configs = {
            "gpt-5-codex": {
                "context_window": 200000,
                "cost_input": 0.000003,
                "cost_output": 0.000015,
                "capabilities": ["planning", "code_write", "reasoning", "function_calling"]
            },
            "gpt-4-turbo": {
                "context_window": 128000,
                "cost_input": 0.000010,
                "cost_output": 0.000030,
                "capabilities": ["planning", "reasoning", "vision", "function_calling"]
            },
            "gpt-3.5-turbo": {
                "context_window": 16385,
                "cost_input": 0.0000005,
                "cost_output": 0.0000015,
                "capabilities": ["chat", "completion", "fast_tasks"]
            }
        }

        config = model_configs.get(model, model_configs["gpt-4-turbo"])

        super().__init__(
            provider_name="openai",
            service_name=f"OpenAI {model} Adapter",
            model=model,
            port=8100,
            api_key_env_var="OPENAI_API_KEY",
            capabilities=config["capabilities"],
            context_window=config["context_window"],
            cost_per_input_token=config["cost_input"],
            cost_per_output_token=config["cost_output"],
            version="1.0.0",
            api_base_url=api_base_url
        )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base_url
        )

    async def _call_provider_api(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Call OpenAI API

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        # Convert Pydantic request to OpenAI format
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        # Call OpenAI API
        response: ChatCompletion = await self.client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stream=False,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )

        # Convert OpenAI response to our schema
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
            system_fingerprint=response.system_fingerprint
        )

    async def startup(self):
        """Service startup"""
        await super().startup()
        print(f"✅ OpenAI client initialized for model: {self.model}")


# Create service instance
service = OpenAIAdapter(
    model="gpt-4-turbo"  # Default model, can be changed via env var
)
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting OpenAI Adapter (Port 8100)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("OpenAI Adapter Ready")
    print(f"  URL: http://localhost:8100")
    print(f"  Model: {service.model}")
    print(f"  Capabilities: {', '.join(service.capabilities)}")
    print("  OpenAPI Docs: http://localhost:8100/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down OpenAI Adapter...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.cloud_providers.openai.openai_adapter:app",
        host="127.0.0.1",
        port=8100,
        log_level="info"
    )
