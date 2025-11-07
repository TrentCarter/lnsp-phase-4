"""
Ollama API client for LLM services
Handles communication with Ollama backend
"""

import httpx
import json
import time
from typing import Dict, List, Optional, Any
from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    GenerateRequest, GenerateResponse,
    ChatMessage, ChatChoice, Usage
)


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        """
        Initialize Ollama client

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 120s)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def health_check(self) -> bool:
        """
        Check if Ollama server is healthy

        Returns:
            True if server is responding, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """
        List available models in Ollama

        Returns:
            List of model names
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []

    async def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is available in Ollama

        Args:
            model_name: Model identifier (e.g., "llama3.1:8b")

        Returns:
            True if model is available, False otherwise
        """
        models = await self.list_models()
        return model_name in models

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate completion using Ollama /api/generate endpoint

        Args:
            request: Generate request

        Returns:
            Generate response

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        payload = request.model_dump(exclude_none=True)

        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        return GenerateResponse(**data)

    async def chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate chat completion (converts OpenAI format to Ollama, then back)

        Args:
            request: Chat completion request (OpenAI format)

        Returns:
            Chat completion response (OpenAI format)

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        # Convert OpenAI-style request to Ollama format
        ollama_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        ollama_payload = {
            "model": request.model,
            "messages": ollama_messages,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
        }

        # Add optional parameters
        if request.max_tokens:
            ollama_payload["options"]["num_predict"] = request.max_tokens

        if request.stop:
            ollama_payload["options"]["stop"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )

        # Call Ollama /api/chat endpoint
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=ollama_payload
        )
        response.raise_for_status()
        data = response.json()

        # Convert Ollama response to OpenAI format
        message_content = data.get("message", {}).get("content", "")
        finish_reason = "stop" if data.get("done", True) else "length"

        # Extract token counts (Ollama provides these in different fields)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ChatCompletionResponse(
            id=f"chat-{int(time.time() * 1000)}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=message_content
                    ),
                    finish_reason=finish_reason
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    async def extract_with_prompt(
        self,
        model: str,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.0,
        format_json: bool = True
    ) -> Dict[str, Any]:
        """
        Helper method for structured extraction tasks

        Args:
            model: Model name
            system_prompt: System instructions
            user_query: User query
            temperature: Sampling temperature (default: 0 for deterministic)
            format_json: Request JSON output format

        Returns:
            Parsed JSON response
        """
        # Import here to avoid circular dependency
        from .schemas import GenerateOptions as SchemaGenerateOptions

        # Build options if needed
        options = None
        if temperature != 0.7:
            options = SchemaGenerateOptions(temperature=temperature)

        request = GenerateRequest(
            model=model,
            prompt=user_query,
            system=system_prompt,
            stream=False,
            format="json" if format_json else None,
            options=options
        )

        response = await self.generate(request)

        # Try to parse JSON response
        try:
            return json.loads(response.response)
        except json.JSONDecodeError:
            # If not valid JSON, return raw text
            return {"raw_text": response.response}
