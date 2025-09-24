"""
Local Llama client implementation for LNSP local-only LLM policy.
Enforces local provider with no cloud fallback.
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaResponse:
    text: str
    latency_ms: int
    bytes_in: int
    bytes_out: int
    provider: str = "local_llama"
    model: str = ""


def _get_local_endpoint() -> str:
    """Get local LLM endpoint from environment."""
    endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://127.0.0.1:11434")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _get_model() -> str:
    """Get local LLM model from environment."""
    return os.getenv("LNSP_LLM_MODEL", "llama3.1:8b-instruct")


def _validate_local_policy() -> None:
    """Validate that local-only policy is enforced."""
    if os.getenv("LNSP_ALLOW_MOCK", "0") == "1":
        raise RuntimeError("Local Llama policy violation: LNSP_ALLOW_MOCK=1 (mock fallback disabled)")

    provider = os.getenv("LNSP_LLM_PROVIDER", "")
    if provider and provider != "local_llama":
        raise RuntimeError(f"Local Llama policy violation: LNSP_LLM_PROVIDER={provider} (must be 'local_llama')")


def test_connectivity(timeout: float = 5.0) -> bool:
    """Test connectivity to local LLM endpoint."""
    try:
        endpoint = _get_local_endpoint()
        # Try Ollama health check endpoint
        response = requests.get(f"{endpoint}/api/version", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def call_local_llama(prompt: str, system_prompt: Optional[str] = None) -> LlamaResponse:
    """
    Call local Llama model with metrics tracking.

    Args:
        prompt: User prompt text
        system_prompt: Optional system prompt

    Returns:
        LlamaResponse with text and metrics

    Raises:
        RuntimeError: If policy violations or empty response
        requests.RequestException: If network/API errors
    """
    _validate_local_policy()

    endpoint = _get_local_endpoint()
    model = _get_model()

    # Build request
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    # Calculate request size
    request_json = requests.models.RequestEncodingMixin._encode_json(request_body)
    bytes_in = len(request_json.encode('utf-8'))

    # Make request with timing
    start_time = time.time()
    try:
        response = requests.post(
            f"{endpoint}/api/chat",
            json=request_body,
            timeout=60.0
        )
        response.raise_for_status()
    except requests.RequestException as e:
        latency_ms = int((time.time() - start_time) * 1000)
        raise RuntimeError(f"Local Llama endpoint failed: {e} (latency: {latency_ms}ms)")

    latency_ms = int((time.time() - start_time) * 1000)

    # Parse response
    response_data = response.json()
    text = response_data.get("message", {}).get("content", "")

    # Calculate response size
    bytes_out = len(response.content)

    # Validate response
    if not text or not text.strip():
        raise RuntimeError(
            f"Local Llama returned empty text (model: {model}, "
            f"endpoint: {endpoint}, latency: {latency_ms}ms)"
        )

    return LlamaResponse(
        text=text.strip(),
        latency_ms=latency_ms,
        bytes_in=bytes_in,
        bytes_out=bytes_out,
        provider="local_llama",
        model=model
    )


def call_local_llama_simple(prompt: str) -> str:
    """Simple interface returning just the text response."""
    response = call_local_llama(prompt)
    return response.text