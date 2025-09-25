"""
Local Llama client implementation for LNSP local-only LLM policy.
Enforces local provider with no cloud fallback.
"""

import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any

_CPESH_TIMEOUT_ENV = "LNSP_CPESH_TIMEOUT_S"


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
    endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _get_model() -> str:
    """Get local LLM model from environment."""
    return os.getenv("LNSP_LLM_MODEL", "llama3.1:8b")


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
    # Combine system prompt and user prompt
    full_prompt = ""
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    request_body = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    # Calculate request size
    request_json = json.dumps(request_body)
    bytes_in = len(request_json.encode('utf-8'))

    # Make request with timing
    start_time = time.time()
    try:
        response = requests.post(
            f"{endpoint}/api/generate",
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
    text = response_data.get("response", "")

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


class LocalLlamaClient:
    """Client class for local Llama API with JSON completion support."""

    def __init__(self, endpoint: str = None, model: str = None):
        """
        Initialize the local Llama client.

        Args:
            endpoint: Ollama API endpoint (default: LNSP_LLM_ENDPOINT or http://127.0.0.1:11434)
            model: Model to use (default: LNSP_LLM_MODEL or llama3.1:8b-instruct)
        """
        self.endpoint = endpoint or _get_local_endpoint()
        self.model = model or _get_model()
        self._session = requests.Session()

    def _resolve_timeout(self, timeout_s: Optional[float]) -> float:
        if timeout_s and timeout_s > 0:
            return float(timeout_s)
        try:
            env_val = float(os.getenv(_CPESH_TIMEOUT_ENV, "12"))
            return env_val if env_val > 0 else 12.0
        except (TypeError, ValueError):
            return 12.0

    def complete_json(self, prompt: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"num_predict": 256, "temperature": 0},
            "stream": False,
            "format": "json",
        }

        resolved_timeout = self._resolve_timeout(timeout_s)

        try:
            response = self._session.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=resolved_timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Local Llama request timed out after {resolved_timeout}s"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Local Llama request failed: {exc}") from exc

        data = response.json()
        text = data.get("response", "")

        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                snippet = text[start:end]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass

        return {
            "concept": "",
            "probe": prompt[:100],
            "expected": "",
            "soft_negative": "",
            "hard_negative": "",
            "insufficient_evidence": True,
            "error": "Local Llama returned non-JSON payload",
        }
