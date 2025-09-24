"""
LLM client router for LNSP.
Routes to local_llama when LNSP_LLM_PROVIDER=local_llama.
"""

import os
from typing import Optional
from .local_llama_client import LlamaResponse, call_local_llama, test_connectivity


def get_llm_client():
    """Get configured LLM client based on provider setting."""
    provider = os.getenv("LNSP_LLM_PROVIDER", "")

    if provider == "local_llama":
        return LocalLlamaClient()
    else:
        # For backward compatibility, fall back to existing llm_bridge
        # but this should be explicitly set to "local_llama" in production
        from ..llm_bridge import annotate_with_llm
        return LegacyClient()


class LocalLlamaClient:
    """Local Llama client enforcing local-only policy."""

    def __init__(self):
        self.provider = "local_llama"

    def test_connection(self) -> bool:
        """Test connection to local LLM endpoint."""
        return test_connectivity()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LlamaResponse:
        """Generate response using local Llama."""
        return call_local_llama(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if local Llama is available."""
        return self.test_connection()


class LegacyClient:
    """Legacy client for backward compatibility."""

    def __init__(self):
        self.provider = "legacy"

    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate using legacy llm_bridge."""
        from ..llm_bridge import annotate_with_llm
        # This is a simplified interface - real usage would need proper parameters
        return {"text": "Legacy interface - use LocalLlamaClient for new code"}

    def is_available(self) -> bool:
        """Legacy client availability."""
        return True