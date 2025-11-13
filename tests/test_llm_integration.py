"""
Comprehensive LLM Integration Tests
Tests both local (Ollama) and API-based LLMs (Kimi, Anthropic, Google, OpenAI)

Usage:
    # Test all providers
    pytest tests/test_llm_integration.py -v

    # Test specific provider
    pytest tests/test_llm_integration.py -v -k "test_ollama"
    pytest tests/test_llm_integration.py -v -k "test_kimi"

    # Test with output
    pytest tests/test_llm_integration.py -v -s
"""

import pytest
import requests
import json
import time
import os
from typing import Dict, List, Any


# Test configuration
GATEWAY_URL = "http://localhost:6120"
TEST_TIMEOUT = 30  # seconds


class TestLLMIntegration:
    """Integration tests for all LLM providers"""

    @pytest.fixture(autouse=True)
    def check_gateway(self):
        """Verify Gateway is running before each test"""
        try:
            response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
            assert response.status_code == 200, "Gateway is not healthy"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Gateway not running at {GATEWAY_URL}: {e}")

    def _stream_chat(self, model: str, content: str, expected_status: int = 200) -> Dict[str, Any]:
        """
        Helper method to send chat request and parse streaming response

        Returns:
            Dict with keys: status_code, events, error, response_text, usage
        """
        session_id = f"test-{model.replace('/', '-')}-{int(time.time())}"

        request_data = {
            "session_id": session_id,
            "message_id": f"msg-{int(time.time())}",
            "agent_id": "test-agent",
            "model": model,
            "content": content
        }

        result = {
            "status_code": None,
            "events": [],
            "error": None,
            "response_text": "",
            "usage": None,
            "request": request_data
        }

        try:
            response = requests.post(
                f"{GATEWAY_URL}/chat/stream",
                json=request_data,
                stream=True,
                timeout=TEST_TIMEOUT
            )

            result["status_code"] = response.status_code

            if response.status_code != expected_status:
                result["error"] = f"Expected status {expected_status}, got {response.status_code}"
                return result

            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                try:
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    result["events"].append(event_data)

                    # Accumulate response text
                    if event_data.get("type") == "token":
                        result["response_text"] += event_data.get("content", "")

                    # Capture usage data
                    if event_data.get("type") == "usage":
                        result["usage"] = event_data.get("usage")

                    # Capture error
                    if event_data.get("type") == "error":
                        result["error"] = event_data.get("message")

                    # Check for status_update errors
                    if event_data.get("type") == "status_update" and event_data.get("status") == "error":
                        result["error"] = event_data.get("detail")

                except json.JSONDecodeError as e:
                    result["error"] = f"Failed to parse event: {line} - {e}"

        except requests.exceptions.Timeout:
            result["error"] = f"Request timeout after {TEST_TIMEOUT}s"
        except requests.exceptions.RequestException as e:
            result["error"] = f"Request failed: {e}"

        return result

    # ========================================================================
    # LOCAL LLM TESTS (Ollama)
    # ========================================================================

    def test_ollama_service_available(self):
        """Test if Ollama service is accessible"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            assert response.status_code == 200, "Ollama not responding"
            models = response.json().get("models", [])
            assert len(models) > 0, "No Ollama models found"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Ollama not running: {e}")

    @pytest.mark.parametrize("model", [
        "ollama/qwen2.5-coder:7b-instruct",
        "ollama/deepseek-r1:7b-q4_k_m",
    ])
    def test_ollama_streaming(self, model):
        """Test Ollama streaming chat"""
        result = self._stream_chat(model, "Say 'hello' in 3 words")

        # Verify no errors
        assert result["error"] is None, f"Stream error: {result['error']}"

        # Verify response text was generated
        assert len(result["response_text"]) > 0, "No response text received"

        # Verify event sequence
        event_types = [e.get("type") for e in result["events"]]
        assert "status_update" in event_types, "No status updates"
        assert "token" in event_types, "No tokens streamed"
        assert "done" in event_types, "No done signal"

        print(f"\n[{model}] Response: {result['response_text'][:100]}")
        if result["usage"]:
            print(f"[{model}] Usage: {result['usage']}")

    # ========================================================================
    # API LLM TESTS (Kimi)
    # ========================================================================

    def test_kimi_api_key_configured(self):
        """Test if Kimi API key is configured"""
        api_key = os.getenv("KIMI_API_KEY")
        assert api_key is not None, "KIMI_API_KEY not set"
        assert not api_key.startswith("your_"), "KIMI_API_KEY is placeholder"
        assert len(api_key) > 20, "KIMI_API_KEY looks invalid"

    @pytest.mark.parametrize("model", [
        "kimi/kimi-k2-turbo-preview",  # Latest K2 model
        "kimi/moonshot-v1-8k",
        "kimi/moonshot-v1-32k",
        "kimi/moonshot-v1-128k",
    ])
    def test_kimi_streaming(self, model):
        """Test Kimi (Moonshot) streaming chat with all models"""
        # Skip if API key not configured
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            pytest.skip("Kimi API key not configured")

        result = self._stream_chat(model, "Say 'hello' in 3 words")

        # Verify no errors
        if result["error"]:
            if "401" in result["error"] or "Invalid Authentication" in result["error"]:
                pytest.skip(f"Kimi API key invalid: {result['error']}")
            else:
                pytest.fail(f"Stream error: {result['error']}")

        # Verify response
        assert len(result["response_text"]) > 0, "No response text received"

        # Verify usage data (may be 0 for some providers)
        # assert result["usage"] is not None, "No usage data received"

        print(f"\n[{model}] Response: {result['response_text']}")
        if result["usage"]:
            print(f"[{model}] Usage: {result['usage']}")

    # ========================================================================
    # API LLM TESTS (Anthropic)
    # ========================================================================

    def test_anthropic_api_key_configured(self):
        """Test if Anthropic API key is configured"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        assert api_key is not None, "ANTHROPIC_API_KEY not set"
        assert not api_key.startswith("your_"), "ANTHROPIC_API_KEY is placeholder"

    def test_anthropic_streaming(self):
        """Test Anthropic (Claude) streaming chat"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("your_"):
            pytest.skip("Anthropic API key not configured")

        result = self._stream_chat("anthropic/claude-3-5-sonnet-20241022", "Say 'hello' in 3 words")

        if result["error"]:
            if "401" in result["error"] or "authentication" in result["error"].lower():
                pytest.skip(f"Anthropic API key invalid: {result['error']}")
            else:
                pytest.fail(f"Stream error: {result['error']}")

        assert len(result["response_text"]) > 0, "No response text received"
        assert result["usage"] is not None, "No usage data received"

        print(f"\n[Anthropic] Response: {result['response_text']}")
        print(f"[Anthropic] Usage: {result['usage']}")

    # ========================================================================
    # API LLM TESTS (Google)
    # ========================================================================

    def test_google_api_key_configured(self):
        """Test if Google API key is configured"""
        api_key = os.getenv("GEMINI_API_KEY")
        assert api_key is not None, "GEMINI_API_KEY not set"
        assert not api_key.startswith("your_"), "GEMINI_API_KEY is placeholder"

    def test_google_streaming(self):
        """Test Google (Gemini) streaming chat"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            pytest.skip("Google API key not configured")

        result = self._stream_chat("google/gemini-2.5-flash", "Say 'hello' in 3 words")

        if result["error"]:
            if "401" in result["error"] or "API key" in result["error"]:
                pytest.skip(f"Google API key invalid: {result['error']}")
            else:
                pytest.fail(f"Stream error: {result['error']}")

        assert len(result["response_text"]) > 0, "No response text received"

        print(f"\n[Google] Response: {result['response_text']}")
        if result["usage"]:
            print(f"[Google] Usage: {result['usage']}")

    # ========================================================================
    # API LLM TESTS (OpenAI)
    # ========================================================================

    def test_openai_api_key_configured(self):
        """Test if OpenAI API key is configured"""
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY not set"
        assert not api_key.startswith("your_"), "OPENAI_API_KEY is placeholder"

    def test_openai_streaming(self):
        """Test OpenAI (GPT) streaming chat"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            pytest.skip("OpenAI API key not configured")

        result = self._stream_chat("openai/gpt-4o-mini", "Say 'hello' in 3 words")

        if result["error"]:
            if "401" in result["error"] or "API key" in result["error"]:
                pytest.skip(f"OpenAI API key invalid: {result['error']}")
            else:
                pytest.fail(f"Stream error: {result['error']}")

        assert len(result["response_text"]) > 0, "No response text received"
        assert result["usage"] is not None, "No usage data received"

        print(f"\n[OpenAI] Response: {result['response_text']}")
        print(f"[OpenAI] Usage: {result['usage']}")

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    def test_invalid_model_prefix(self):
        """Test handling of unknown provider prefix"""
        result = self._stream_chat("unknown/model", "test")

        # Should fallback to Ollama (based on gateway.py:416-421)
        # May fail if Ollama is down, but shouldn't crash
        assert result["status_code"] == 200, "Gateway should handle unknown providers"

    def test_empty_content(self):
        """Test handling of empty message content"""
        result = self._stream_chat("ollama/qwen2.5-coder:7b-instruct", "")

        # Should either succeed with empty response or gracefully error
        assert result["status_code"] == 200, "Gateway should handle empty content"

    def test_very_long_content(self):
        """Test handling of very long messages"""
        long_content = "test " * 1000  # ~5000 chars
        result = self._stream_chat("ollama/qwen2.5-coder:7b-instruct", long_content)

        assert result["status_code"] == 200, "Gateway should handle long content"


# ============================================================================
# STANDALONE TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick debugging
    """
    import sys

    print("=" * 80)
    print("LLM Integration Test Suite")
    print("=" * 80)

    test_suite = TestLLMIntegration()

    # Check Gateway
    print("\n[1/9] Checking Gateway...")
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        assert response.status_code == 200, "Gateway is not healthy"
        print("✓ Gateway is running")
    except Exception as e:
        print(f"✗ Gateway check failed: {e}")
        sys.exit(1)

    # Test Ollama
    print("\n[2/9] Testing Ollama service...")
    try:
        test_suite.test_ollama_service_available()
        print("✓ Ollama is running")
    except Exception as e:
        print(f"⊘ Ollama test skipped: {e}")

    print("\n[3/9] Testing Ollama streaming...")
    try:
        test_suite.test_ollama_streaming("ollama/qwen2.5-coder:7b-instruct")
        print("✓ Ollama streaming works")
    except Exception as e:
        print(f"✗ Ollama streaming failed: {e}")

    # Test Kimi
    print("\n[4/9] Testing Kimi API key...")
    kimi_key = os.getenv("KIMI_API_KEY")
    if not kimi_key or kimi_key.startswith("your_"):
        print("⊘ Kimi API key not configured - skipping tests")
        kimi_available = False
    else:
        print(f"✓ Kimi API key configured (length: {len(kimi_key)})")
        kimi_available = True

    if kimi_available:
        print("\n[5/9] Testing Kimi streaming...")
        try:
            result = test_suite._stream_chat("kimi/moonshot-v1-8k", "Say 'hello' in 3 words")
            if result["error"]:
                print(f"⊘ Kimi test failed: {result['error']}")
            else:
                print(f"✓ Kimi streaming works")
                print(f"[Kimi] Response: {result['response_text']}")
                print(f"[Kimi] Usage: {result['usage']}")
        except Exception as e:
            print(f"⊘ Kimi test failed: {e}")
    else:
        print("\n[5/9] Skipping Kimi streaming test (API key not configured)")

    # Test Anthropic
    print("\n[6/9] Testing Anthropic API key...")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key or anthropic_key.startswith("your_"):
        print("⊘ Anthropic API key not configured - skipping tests")
        anthropic_available = False
    else:
        print(f"✓ Anthropic API key configured (length: {len(anthropic_key)})")
        anthropic_available = True

    if anthropic_available:
        print("\n[7/9] Testing Anthropic streaming...")
        try:
            result = test_suite._stream_chat("anthropic/claude-3-5-sonnet-20241022", "Say 'hello' in 3 words")
            if result["error"]:
                print(f"⊘ Anthropic test failed: {result['error']}")
            else:
                print(f"✓ Anthropic streaming works")
                print(f"[Anthropic] Response: {result['response_text']}")
        except Exception as e:
            print(f"⊘ Anthropic test failed: {e}")
    else:
        print("\n[7/9] Skipping Anthropic streaming test (API key not configured)")

    # Test Google
    print("\n[8/9] Testing Google API key...")
    google_key = os.getenv("GEMINI_API_KEY")
    if not google_key or google_key.startswith("your_"):
        print("⊘ Google API key not configured - skipping tests")
        google_available = False
    else:
        print(f"✓ Google API key configured (length: {len(google_key)})")
        google_available = True

    if google_available:
        print("\n[9/9] Testing Google streaming...")
        try:
            result = test_suite._stream_chat("google/gemini-2.5-flash", "Say 'hello' in 3 words")
            if result["error"]:
                print(f"⊘ Google test failed: {result['error']}")
            else:
                print(f"✓ Google streaming works")
                print(f"[Google] Response: {result['response_text']}")
        except Exception as e:
            print(f"⊘ Google test failed: {e}")
    else:
        print("\n[9/9] Skipping Google streaming test (API key not configured)")

    print("\n" + "=" * 80)
    print("Test suite complete!")
    print("=" * 80)
