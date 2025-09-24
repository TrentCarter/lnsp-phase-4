"""Tests for local Llama LLM client."""

import pytest
import os
from unittest.mock import patch, MagicMock

from src.llm.local_llama_client import (
    LocalLlamaClient,
    call_local_llama,
    test_connectivity,
    _validate_local_policy
)


class TestLocalLlamaClient:
    """Test local Llama client functionality."""

    def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = LocalLlamaClient()
        assert client.endpoint == "http://127.0.0.1:11434"
        assert client.model == "llama3.1:8b-instruct"
        assert client.timeout == 120

    def test_custom_initialization(self):
        """Test client with custom parameters."""
        client = LocalLlamaClient(
            endpoint="http://localhost:8080",
            model="llama2:7b",
            timeout=60
        )
        assert client.endpoint == "http://localhost:8080"
        assert client.model == "llama2:7b"
        assert client.timeout == 60

    @patch('src.llm.local_llama_client.requests.post')
    def test_generate_success(self, mock_post):
        """Test successful generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = LocalLlamaClient()
        result = client.generate("Test prompt")

        assert result["text"] == "Test response"
        assert "latency_ms" in result
        assert "bytes_in" in result
        assert "bytes_out" in result
        assert result["model"] == "llama3.1:8b-instruct"
        assert result["provider"] == "local_llama"

    @patch('src.llm.local_llama_client.requests.post')
    def test_generate_empty_response(self, mock_post):
        """Test handling of empty responses."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = LocalLlamaClient()

        with pytest.raises(ValueError, match="returned empty response"):
            client.generate("Test prompt")

    @patch('src.llm.local_llama_client.requests.get')
    def test_connectivity_success(self, mock_get):
        """Test successful connectivity check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = test_connectivity()
        assert result is True

    @patch('src.llm.local_llama_client.requests.get')
    def test_connectivity_failure(self, mock_get):
        """Test failed connectivity check."""
        mock_get.side_effect = Exception("Connection failed")

        result = test_connectivity()
        assert result is False

    def test_policy_validation_mock_disabled(self):
        """Test policy validation with mock disabled."""
        with patch.dict(os.environ, {"LNSP_ALLOW_MOCK": "0"}):
            # Should not raise
            _validate_local_policy()

    def test_policy_validation_mock_enabled(self):
        """Test policy validation with mock enabled."""
        with patch.dict(os.environ, {"LNSP_ALLOW_MOCK": "1"}):
            with pytest.raises(RuntimeError, match="mock fallback disabled"):
                _validate_local_policy()

    def test_policy_validation_wrong_provider(self):
        """Test policy validation with wrong provider."""
        with patch.dict(os.environ, {
            "LNSP_ALLOW_MOCK": "0",
            "LNSP_LLM_PROVIDER": "openai"
        }):
            with pytest.raises(RuntimeError, match="must be 'local_llama'"):
                _validate_local_policy()

    @patch('src.llm.local_llama_client.requests.post')
    @patch('src.llm.local_llama_client._validate_local_policy')
    def test_call_local_llama_success(self, mock_validate, mock_post):
        """Test successful call_local_llama."""
        mock_validate.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'{"message":{"content":"Test response"}}'
        mock_post.return_value = mock_response

        result = call_local_llama("Test prompt")

        assert result.text == "Test response"
        assert result.provider == "local_llama"
        assert isinstance(result.latency_ms, int)
        assert result.bytes_in > 0
        assert result.bytes_out > 0

    @patch('src.llm.local_llama_client._validate_local_policy')
    def test_call_local_llama_empty_response(self, mock_validate):
        """Test call_local_llama with empty response."""
        mock_validate.return_value = None

        with patch('src.llm.local_llama_client.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": ""}}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError, match="returned empty text"):
                call_local_llama("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__])
