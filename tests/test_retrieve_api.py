"""Unit tests for LNSP retrieval API."""

import os
import pytest
from fastapi.testclient import TestClient

# Use stub searcher in test mode to avoid heavy dependencies
if os.getenv("LNSP_TEST_MODE", "0") == "1":
    from src.search_backends.stub import StubSearcher
    # Monkey patch the retrieval context to use stub
    import src.api.retrieve
    src.api.retrieve.RetrievalContext._get_searcher = lambda self: StubSearcher()

from src.api.retrieve import app

client = TestClient(app)


class TestRetrievalAPI:
    """Test suite for retrieval API endpoints."""

    def test_health_endpoint(self):
        """Test health check endpoint returns proper status."""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ready", "empty"]

    def test_search_endpoint_basic(self):
        """Test basic search functionality."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "lane" in data
        assert "mode" in data
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= 5

    @pytest.mark.parametrize("lane", ["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"])
    def test_lane_routing(self, lane):
        """Test that different lanes are properly routed."""
        payload = {
            "q": "test query",
            "lane": lane,
            "top_k": 3
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["lane"] == lane

    @pytest.mark.parametrize("top_k", [1, 5, 10])
    def test_top_k_shape(self, top_k):
        """Test that top_k parameter controls result count."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": top_k
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) <= top_k

    @pytest.mark.heavy
    def test_score_monotonicity(self):
        """Test that scores are in descending order (higher scores first)."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": 10
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        items = data["items"]

        # Extract scores, filtering out None values
        scores = [item.get("score") for item in items if item.get("score") is not None]

        # Check that scores are in descending order (allowing for ties)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Scores not monotonic: {scores[i]} < {scores[i + 1]}"

    def test_invalid_lane(self):
        """Test that invalid lane returns error."""
        payload = {
            "q": "test query",
            "lane": "INVALID_LANE",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        # Should either return 422 (validation error) or handle gracefully
        assert response.status_code in [200, 422]

    def test_empty_query(self):
        """Test handling of empty query."""
        payload = {
            "q": "",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200  # Should handle gracefully

    def test_trace_id_generation(self):
        """Test that trace_id is generated or can be provided."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)
        assert len(data["trace_id"]) > 0

    def test_custom_trace_id(self):
        """Test providing custom trace_id via header."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        headers = {"x-trace-id": "custom-test-id-123"}
        response = client.post("/search", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data.get("trace_id") == "custom-test-id-123"

    def test_custom_trace_id(self):
        """Test providing custom trace_id via header."""
        payload = {
            "q": "test query",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        headers = {"x-trace-id": "custom-test-id-123"}
        response = client.post("/search", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data.get("trace_id") == "custom-test-id-123"
