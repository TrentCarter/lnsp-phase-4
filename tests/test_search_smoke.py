"""Smoke test for LNSP search API - basic functionality validation."""

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


class TestSearchSmoke:
    """Smoke tests for search API - minimal functionality checks."""

    def test_search_returns_200(self):
        """Test that /search endpoint returns 200 status."""
        payload = {
            "q": "What is photosynthesis?",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200

    def test_search_returns_at_least_one_hit(self):
        """Test that search returns at least one result when system is ready."""
        payload = {
            "q": "What is photosynthesis?",
            "lane": "L1_FACTOID",
            "top_k": 5
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        # Only check for hits if the system is loaded (not empty)
        health_response = client.get("/healthz")
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data.get("status") == "ready":
                assert len(data["items"]) >= 1, "Expected at least 1 hit when system is ready"

    def test_faiss_dimension_is_784(self):
        """Test that FAISS index has dimension 784."""
        response = client.get("/admin/faiss")
        assert response.status_code == 200
        data = response.json()
        assert "dim" in data
        assert data["dim"] == 784, f"Expected dimension 784, got {data['dim']}"

    def test_faiss_has_vectors(self):
        """Test that FAISS index has some vectors loaded."""
        response = client.get("/admin/faiss")
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data
        assert data["vectors"] > 0, f"Expected vectors > 0, got {data['vectors']}"
