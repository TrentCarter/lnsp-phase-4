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

    def test_faiss_dimension_is_768(self):
        """Test that FAISS index has dimension 768 for pure GTR mode."""
        response = client.get("/admin/faiss")
        assert response.status_code == 200
        data = response.json()
        assert "dim" in data
        assert data["dim"] == 768, f"Expected dimension 768, got {data['dim']}"

    def test_search_returns_hydrated_fields(self):
        """Test that search results include hydrated metadata fields."""
        payload = {
            "q": "What is photosynthesis?",
            "lane": "L1_FACTOID",
            "top_k": 3
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Check basic structure
        assert "items" in data
        items = data["items"]

        # Only validate hydrated fields if we have results
        if items:
            for item in items:
                # Required hydrated fields from P13
                assert "doc_id" in item, "Missing doc_id in search result"
                assert "cpe_id" in item, "Missing cpe_id in search result"
                assert "concept_text" in item, "Missing concept_text in search result"
                assert "tmd_code" in item, "Missing tmd_code in search result"
                assert "lane_index" in item, "Missing lane_index in search result"

                # Check metadata structure
                assert "metadata" in item, "Missing metadata in search result"
                metadata = item["metadata"]
                assert "concept_text" in metadata, "Missing concept_text in metadata"
                assert "doc_id" in metadata, "Missing doc_id in metadata"

                # Validate data types
                assert isinstance(item["doc_id"], str), "doc_id should be string"
                assert isinstance(item["cpe_id"], str), "cpe_id should be string"
                assert isinstance(item["concept_text"], str), "concept_text should be string"
                assert isinstance(item["lane_index"], int), "lane_index should be int"

    def test_faiss_has_vectors(self):
        """Test that FAISS index has some vectors loaded."""
        response = client.get("/admin/faiss")
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data
        assert data["vectors"] > 0, f"Expected vectors > 0, got {data['vectors']}"
