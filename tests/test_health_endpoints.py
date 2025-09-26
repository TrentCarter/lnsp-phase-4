"""
Tests for health endpoints
"""

import json
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path


class TestHealthEndpoints:
    """Test health endpoint functionality."""

    def test_health_faiss_endpoint_with_index_meta(self, client):
        """Test /health/faiss endpoint when index_meta.json exists."""
        # Create mock index metadata
        index_meta = {
            "/artifacts/test_ivf_flat_ip.index": {
                "vectors": 10000,
                "trained": True,
                "nlist": 512,
                "nprobe": 8,
                "metric": "ip",
                "index_type": "ivf_flat",
                "code_size": 1024000,
                "build_time": 45.2
            }
        }

        with patch('src.api.retrieve.get_context') as mock_get_context:
            mock_ctx = mock_get_context.return_value
            mock_ctx.loaded = True
            mock_ctx.npz_path = "artifacts/fw10k_vectors.npz"

            with patch('builtins.open', mock_open()) as mock_file:
                mock_file.return_value.read.return_value = json.dumps(index_meta)

                with patch('os.path.exists', return_value=True):
                    response = client.get("/health/faiss")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["loaded"] is True
                    assert data["type"] == "ivf_flat"
                    assert data["metric"] == "ip"
                    assert data["nlist"] == 512
                    assert data["nprobe"] == 8
                    assert data["ntotal"] == 10000

    def test_health_faiss_endpoint_without_index_meta(self, client):
        """Test /health/faiss endpoint when index_meta.json doesn't exist."""
        with patch('src.api.retrieve.get_context') as mock_get_context:
            mock_ctx = mock_get_context.return_value
            mock_ctx.loaded = False
            mock_ctx.npz_path = "artifacts/fw10k_vectors.npz"

            with patch('builtins.open', side_effect=FileNotFoundError):
                with patch('os.path.exists', return_value=False):
                    response = client.get("/health/faiss")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["loaded"] is False
                    assert data["type"] == "N/A"
                    assert data["metric"] == "IP"
                    assert data["nlist"] == 0
                    assert data["nprobe"] == 16  # default from env
                    assert data["ntotal"] == 0

    def test_cache_stats_endpoint_empty_cache(self, client):
        """Test /cache/stats endpoint with empty cache."""
        with patch('src.api.retrieve.get_context') as mock_get_context:
            mock_ctx = mock_get_context.return_value
            mock_ctx.cpesh_cache = {}

            response = client.get("/cache/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["entries"] == 0
            assert data["oldest_created_at"] is None
            assert data["newest_last_accessed"] is None
            assert data["p50_access_age"] is None
            assert data["top_docs_by_access"] == []

    def test_cache_stats_endpoint_with_data(self, client):
        """Test /cache/stats endpoint with populated cache."""
        from datetime import datetime

        # Create mock cache data
        mock_cache = {
            "doc1": {
                "doc_id": "doc1",
                "cpesh": {
                    "concept": "test concept",
                    "created_at": "2025-01-01T00:00:00Z",
                    "last_accessed": "2025-01-01T01:00:00Z"
                },
                "access_count": 5
            },
            "doc2": {
                "doc_id": "doc2",
                "cpesh": {
                    "concept": "another concept",
                    "created_at": "2025-01-01T02:00:00Z",
                    "last_accessed": "2025-01-01T03:00:00Z"
                },
                "access_count": 3
            }
        }

        with patch('src.api.retrieve.get_context') as mock_get_context:
            mock_ctx = mock_get_context.return_value
            mock_ctx.cpesh_cache = mock_cache

            response = client.get("/cache/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["entries"] == 2
            assert data["oldest_created_at"] == "2025-01-01T00:00:00Z"
            assert data["newest_last_accessed"] == "2025-01-01T03:00:00Z"
            assert "p50_access_age" in data
            assert len(data["top_docs_by_access"]) == 2

    def test_metrics_slo_endpoint_with_report(self, client):
        """Test /metrics/slo endpoint when day_s3_report.md exists."""
        mock_report_content = """
        Last run: 2025-01-01T12:00:00Z

        Performance Metrics:
        Hit@1: 85.5%
        Hit@3: 92.3%

        Latency Analysis:
        P50: 45.2 ms
        P95: 123.4 ms

        Recommended nprobe: 16
        """

        with patch('builtins.open', mock_open(read_data=mock_report_content)):
            response = client.get("/metrics/slo")

            assert response.status_code == 200
            data = response.json()
            assert data["last_run"] == "2025-01-01T12:00:00Z"
            assert data["hit_at_1"] == 85.5
            assert data["hit_at_3"] == 92.3
            assert data["p50_latency"] == 45.2
            assert data["p95_latency"] == 123.4
            assert data["nprobe_recommended"] == 16

    def test_metrics_slo_endpoint_without_report(self, client):
        """Test /metrics/slo endpoint when day_s3_report.md doesn't exist."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            response = client.get("/metrics/slo")

            assert response.status_code == 200
            data = response.json()
            assert data["last_run"] is None
            assert data["hit_at_1"] is None
            assert data["hit_at_3"] is None
            assert data["p50_latency"] is None
            assert data["p95_latency"] is None
            assert data["nprobe_recommended"] is None
