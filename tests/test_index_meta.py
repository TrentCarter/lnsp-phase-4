"""
Tests for index metadata and health endpoints
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.faiss_index import _save_index_meta


class TestIndexMeta:
    """Test index metadata functionality."""

    def test_save_index_meta_creates_directory(self):
        """Test that _save_index_meta creates artifacts directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "artifacts" / "index_meta.json"
            index_path = "/fake/path/test.index"
            meta = {
                "test": "data",
                "vectors": 1000,
                "nlist": 256,
                "requested_nlist": 300,
                "max_safe_nlist": 250
            }

            _save_index_meta(index_path, meta)

            assert meta_path.exists()
            with open(meta_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data[index_path] == meta

            # Verify new fields are present
            assert "requested_nlist" in saved_data[index_path]
            assert "max_safe_nlist" in saved_data[index_path]

    def test_save_index_meta_merges_with_existing(self):
        """Test that _save_index_meta merges with existing metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "artifacts" / "index_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            # Create existing data
            existing_data = {"/old/path.index": {"old": "data"}}
            with open(meta_path, 'w') as f:
                json.dump(existing_data, f)

            # Save new data
            index_path = "/new/path.index"
            meta = {"test": "data", "vectors": 1000}
            _save_index_meta(index_path, meta)

            with open(meta_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data[index_path] == meta
            assert saved_data["/old/path.index"] == {"old": "data"}

    def test_save_index_meta_handles_invalid_json(self):
        """Test that _save_index_meta handles invalid existing JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "artifacts" / "index_meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            # Write invalid JSON
            with open(meta_path, 'w') as f:
                f.write("invalid json content")

            # Save new data
            index_path = "/test/path.index"
            meta = {"test": "data"}
            _save_index_meta(index_path, meta)

            with open(meta_path, 'r') as f:
                saved_data = json.load(f)

            assert saved_data[index_path] == meta


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
                "build_time": 45.2,
                "requested_nlist": 600,
                "max_safe_nlist": 250
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


class TestIndexValidation:
    """Test index training validation."""

    def test_training_validation_sufficient_data(self):
        """Test that training proceeds when data is sufficient."""
        import numpy as np
        from src.faiss_index import build_ivf_flat_cosine

        # Create sufficient training data (more than 40*nlist)
        nlist = 10
        vectors = np.random.random((400, 768)).astype(np.float32)  # 400 > 40*10

        index, trained = build_ivf_flat_cosine(vectors, nlist=nlist)

        assert trained is True
        assert index is not None

    def test_training_validation_insufficient_data(self):
        """Test that training fails when data is insufficient."""
        import numpy as np
        from src.faiss_index import build_ivf_flat_cosine

        # Create insufficient training data (less than 40*nlist)
        nlist = 10
        vectors = np.random.random((300, 768)).astype(np.float32)  # 300 < 40*10

        index, trained = build_ivf_flat_cosine(vectors, nlist=nlist)

        assert trained is False
        assert index is None

    def test_training_validation_pq_sufficient_data(self):
        """Test that PQ training proceeds when data is sufficient."""
        import numpy as np
        from src.faiss_index import build_ivf_pq_cosine

        # Create sufficient training data (more than 40*nlist)
        nlist = 10
        vectors = np.random.random((400, 768)).astype(np.float32)  # 400 > 40*10

        index, trained = build_ivf_pq_cosine(vectors, nlist=nlist)

        assert trained is True
        assert index is not None

    def test_training_validation_pq_insufficient_data(self):
        """Test that PQ training fails when data is insufficient."""
        import numpy as np
        from src.faiss_index import build_ivf_pq_cosine

        # Create insufficient training data (less than 40*nlist)
        nlist = 10
        vectors = np.random.random((300, 768)).astype(np.float32)  # 300 < 40*10

        index, trained = build_ivf_pq_cosine(vectors, nlist=nlist)

        assert trained is False
        assert index is None


class TestIndexMetaFields:
    """Test index metadata field validation."""

    def test_index_meta_keys(self):
        """Test that index metadata contains all required keys."""
        import tempfile
        import numpy as np
        from pathlib import Path
        from src.faiss_index import main

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test NPZ file
            vectors = np.random.random((1000, 768)).astype(np.float32)
            ids = np.arange(1000)
            doc_ids = [f"doc_{i}" for i in range(1000)]
            concept_texts = [f"concept_{i}" for i in range(1000)]
            tmd_dense = np.random.random((1000, 16)).astype(np.float32)
            lane_indices = np.random.randint(0, 3, 1000)

            npz_path = Path(tmpdir) / "test_vectors.npz"
            np.savez(
                npz_path,
                vectors=vectors,
                ids=ids,
                doc_ids=doc_ids,
                concept_texts=concept_texts,
                tmd_dense=tmd_dense,
                lane_indices=lane_indices
            )

            # Mock artifacts directory
            artifacts_path = Path(tmpdir) / "artifacts"
            artifacts_path.mkdir()

            # Mock the main function to save to our test directory
            import sys
            from unittest.mock import patch

            with patch('sys.argv', ['faiss_index.py', '--npz', str(npz_path), '--nlist', '100']):
                with patch('src.faiss_index._save_index_meta') as mock_save:
                    with patch('faiss.write_index'):
                        with patch('os.makedirs'):
                            try:
                                main()
                            except SystemExit:
                                pass

                    # Check that metadata was saved with correct fields
                    assert mock_save.called
                    args, kwargs = mock_save.call_args
                    meta = args[1]

                    # Check all required keys are present
                    for k in ["type", "metric", "nlist", "nprobe", "count", "build_seconds", "requested_nlist", "max_safe_nlist"]:
                        assert k in meta

                    # Check constraint: nlist <= max_safe_nlist
                    assert meta["nlist"] <= meta["max_safe_nlist"]

                    # Check requested_nlist handling
                    if meta["requested_nlist"] is not None:
                        assert meta["nlist"] <= max(meta["requested_nlist"], 0)

    def test_nlist_auto_reduction_metadata(self):
        """Test that nlist auto-reduction is properly recorded in metadata."""
        from src.faiss_index import calculate_nlist

        # Test case where requested nlist is reduced
        n_vectors = 10000
        requested = 512
        actual = calculate_nlist(n_vectors, requested)
        max_safe = n_vectors // 40

        # Verify the calculation matches what metadata should record
        assert actual == min(requested, max_safe)
        assert actual <= max_safe
