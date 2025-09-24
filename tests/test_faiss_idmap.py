#!/usr/bin/env python3
"""Unit tests for FAISS ID mapping functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_faiss import FaissDB


def test_faiss_id_mapping():
    """Test that FAISS returns doc_ids correctly for ID-mapped indices."""
    # Create test data
    np.random.seed(42)
    vectors = np.random.randn(100, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # L2 normalize

    # Create test metadata
    doc_ids = np.array([f"doc_{i}" for i in range(100)], dtype=object)
    cpe_ids = np.arange(100, dtype=object)
    concept_texts = np.array([f"concept {i}" for i in range(100)], dtype=object)
    lane_indices = np.random.randint(0, 3, 100)

    # Save test NPZ
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        npz_path = f.name
        np.savez(f, vectors=vectors, doc_ids=doc_ids, cpe_ids=cpe_ids,
                concept_texts=concept_texts, lane_indices=lane_indices)

    try:
        # Create FAISS index (simulating the build process)
        import faiss

        nlist = 4
        quant = faiss.IndexFlatIP(768)
        ivf_index = faiss.IndexIVFFlat(quant, 768, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf_index.train(vectors)

        # Create IDMap2 with empty IVF index
        index = faiss.IndexIDMap2(ivf_index)

        # Add vectors with IDs (this also adds to the underlying IVF)
        ids = np.arange(100, dtype=np.int64)  # Positional IDs
        index.add_with_ids(vectors, ids)

        # Save index
        with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as f:
            index_path = f.name
            faiss.write_index(index, index_path)

        try:
            # Test loading with FaissDB
            db = FaissDB(meta_npz_path=npz_path)
            db.load(index_path)

            # Test search
            query = vectors[0:1]  # Use first vector as query
            results = db.search_legacy(query[0], topk=5)

            # Verify results
            assert len(results) == 5
            assert results[0]['doc_id'] == 'doc_0'  # Should return doc_ids
            assert results[0]['cpe_id'] == '0'
            assert results[0]['score'] > 0.9  # Should be very similar to itself

            # Verify all results have required fields
            for result in results:
                assert 'doc_id' in result
                assert 'cpe_id' in result
                assert 'score' in result
                assert 'rank' in result
                assert 'metadata' in result
                assert 'concept_text' in result['metadata']

            print("✅ FAISS ID mapping test passed")

        finally:
            os.unlink(index_path)

    finally:
        os.unlink(npz_path)


if __name__ == "__main__":
    test_faiss_id_mapping()
