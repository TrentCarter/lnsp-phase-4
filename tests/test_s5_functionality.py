#!/usr/bin/env python3
"""Test CPESH cache functionality."""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.retrieve import RetrievalContext
from schemas import SearchRequest, SearchItem, CPESH


class TestCPESHCache(unittest.TestCase):
    """Test CPESH caching functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.temp_dir) / "test_cpesh_cache.jsonl"

        # Mock environment
        self.env_patcher = patch.dict(os.environ, {
            'LNSP_CPESH_CACHE': str(self.cache_path),
            'LNSP_CPESH_MAX_K': '2',
            'LNSP_CPESH_TIMEOUT_S': '4',
        })
        self.env_patcher.start()

    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cpesh_cache_smoke(self):
        """Test that CPESH cache works correctly."""
        # Create a mock hit
        mock_hit = {
            "id": "test_doc_123",
            "doc_id": "test_doc_123",
            "score": 0.8,
            "concept_text": "This is a test document about artificial intelligence.",
            "tmd_code": "2.0.27",
            "lane_index": 0
        }

        # Create mock LLM client
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "concept": "artificial intelligence",
            "probe": "What is artificial intelligence?",
            "expected": "A field of computer science",
            "soft_negative": "Machine learning",
            "hard_negative": "Human intelligence",
            "insufficient_evidence": False
        }

        # Create a minimal retrieval context
        with patch('api.retrieve.FaissDB'), \
             patch('api.retrieve.EmbeddingBackend'), \
             patch('api.retrieve.LightRAGHybridRetriever'):

            # Mock the context to avoid heavy initialization
            ctx = MagicMock()
            ctx.cpesh_max_k = 2
            ctx.cpesh_timeout_s = 4.0
            ctx.cpesh_cache_path = self.cache_path
            ctx.cpesh_cache = {}
            ctx.id_quality = {}
            ctx.w_cos = 0.85
            ctx.w_q = 0.15
            ctx._ensure_llm.return_value = mock_llm

            # Test cache hit scenario
            cached_cpesh = {
                "doc_id": "doc_demo",
                "quality": 0.91,
                "cosine": 0.62,
                "insufficient_evidence": False,
            }

    def test_format_tmd_code_from_bits(self):
        """Test TMD formatting from packed bits."""
        # Test with packed bits (domain=2, task=0, modifier=27)
        packed_bits = (2 << 12) | (0 << 7) | (27 << 1)  # 0b0010000000011011 = 8203
        result = format_tmd_code(packed_bits)
        self.assertEqual(result, "2.0.27")
        print(f"✅ TMD formatting from bits: {packed_bits} -> {result}")

    def test_format_tmd_code_from_dict(self):
        """Test TMD formatting from dictionary."""
        # Test with dict containing individual codes
        tmd_dict = {
            "tmd_bits": None,
            "domain_code": 2,
            "task_code": 0,
            "modifier_code": 27
        }
        result = format_tmd_code(tmd_dict)
        self.assertEqual(result, "2.0.27")
        print(f"✅ TMD formatting from dict: {tmd_dict} -> {result}")

    def test_format_tmd_code_fallback(self):
        """Test TMD formatting fallback."""
        # Test fallback to existing tmd_code
        tmd_dict = {
            "tmd_code": "9.5.16",
            "domain_code": None,
            "task_code": None,
            "modifier_code": None
        }
        result = format_tmd_code(tmd_dict)
        self.assertEqual(result, "9.5.16")
        print(f"✅ TMD formatting fallback: {tmd_dict} -> {result}")

    def test_pack_unpack_tmd(self):
        """Test TMD packing and unpacking."""
        # Pack TMD codes
        domain, task, modifier = 2, 0, 27
        packed = pack_tmd(domain, task, modifier)

        # Unpack and verify
        unpacked_domain, unpacked_task, unpacked_modifier = unpack_tmd(packed)

        self.assertEqual(unpacked_domain, domain)
        self.assertEqual(unpacked_task, task)
        self.assertEqual(unpacked_modifier, modifier)

        print(f"✅ TMD pack/unpack: {domain}.{task}.{modifier} -> {packed} -> {unpacked_domain}.{unpacked_task}.{unpacked_modifier}")


class TestCPESHCache(unittest.TestCase):
    """Test CPESH cache functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_path = Path(self.temp_dir) / "test_cpesh_cache.jsonl"

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cpesh_cache_format(self):
        """Test CPESH cache file format."""
        # Create a sample CPESH cache entry
        cache_entry = {
            "doc_id": "test_doc_123",
            "cpesh": {
                "concept": "artificial intelligence",
                "probe": "What is artificial intelligence?",
                "expected": "A field of computer science",
                "soft_negative": "Machine learning",
                "hard_negative": "Human intelligence"
            }
        }

        # Write to cache file
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")

        # Read back and verify
        with open(self.cache_path, "r") as f:
            line = f.readline().strip()
            read_entry = json.loads(line)

        self.assertEqual(read_entry["doc_id"], cache_entry["doc_id"])
        self.assertEqual(read_entry["cpesh"]["concept"], cache_entry["cpesh"]["concept"])

        print(f"✅ CPESH cache format OK: {cache_entry['doc_id']}")


if __name__ == "__main__":
    print("Running S5 tests...")
    unittest.main(verbosity=2)
