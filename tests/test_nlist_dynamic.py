"""
Tests for dynamic nlist calculation functionality
"""

import math
import pytest
from src.faiss_index import calculate_nlist


class TestDynamicNlist:
    """Test dynamic nlist calculation logic."""

    def test_small_auto_maxsafe(self):
        """Test that small datasets use max_safe_nlist automatically."""
        assert calculate_nlist(5000, None) == 5000 // 40  # 125

    def test_10k_default_band(self):
        """Test 10k vectors use min(200, max_safe) policy."""
        # 10k → min(200, max_safe)
        assert calculate_nlist(10000, None) == 200

    def test_25k_band(self):
        """Test 25k vectors use min(512, max_safe) policy."""
        # 25k < 40k → min(512, max_safe) where max_safe=625
        assert calculate_nlist(25000, None) == 512

    def test_120k_sqrt(self):
        """Test large datasets (>100k) use sqrt(N) policy."""
        n = 120000
        expected = min(int(math.sqrt(n)), n // 40)
        assert calculate_nlist(n, None) == expected

    def test_requested_downshift(self):
        """Test that requested nlist is downshifted when it exceeds max_safe."""
        # request 512 on 10k should downshift to max_safe=250
        assert calculate_nlist(10000, 512) == 250

    def test_requested_safe(self):
        """Test that safe requested nlist is used as-is."""
        # request 200 on 10k is safe and equals policy's pick
        assert calculate_nlist(10000, 200) == 200

    def test_edge_case_minimum(self):
        """Test minimum nlist enforcement."""
        # Very small dataset
        assert calculate_nlist(10, None) == 1  # max(1, 10//40) = 1

    def test_edge_case_boundary_8k(self):
        """Test boundary between <8k and 8k-20k bands."""
        # Just below 8k
        assert calculate_nlist(7999, None) == 7999 // 40  # 199

        # At 8k boundary
        assert calculate_nlist(8000, None) == 200  # min(200, 8000//40)

    def test_edge_case_boundary_20k(self):
        """Test boundary between 8k-20k and 20k-40k bands."""
        # Just below 20k
        assert calculate_nlist(19999, None) == 200  # min(200, 19999//40)

        # At 20k boundary
        assert calculate_nlist(20000, None) == 512  # min(512, 20000//40)

    def test_edge_case_boundary_40k(self):
        """Test boundary between 20k-40k and 40k-100k bands."""
        # Just below 40k
        assert calculate_nlist(39999, None) == 512  # min(512, 39999//40)

        # At 40k boundary
        assert calculate_nlist(40000, None) == min(1024, 40000//40)  # min(1024, 1000)

    def test_edge_case_boundary_100k(self):
        """Test boundary between 40k-100k and >100k bands."""
        # Just below 100k
        assert calculate_nlist(99999, None) == min(1024, 99999//40)  # min(1024, 2499)

        # At 100k boundary
        assert calculate_nlist(100000, None) == min(int(math.sqrt(100000)), 100000//40)  # min(316, 2500)

    def test_large_dataset_sqrt_dominates(self):
        """Test that for very large datasets, sqrt(N) is used when smaller than max_safe."""
        n = 10000  # sqrt(10000) = 100, max_safe = 250
        assert calculate_nlist(n, None) == min(100, 250)  # 100

    def test_very_large_dataset_max_safe_dominates(self):
        """Test that max_safe is enforced even when sqrt(N) is larger."""
        n = 1000000  # sqrt(1000000) = 1000, max_safe = 25000
        assert calculate_nlist(n, None) == min(1000, 25000)  # 1000

    def test_explicit_none_handling(self):
        """Test that None is handled the same as not providing requested_nlist."""
        assert calculate_nlist(10000, None) == calculate_nlist(10000)

    def test_zero_vectors_edge_case(self):
        """Test edge case with zero vectors."""
        assert calculate_nlist(0, None) == 1  # max(1, 0//40) = 1
