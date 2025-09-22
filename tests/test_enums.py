"""Tests for frozen TMD enumerations."""

from __future__ import annotations

import pytest

from src.enums import (
    DOMAIN_LABELS,
    TASK_LABELS,
    MODIFIER_LABELS,
    DOMAIN_TO_CODE,
    TASK_TO_CODE,
    MODIFIER_TO_CODE,
    CODE_TO_DOMAIN,
    CODE_TO_TASK,
    CODE_TO_MODIFIER,
    domain_code,
    task_code,
    modifier_code,
    domain_label,
    task_label,
    modifier_label,
    EnumLookupError,
    TMDEntry,
)


class TestFrozenEnums:
    """Test that enums are properly frozen and bidirectional."""

    def test_domain_labels_frozen(self):
        assert len(DOMAIN_LABELS) == 16
        assert DOMAIN_LABELS[0] == "science"
        assert DOMAIN_LABELS[-1] == "sociology"

    def test_task_labels_frozen(self):
        assert len(TASK_LABELS) == 32
        assert TASK_LABELS[0] == "fact_retrieval"
        assert TASK_LABELS[-1] == "prompt_completion"

    def test_modifier_labels_frozen(self):
        assert len(MODIFIER_LABELS) == 64
        assert MODIFIER_LABELS[0] == "biochemical"
        assert MODIFIER_LABELS[-1] == "resilient"

    def test_bidirectional_mappings(self):
        """Test that label -> code -> label is identity."""
        for label in DOMAIN_LABELS:
            code = DOMAIN_TO_CODE[label]
            assert CODE_TO_DOMAIN[code] == label

        for label in TASK_LABELS:
            code = TASK_TO_CODE[label]
            assert CODE_TO_TASK[code] == label

        for label in MODIFIER_LABELS:
            code = MODIFIER_TO_CODE[label]
            assert CODE_TO_MODIFIER[code] == label


class TestEnumFunctions:
    """Test enum lookup functions."""

    def test_valid_domain_lookups(self):
        assert domain_code("science") == 0
        assert domain_code("art") == 9
        assert domain_code("sociology") == 15

    def test_valid_task_lookups(self):
        assert task_code("fact_retrieval") == 0
        assert task_code("code_generation") == 14
        assert task_code("prompt_completion") == 31

    def test_valid_modifier_lookups(self):
        assert modifier_code("biochemical") == 0
        assert modifier_code("historical") == 5
        assert modifier_code("resilient") == 63

    def test_invalid_domain_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown domain label: nonexistent"):
            domain_code("nonexistent")

    def test_invalid_task_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown task label: nonexistent"):
            task_code("nonexistent")

    def test_invalid_modifier_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown modifier label: nonexistent"):
            modifier_code("nonexistent")

    def test_reverse_lookups(self):
        assert domain_label(0) == "science"
        assert task_label(0) == "fact_retrieval"
        assert modifier_label(0) == "biochemical"

    def test_invalid_reverse_domain_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown domain code: 999"):
            domain_label(999)

    def test_invalid_reverse_task_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown task code: 999"):
            task_label(999)

    def test_invalid_reverse_modifier_lookup(self):
        with pytest.raises(EnumLookupError, match="Unknown modifier code: 999"):
            modifier_label(999)


class TestTMDEntry:
    """Test TMDEntry dataclass."""

    def test_lane_index_computation(self):
        # Test with known TMD values
        entry = TMDEntry(domain_code=9, task_code=0, modifier_code=5)  # art, fact_retrieval, historical
        lane = entry.lane_index
        assert isinstance(lane, int)
        assert 0 <= lane <= 32767

    def test_dataclass_immutable(self):
        entry = TMDEntry(domain_code=0, task_code=0, modifier_code=0)
        with pytest.raises(AttributeError):
            entry.domain_code = 1

