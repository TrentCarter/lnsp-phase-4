#!/usr/bin/env python3
"""
Test for the data generator utility.
"""

import pytest
from tests.data_generator import generate_synthetic_data_points

def test_generate_synthetic_data_points():
    """Tests that the data generator produces the correct number of points and all fields are populated."""
    num_points = 5
    data = generate_synthetic_data_points(num_points)

    assert len(data) == num_points

    for point in data:
        # Check top-level keys
        assert all(k in point for k in ['cpe_id', 'doc_id', 'created_at', 'last_accessed', 'access_count', 'cpesh', 'tmd'])

        # Check CPESH keys
        cpesh = point['cpesh']
        assert all(k in cpesh for k in ['concept_text', 'probe_question', 'expected_answer', 'soft_negative', 'hard_negative'])
        assert isinstance(cpesh['concept_text'], str)

        # Check TMD keys
        tmd = point['tmd']
        assert all(k in tmd for k in ['domain_name', 'task_name', 'modifier_name', 'domain_code', 'task_code', 'modifier_code'])
        assert isinstance(tmd['domain_name'], str)

        # Check other metadata
        assert isinstance(point['access_count'], int)
        assert 'Z' in point['created_at']

    print(f"\nSuccessfully generated and validated {num_points} data points.")


def test_generate_single_data_point():
    """Tests that the generator works for a single data point."""
    data = generate_synthetic_data_points(1)
    assert len(data) == 1
    assert isinstance(data[0]['cpe_id'], str)


def test_generator_invalid_input():
    """Tests that the generator raises an error for invalid input."""
    with pytest.raises(ValueError):
        generate_synthetic_data_points(0)

    with pytest.raises(ValueError):
        generate_synthetic_data_points(-1)
