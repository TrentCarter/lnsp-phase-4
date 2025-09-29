import pytest

from tests.inspect_cpesh_dump import collect_cpesh_samples


def test_cpesh_text_and_meta_present():
    rows = collect_cpesh_samples(limit=5, include_active=True, include_segments=True)
    if not rows:
        pytest.skip("no CPESH samples available")

    assert isinstance(rows, list)
    for row in rows:
        assert row.get("cpe_id"), "cpe_id missing"
        assert row.get("created_at"), "created_at missing"
        assert "access_count" in row, "access_count missing"
        assert "tmd_text" in row and isinstance(row["tmd_text"], str)

        for key in ("concept_text", "probe_question", "expected_answer"):
            if key in row and row[key] is not None:
                assert isinstance(row[key], str), f"{key} should be a string"

        if "lane_index" in row and row["lane_index"] is not None:
            assert isinstance(row["lane_index"], int)
