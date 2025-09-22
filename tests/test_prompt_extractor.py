from __future__ import annotations
from src.prompt_extractor import extract_stub


def test_extract_stub_music():
    c = "! (The Dismemberment Plan album)\n! is the debut studio album by American indie rock band The Dismemberment Plan. It was released on October 2, 1995 on DeSoto Records."
    ex = extract_stub(c)
    assert ex.domain == "art"
    assert ex.task == "fact_retrieval"
    assert ex.content_type == "factual"
    assert ex.concept_text.endswith(".")
    assert ex.expected_answer.endswith(".")
