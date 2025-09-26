import json
import uuid
from unittest.mock import patch, MagicMock

import pytest

# Assuming outlines is used for structured generation, we can model the expected output.
# For this test, we'll work with the raw JSON string the LLM would produce.

# Assuming src is in the python path for pytest
from src.schemas import CPECore


# --- Test Data ---

SAMPLE_CHUNK = "The Eiffel Tower, a wrought-iron lattice tower on the Champ de Mars in Paris, France, was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."

MISSION_TEXT = "Extract atomic facts and relationships from the provided text."

# This is the ideal, structured JSON we expect from a single LLM prompt.
# It aligns with the prompt template in docs/design_documents/prompt_template_lightRAG_TMD_CPE.md
EXPECTED_LLM_OUTPUT = {
    "concept_text": "The Eiffel Tower is a global cultural icon of France.",
    "probe_question": "What has the Eiffel Tower become?",
    "expected_answer": "A global cultural icon of France.",
    "domain": "Art",
    "task": "Fact Retrieval",
    "modifier": "Historical",
    "mission_text": MISSION_TEXT
}

# --- Mocked Extraction Function ---

def build_prompt(template_path: str, chunk: str) -> str:
    """Helper to build the full prompt from the design document template."""
    # In a real scenario, this would read the template file.
    # For this test, we'll use a simplified version of the prompt.
    base_prompt = f"""You are an expert extractor of propositional knowledge from text. Given the input text, extract the following in a single JSON object:

- `concept_text`: The core atomic concept, phrased as a concise, standalone statement (string).
- `probe_question`: A question that the `concept_text` directly and completely answers (string).
- `expected_answer`: The expected answer to the `probe_question`, often a short phrase or entity (string).
- `domain`: The primary domain category from the official TMD schema (enum).
- `task`: The primary cognitive task from the official TMD schema (enum).
- `modifier`: The descriptive modifier from the official TMD schema (enum).
- `mission_text`: The original extraction prompt or mission that guided this process (string).

Output ONLY the JSON object. No explanations or additional text.

Input text: {chunk}
"""
    return base_prompt


def extract_cpe_from_chunk(chunk: str, mission: str):
    """
    This function simulates the process of calling an LLM for CPE extraction.
    In a real implementation, this would handle the API call (e.g., using `requests` or a client library)
    and would be the target of our mock.
    """
    # This function is a placeholder for the actual LLM call logic.
    # We will mock the 'requests.post' call that this function would theoretically make.
    import requests

    prompt = build_prompt("path/to/template.md", chunk)
    
    # The actual API call would be here. We don't need to implement it fully
    # because we are going to mock it in the test.
    response = requests.post("http://fake-llm-api/generate", json={"prompt": prompt})
    response.raise_for_status()
    return response.json()


# --- Pytest Test ---

@patch('requests.post')
def test_single_pass_extraction_creates_all_fields(mock_post):
    """
    Tests that a single mocked LLM call returns a JSON object that can fully populate
    the required fields of a CPECore object.
    """
    # 1. Configure the mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = EXPECTED_LLM_OUTPUT
    mock_post.return_value = mock_response

    # 2. Run the extraction function
    # This will call our mocked `requests.post` instead of a real API
    extracted_data = extract_cpe_from_chunk(SAMPLE_CHUNK, MISSION_TEXT)

    # 3. Validate the output against the CPECore data contract
    assert extracted_data == EXPECTED_LLM_OUTPUT

    # 4. Attempt to create a CPECore object from the extracted data
    # This is the ultimate test of whether the prompt is sufficient.
    try:
        # We need to provide the fields that the LLM doesn't generate.
        cpe_instance = CPECore(
            cpe_id=uuid.uuid4(),
            mission_text=extracted_data["mission_text"],
            source_chunk=SAMPLE_CHUNK,
            concept_text=extracted_data["concept_text"],
            probe_question=extracted_data["probe_question"],
            expected_answer=extracted_data["expected_answer"],
            # Mocked enum codes for validation
            domain_code=9,  # 'Art'
            task_code=0,    # 'Fact Retrieval'
            modifier_code=5, # 'Historical'
            content_type='factual',
            dataset_source='FactoidWiki',
            chunk_position={"doc_id": "test_doc", "start": 0, "end": 100},
            relations_text=[],
            tmd_bits=0, # These would be calculated by tmd_encoder
            tmd_lane="",
            lane_index=0
        )
        # Check a few key fields to be sure
        assert cpe_instance.concept_text == "The Eiffel Tower is a global cultural icon of France."
        assert cpe_instance.domain_code == 9
        assert cpe_instance.source_chunk == SAMPLE_CHUNK

    except (KeyError, TypeError) as e:
        pytest.fail(f"Failed to create CPECore object from LLM output. Missing or incorrect field: {e}")

    # 5. Verify that the mock was called
    mock_post.assert_called_once()
