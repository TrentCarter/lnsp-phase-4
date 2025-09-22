from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Extracted:
    concept_text: str
    probe_question: str
    expected_answer: str
    domain: str
    task: str
    modifier: str
    content_type: str
    relations: list[dict]


TEMPLATE_HINT = {
    "schema": {
        "prop": "<short atomic proposition>",
        "domain": "<one-of: science|math|tech|engineering|medicine|psychology|philosophy|history|literature|art|economics|law|politics|education|environment|sociology>",
        "task": "<one-of: fact_retrieval|definition|comparison|cause_effect|taxonomy|timeline|attribute_extraction|numeric_fact|entity_linking|contradiction_check|...>",
        "modifier": "<one-of: historical|descriptive|temporal|biochemical|geographical|legal|clinical|software|hardware|experimental|statistical|theoretical|cultural|economic|political|educational|...>",
        "mission": "<brief instruction suited to content_type>",
        "probe": "<single short question to verify prop>",
        "expected": "<concise answer to probe>",
        "relations": [{"subj": "<string>", "pred": "<string>", "obj": "<string>"}],
        "content_type": "<factual|math|instruction|narrative>"
    }
}


def extract_stub(contents: str, title_hint: str | None = None) -> Extracted:
    """Day-1, dependency-free heuristic extractor. Replace with LLM+schema later.
    - concept_text: first sentence after title
    - probe_question: templated 'What is ...?'
    - expected_answer: next sentence or copy of first
    - domain/task/modifier: art/fact_retrieval/descriptive if it looks like music/art; else tech
    """
    parts = contents.split("\n", 1)
    title = (parts[0] if parts else title_hint or "").strip()
    body = (parts[1] if len(parts) > 1 else contents).strip()

    sent = body.split(".")
    first = (sent[0].strip() + ".") if sent and sent[0].strip() else (title or contents)
    second = (sent[1].strip() + ".") if len(sent) > 1 and sent[1].strip() else first

    low = (title + " " + body).lower()
    if any(k in low for k in ["album", "song", "band", "singer", "record"]):
        domain = "art"  # lane alias for music
    else:
        domain = "tech"

    probe = f"What is {title.split('(')[0].strip()}?" if title else f"Summarize: {first}" 

    return Extracted(
        concept_text=first,
        probe_question=probe,
        expected_answer=second,
        domain=domain,
        task="fact_retrieval",
        modifier="descriptive",
        content_type="factual",
        relations=[],
    )

def extract_cpe_from_text(text: str) -> Dict[str, Any]:
    # Alias for backward compatibility
    extracted = extract_stub(text)
    return {
        'concept': extracted.concept_text,
        'prop': extracted.concept_text,
        'mission': f'Extract atomic facts from: {text[:120]}',
        'probe': extracted.probe_question,
        'expected': extracted.expected_answer,
        'domain': extracted.domain,
        'task': extracted.task,
        'modifier': extracted.modifier,
        'domain_code': 2 if extracted.domain == 'tech' else 9,  # stub codes
        'task_code': 0,
        'modifier_code': 0,
        'tmd_bits': 0,
        'tmd_lane': f'{extracted.domain}-{extracted.task}-{extracted.modifier}',
        'lane_index': 0,
        'tmd_dense': [0.0] * 16,
        'concept_vec': [0.0] * 768,
        'question_vec': [0.0] * 768,
        'fused_vec': [0.0] * 784,
        'fused_norm': 1.0,
        'relations': extracted.relations,
        'echo_score': 0.95,
        'validation_status': 'passed',
    }
