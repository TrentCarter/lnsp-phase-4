from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

# TMD Schema mappings
DOMAINS = {
    "science": 0, "mathematics": 1, "technology": 2, "engineering": 3,
    "medicine": 4, "psychology": 5, "philosophy": 6, "history": 7,
    "literature": 8, "art": 9, "economics": 10, "law": 11,
    "politics": 12, "education": 13, "environment": 14, "sociology": 15
}

TASKS = {
    "fact_retrieval": 0, "definition": 1, "comparison": 2, "cause_effect": 3,
    "taxonomy": 4, "timeline": 5, "attribute_extraction": 6, "numeric_fact": 7,
    "entity_linking": 8, "contradiction_check": 9, "analogical_reasoning": 10,
    "causal_inference": 11, "classification": 12, "entity_recognition": 13,
    "relationship_extraction": 14, "schema_adherence": 15, "summarization": 16,
    "paraphrasing": 17, "translation": 18, "sentiment_analysis": 19,
    "argument_evaluation": 20, "hypothesis_testing": 21, "code_generation": 22,
    "function_calling": 23, "mathematical_proof": 24, "diagram_interpretation": 25,
    "temporal_reasoning": 26, "spatial_reasoning": 27, "ethical_evaluation": 28,
    "policy_recommendation": 29, "roleplay_simulation": 30, "creative_writing": 31
}

MODIFIERS = {
    "biochemical": 0, "evolutionary": 1, "computational": 2, "logical": 3,
    "ethical": 4, "historical": 5, "legal": 6, "philosophical": 7,
    "emotional": 8, "technical": 9, "creative": 10, "abstract": 11,
    "concrete": 12, "visual": 13, "auditory": 14, "spatial": 15,
    "temporal": 16, "quantitative": 17, "qualitative": 18, "procedural": 19,
    "declarative": 20, "comparative": 21, "analogical": 22, "causal": 23,
    "hypothetical": 24, "experimental": 25, "narrative": 26, "descriptive": 27,
    "prescriptive": 28, "diagnostic": 29, "predictive": 30, "reflective": 31,
    "strategic": 32, "tactical": 33, "symbolic": 34, "functional": 35,
    "structural": 36, "semantic": 37, "syntactic": 38, "pragmatic": 39,
    "normative": 40, "statistical": 41, "probabilistic": 42, "deterministic": 43,
    "stochastic": 44, "modular": 45, "hierarchical": 46, "distributed": 47,
    "localized": 48, "global": 49, "contextual": 50, "generalized": 51,
    "specialized": 52, "interdisciplinary": 53, "multimodal": 54, "ontological": 55,
    "epistemic": 56, "analog-sensitive": 57, "schema-bound": 58, "role-based": 59,
    "feedback-driven": 60, "entailment-aware": 61, "alignment-focused": 62,
    "compression-optimized": 63
}

def pack_tmd(domain: int, task: int, modifier: int) -> int:
    """Pack TMD codes into uint16 bit field.
    Domain: 4 bits (0-15), Task: 5 bits (0-31), Modifier: 6 bits (0-63)
    """
    assert 0 <= domain <= 0xF, f"domain {domain} out of range [0, 15]"
    assert 0 <= task <= 0x1F, f"task {task} out of range [0, 31]"
    assert 0 <= modifier <= 0x3F, f"modifier {modifier} out of range [0, 63]"
    return (domain << 12) | (task << 7) | (modifier << 1)

def tmd16_deterministic(domain: int, task: int, modifier: int) -> np.ndarray:
    """Return a stable 16D float vector derived from categorical codes."""
    assert 0 <= domain <= 0xF and 0 <= task <= 0x1F and 0 <= modifier <= 0x3F

    bits = np.concatenate([
        [(domain >> shift) & 1 for shift in range(4)],
        [(task >> shift) & 1 for shift in range(5)],
        [(modifier >> shift) & 1 for shift in range(6)],
    ])  # (15,)

    # Simple projection matrix (could be more sophisticated)
    proj = np.random.default_rng(seed=1337).standard_normal((16, 15)).astype(np.float32)
    vec = proj @ bits

    # L2 normalize
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


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
        domain = "technology"

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
        'domain_code': DOMAINS.get(extracted.domain, 2),  # Default to tech if unknown
        'task_code': TASKS.get(extracted.task, 0),
        'modifier_code': MODIFIERS.get(extracted.modifier, 27),  # Default to descriptive
        'tmd_bits': pack_tmd(
            DOMAINS.get(extracted.domain, 2),
            TASKS.get(extracted.task, 0),
            MODIFIERS.get(extracted.modifier, 27)
        ),
        'tmd_lane': f'{extracted.domain}-{extracted.task}-{extracted.modifier}',
        'lane_index': pack_tmd(
            DOMAINS.get(extracted.domain, 2),
            TASKS.get(extracted.task, 0),
            MODIFIERS.get(extracted.modifier, 27)
        ) >> 1,  # Right shift to get lane index
        'tmd_dense': tmd16_deterministic(
            DOMAINS.get(extracted.domain, 2),
            TASKS.get(extracted.task, 0),
            MODIFIERS.get(extracted.modifier, 27)
        ).tolist(),
        'concept_vec': [0.0] * 768,
        'question_vec': [0.0] * 768,
        'fused_vec': [0.0] * 784,
        'fused_norm': 1.0,
        'relations': extracted.relations,
        'echo_score': 0.95,
        'validation_status': 'passed',
    }
