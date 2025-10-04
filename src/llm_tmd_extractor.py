#!/usr/bin/env python3
"""
LLM-based TMD Extractor using Llama 3.1:8b

Uses local Llama to extract Domain/Task/Modifier codes from concept text.
Replaces pattern-based extraction (src/tmd_extractor_v2.py) which assigns
incorrect metadata (e.g., "software" → Biochemical modifier).
"""
import os
import json
import re
from typing import Dict, Optional
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.llm.local_llama_client import call_local_llama_simple
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False


# TMD Schema from docs/PRDs/TMD-Schema.md
TMD_PROMPT_TEMPLATE = """You are a metadata extraction expert. Analyze this concept and assign ONE code from each category.

CONCEPT: "{text}"

=== DOMAINS (choose ONE, 0-15) ===
0=Science, 1=Mathematics, 2=Technology, 3=Engineering, 4=Medicine,
5=Psychology, 6=Philosophy, 7=History, 8=Literature, 9=Art,
10=Economics, 11=Law, 12=Politics, 13=Education, 14=Environment, 15=Software

=== TASKS (choose ONE, 0-31) ===
0=Fact Retrieval, 1=Definition Matching, 2=Analogical Reasoning, 3=Causal Inference,
4=Classification, 5=Entity Recognition, 6=Relationship Extraction, 7=Schema Adherence,
8=Summarization, 9=Paraphrasing, 10=Translation, 11=Sentiment Analysis,
12=Argument Evaluation, 13=Hypothesis Testing, 14=Code Generation, 15=Function Calling,
16=Mathematical Proof, 17=Diagram Interpretation, 18=Temporal Reasoning, 19=Spatial Reasoning,
20=Ethical Evaluation, 21=Policy Recommendation, 22=Roleplay Simulation, 23=Creative Writing,
24=Instruction Following, 25=Error Detection, 26=Output Repair, 27=Question Generation,
28=Conceptual Mapping, 29=Knowledge Distillation, 30=Tool Use, 31=Prompt Completion

=== MODIFIERS (choose ONE, 0-63) ===
0=Biochemical, 1=Evolutionary, 2=Computational, 3=Logical, 4=Ethical,
5=Historical, 6=Legal, 7=Philosophical, 8=Emotional, 9=Technical,
10=Creative, 11=Abstract, 12=Concrete, 13=Visual, 14=Auditory,
15=Spatial, 16=Temporal, 17=Quantitative, 18=Qualitative, 19=Procedural,
20=Declarative, 21=Comparative, 22=Analogical, 23=Causal, 24=Hypothetical,
25=Experimental, 26=Narrative, 27=Descriptive, 28=Prescriptive, 29=Diagnostic,
30=Predictive, 31=Reflective, 32=Strategic, 33=Tactical, 34=Symbolic,
35=Functional, 36=Structural, 37=Semantic, 38=Syntactic, 39=Pragmatic,
40=Normative, 41=Statistical, 42=Probabilistic, 43=Deterministic, 44=Stochastic,
45=Modular, 46=Hierarchical, 47=Distributed, 48=Localized, 49=Global,
50=Contextual, 51=Generalized, 52=Specialized, 53=Interdisciplinary, 54=Multimodal,
55=Ontological, 56=Epistemic, 57=Analog-sensitive, 58=Schema-bound, 59=Role-based,
60=Feedback-driven, 61=Entailment-aware, 62=Alignment-focused, 63=Compression-optimized

CRITICAL INSTRUCTIONS:
1. Analyze the concept semantically
2. Choose the BEST matching codes (not defaults!)
3. Return ONLY THREE NUMBERS separated by commas
4. DO NOT include explanations, domain names, or any other text
5. Example valid output: "15,14,9"
6. Example INVALID output: "Software, 14, Technical" or "Domain: 15..."

OUTPUT ONLY THREE NUMBERS:"""


def extract_tmd_with_llm(
    text: str,
    llm_endpoint: Optional[str] = None,
    llm_model: str = "llama3.1:8b"
) -> Dict[str, int]:
    """
    Extract TMD codes using LLM.

    Args:
        text: Concept text to analyze
        llm_endpoint: LLM endpoint (default: LNSP_LLM_ENDPOINT env var or http://localhost:11434)
        llm_model: LLM model name (default: llama3.1:8b)

    Returns:
        dict with domain_code, task_code, modifier_code (0-based integers)

    Example:
        >>> extract_tmd_with_llm("oxidoreductase activity")
        {'domain_code': 4, 'task_code': 5, 'modifier_code': 0}  # Medicine/Entity Recognition/Biochemical
    """
    if not HAS_LLAMA:
        raise RuntimeError("LocalLlamaClient not available. Install dependencies.")

    # Set environment variables for LLM client
    if llm_endpoint:
        os.environ["LNSP_LLM_ENDPOINT"] = llm_endpoint
    if llm_model:
        os.environ["LNSP_LLM_MODEL"] = llm_model

    # Build prompt
    prompt = TMD_PROMPT_TEMPLATE.format(text=text)

    # Call LLM using simple interface
    try:
        response = call_local_llama_simple(prompt)
    except Exception as e:
        # Fallback to defaults on error
        print(f"LLM extraction failed for '{text}': {e}")
        return {'domain_code': 0, 'task_code': 0, 'modifier_code': 0}

    # Parse response
    # Expected format: "domain,task,modifier" or "domain, task, modifier" (with optional spaces)
    match = re.search(r'(\d+),\s*(\d+),\s*(\d+)', response)
    if not match:
        print(f"Failed to parse LLM response for '{text}': {response}")
        return {'domain_code': 0, 'task_code': 0, 'modifier_code': 0}

    domain = int(match.group(1))
    task = int(match.group(2))
    modifier = int(match.group(3))

    # Validate ranges
    if not (0 <= domain <= 15):
        print(f"Invalid domain {domain} for '{text}', clamping")
        domain = max(0, min(15, domain))
    if not (0 <= task <= 31):
        print(f"Invalid task {task} for '{text}', clamping")
        task = max(0, min(31, task))
    if not (0 <= modifier <= 63):
        print(f"Invalid modifier {modifier} for '{text}', clamping")
        modifier = max(0, min(63, modifier))

    return {
        'domain_code': domain,
        'task_code': task,
        'modifier_code': modifier
    }


# Human-readable names for validation
DOMAIN_NAMES = [
    "Science", "Mathematics", "Technology", "Engineering", "Medicine",
    "Psychology", "Philosophy", "History", "Literature", "Art",
    "Economics", "Law", "Politics", "Education", "Environment", "Software"
]

TASK_NAMES = [
    "Fact Retrieval", "Definition Matching", "Analogical Reasoning", "Causal Inference",
    "Classification", "Entity Recognition", "Relationship Extraction", "Schema Adherence",
    "Summarization", "Paraphrasing", "Translation", "Sentiment Analysis",
    "Argument Evaluation", "Hypothesis Testing", "Code Generation", "Function Calling",
    "Mathematical Proof", "Diagram Interpretation", "Temporal Reasoning", "Spatial Reasoning",
    "Ethical Evaluation", "Policy Recommendation", "Roleplay Simulation", "Creative Writing",
    "Instruction Following", "Error Detection", "Output Repair", "Question Generation",
    "Conceptual Mapping", "Knowledge Distillation", "Tool Use", "Prompt Completion"
]

MODIFIER_NAMES = [
    "Biochemical", "Evolutionary", "Computational", "Logical", "Ethical",
    "Historical", "Legal", "Philosophical", "Emotional", "Technical",
    "Creative", "Abstract", "Concrete", "Visual", "Auditory",
    "Spatial", "Temporal", "Quantitative", "Qualitative", "Procedural",
    "Declarative", "Comparative", "Analogical", "Causal", "Hypothetical",
    "Experimental", "Narrative", "Descriptive", "Prescriptive", "Diagnostic",
    "Predictive", "Reflective", "Strategic", "Tactical", "Symbolic",
    "Functional", "Structural", "Semantic", "Syntactic", "Pragmatic",
    "Normative", "Statistical", "Probabilistic", "Deterministic", "Stochastic",
    "Modular", "Hierarchical", "Distributed", "Localized", "Global",
    "Contextual", "Generalized", "Specialized", "Interdisciplinary", "Multimodal",
    "Ontological", "Epistemic", "Analog-sensitive", "Schema-bound", "Role-based",
    "Feedback-driven", "Entailment-aware", "Alignment-focused", "Compression-optimized"
]


def format_tmd_result(text: str, tmd: Dict[str, int]) -> str:
    """Format TMD extraction result for human review."""
    d = tmd['domain_code']
    t = tmd['task_code']
    m = tmd['modifier_code']

    domain_name = DOMAIN_NAMES[d] if 0 <= d < len(DOMAIN_NAMES) else f"INVALID({d})"
    task_name = TASK_NAMES[t] if 0 <= t < len(TASK_NAMES) else f"INVALID({t})"
    modifier_name = MODIFIER_NAMES[m] if 0 <= m < len(MODIFIER_NAMES) else f"INVALID({m})"

    return f"""
Concept: "{text}"
  Domain   ({d:2d}): {domain_name}
  Task     ({t:2d}): {task_name}
  Modifier ({m:2d}): {modifier_name}
"""


if __name__ == "__main__":
    # Test on sample concepts
    test_concepts = [
        "oxidoreductase activity",
        "material entity",
        "software",
        "Gene Ontology",
        "biochemical process",
        "Python programming language",
        "machine learning algorithm",
        "cardiac arrest",
        "World War II",
        "cellular respiration"
    ]

    print("=" * 70)
    print("LLM-Based TMD Extraction Test")
    print("=" * 70)
    print()

    for concept in test_concepts:
        tmd = extract_tmd_with_llm(concept)
        print(format_tmd_result(concept, tmd))
        print("-" * 70)
