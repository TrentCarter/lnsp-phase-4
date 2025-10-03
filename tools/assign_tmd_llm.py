#!/usr/bin/env python3
"""Assign TMD codes for ontology concepts using local Llama and TMD schema."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repo root is on path so we can import local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.local_llama_client import LocalLlamaClient
from src.tmd_encoder import tmd16_deterministic

DOMAINS: List[str] = [
    "Science",
    "Mathematics",
    "Technology",
    "Engineering",
    "Medicine",
    "Psychology",
    "Philosophy",
    "History",
    "Literature",
    "Art",
    "Economics",
    "Law",
    "Politics",
    "Education",
    "Environment",
    "Sociology",
]

TASKS: List[str] = [
    "Fact Retrieval",
    "Definition Matching",
    "Analogical Reasoning",
    "Causal Inference",
    "Classification",
    "Entity Recognition",
    "Relationship Extraction",
    "Schema Adherence",
    "Summarization",
    "Paraphrasing",
    "Translation",
    "Sentiment Analysis",
    "Argument Evaluation",
    "Hypothesis Testing",
    "Code Generation",
    "Function Calling",
    "Mathematical Proof",
    "Diagram Interpretation",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Ethical Evaluation",
    "Policy Recommendation",
    "Roleplay Simulation",
    "Creative Writing",
    "Instruction Following",
    "Error Detection",
    "Output Repair",
    "Question Generation",
    "Conceptual Mapping",
    "Knowledge Distillation",
    "Tool Use",
    "Prompt Completion",
]

MODIFIERS: List[str] = [
    "Biochemical",
    "Evolutionary",
    "Computational",
    "Logical",
    "Ethical",
    "Historical",
    "Legal",
    "Philosophical",
    "Emotional",
    "Technical",
    "Creative",
    "Abstract",
    "Concrete",
    "Visual",
    "Auditory",
    "Spatial",
    "Temporal",
    "Quantitative",
    "Qualitative",
    "Procedural",
    "Declarative",
    "Comparative",
    "Analogical",
    "Causal",
    "Hypothetical",
    "Experimental",
    "Narrative",
    "Descriptive",
    "Prescriptive",
    "Diagnostic",
    "Predictive",
    "Reflective",
    "Strategic",
    "Tactical",
    "Symbolic",
    "Functional",
    "Structural",
    "Semantic",
    "Syntactic",
    "Pragmatic",
    "Normative",
    "Statistical",
    "Probabilistic",
    "Deterministic",
    "Stochastic",
    "Modular",
    "Hierarchical",
    "Distributed",
    "Localized",
    "Global",
    "Contextual",
    "Generalized",
    "Specialized",
    "Interdisciplinary",
    "Multimodal",
    "Ontological",
    "Epistemic",
    "Analog-sensitive",
    "Schema-bound",
    "Role-based",
    "Feedback-driven",
    "Entailment-aware",
    "Alignment-focused",
    "Compression-optimized",
]

DOMAIN_TO_INDEX = {name.lower(): idx for idx, name in enumerate(DOMAINS)}
TASK_TO_INDEX = {name.lower(): idx for idx, name in enumerate(TASKS)}
MODIFIER_TO_INDEX = {name.lower(): idx for idx, name in enumerate(MODIFIERS)}


def load_chain_context(files: Sequence[Path]) -> Dict[str, List[List[str]]]:
    """Map lowercase leaf concept → list of concept chains."""
    mapping: Dict[str, List[List[str]]] = defaultdict(list)
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                concepts = data.get("concepts")
                if not concepts:
                    continue
                leaf = str(concepts[-1]).lower()
                mapping[leaf].append([str(c) for c in concepts])
    return mapping


def build_prompt(concept: str, context_paths: Optional[List[List[str]]]) -> str:
    context_str = "; ".join(" -> ".join(path) for path in context_paths) if context_paths else "N/A"

    domain_menu = "\n".join(f"- {idx}: {name}" for idx, name in enumerate(DOMAINS))
    task_menu = "\n".join(f"- {idx}: {name}" for idx, name in enumerate(TASKS))
    modifier_menu = "\n".join(f"- {idx}: {name}" for idx, name in enumerate(MODIFIERS))

    return f"""You are a classification assistant for the LNSP Task-Modifier-Domain (TMD) schema.
Choose the Domain, Task, and Modifier that best describe the concept using ONLY the options below.
Return valid JSON with keys: domain, task, modifier, confidence (0-1 float), rationale (<=2 sentences).
The values for domain/task/modifier MUST exactly match one of the listed names (case sensitive).

Concept: {concept}
Ontology context (root→leaf path): {context_str}

Domains:\n{domain_menu}
\nTasks:\n{task_menu}
\nModifiers:\n{modifier_menu}
"""


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def map_label(value, names: Sequence[str], mapping: Dict[str, int], label_type: str, concept: str) -> Tuple[int, str]:
    """Resolve a domain/task/modifier label (name or index) to its integer code."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(value)
        if 0 <= idx < len(names):
            return idx, names[idx]
        raise ValueError(f"{label_type.title()} index {idx} out of range for concept '{concept}'")

    if not isinstance(value, str):
        raise ValueError(f"{label_type.title()} value '{value}' for concept '{concept}' must be string or int")

    key = normalize_label(value)
    for candidate, idx in mapping.items():
        if key == candidate.replace("_", "-"):
            return idx, names[idx]

    raw = value.strip().lower()
    if raw in mapping:
        idx = mapping[raw]
        return idx, names[idx]

    # attempt to match ignoring hyphen/space differences
    sanitized = raw.replace("-", " ")
    for candidate, idx in mapping.items():
        if sanitized == candidate.replace("-", " "):
            return idx, names[idx]

    raise ValueError(f"Unmapped {label_type} label '{value}' for concept '{concept}'")


def classify_concept(client: LocalLlamaClient, concept: str, context_paths: Optional[List[List[str]]],
                     retries: int = 3, delay_s: float = 1.0) -> Dict[str, str]:
    prompt = build_prompt(concept, context_paths)
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            result = client.complete_json(prompt, timeout_s=30)
            if not isinstance(result, dict):
                raise ValueError(f"Unexpected response type: {type(result)}")
            if not all(k in result for k in ("domain", "task", "modifier")):
                raise ValueError(f"Missing keys in response: {result}")
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(delay_s)
            continue
    raise RuntimeError(f"Failed to classify concept '{concept}': {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign TMD codes via local Llama.")
    parser.add_argument("input_npz", type=Path, help="Input NPZ file (with concept_vecs and tmd data)")
    parser.add_argument("output_npz", type=Path, help="Output NPZ path with updated TMD codes")
    parser.add_argument("--assignments-jsonl", type=Path, default=Path("outputs/tmd_llm_assignments.jsonl"),
                        help="Where to store assignment records (JSONL)")
    parser.add_argument("--chain-files", nargs="*", type=Path,
                        default=[
                            Path("artifacts/ontology_chains/swo_chains.jsonl"),
                            Path("artifacts/ontology_chains/go_chains.jsonl"),
                            Path("artifacts/ontology_chains/dbpedia_chains.jsonl"),
                        ],
                        help="Ontology chain files used for context lookup")
    parser.add_argument("--target-domain", type=int, default=1,
                        help="Domain code to treat as placeholder (default: 1)")
    parser.add_argument("--target-task", type=int, default=1,
                        help="Task code to treat as placeholder (default: 1)")
    parser.add_argument("--target-modifier", type=int, default=1,
                        help="Modifier code to treat as placeholder (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of concepts to process (debug)")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between LLM calls")
    args = parser.parse_args()

    npz_path = args.input_npz
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    vectors = data["vectors"]
    concept_vecs = data.get("concept_vecs", vectors)
    doc_ids = data["doc_ids"]
    cpe_ids = data["cpe_ids"]
    concept_texts = data["concept_texts"]
    tmd_dense = data["tmd_dense"].copy()
    tmd_codes = data["tmd_codes"].copy()

    target_mask = [
        (int(dom) == args.target_domain and int(tsk) == args.target_task and int(mod) == args.target_modifier)
        for dom, tsk, mod in tmd_codes
    ]
    target_indices = [idx for idx, flag in enumerate(target_mask) if flag]
    if args.limit is not None:
        target_indices = target_indices[: args.limit]

    if not target_indices:
        print("No concepts matched the placeholder TMD codes; nothing to do.")
        return

    print(f"Found {len(target_indices)} concept(s) with placeholder TMD codes.")

    chain_map = load_chain_context(args.chain_files)
    client = LocalLlamaClient()

    assignments_jsonl = args.assignments_jsonl
    assignments_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    cache: Dict[str, Dict[str, str]] = {}

    with assignments_jsonl.open("w", encoding="utf-8") as out_f:
        for idx in target_indices:
            concept = str(concept_texts[idx])
            key = concept.lower()
            context_paths = chain_map.get(key)

            if key in cache:
                result = cache[key]
            else:
                result = classify_concept(client, concept, context_paths)
                cache[key] = result
                time.sleep(max(args.sleep, 0))

            try:
                domain_idx, domain_label = map_label(result["domain"], DOMAINS, DOMAIN_TO_INDEX, "domain", concept)
                task_idx, task_label = map_label(result["task"], TASKS, TASK_TO_INDEX, "task", concept)
                modifier_idx, modifier_label = map_label(result["modifier"], MODIFIERS, MODIFIER_TO_INDEX, "modifier", concept)
            except ValueError as exc:
                raise RuntimeError(f"Invalid labels for concept '{concept}': {exc}\nLLM output: {result}") from exc

            tmd_codes[idx] = np.array([domain_idx, task_idx, modifier_idx], dtype=np.int16)
            tmd_vectors = tmd16_deterministic(domain_idx, task_idx, modifier_idx)
            tmd_dense[idx] = tmd_vectors.astype(np.float32)

            record = {
                "index": idx,
                "concept": concept,
                "doc_id": str(doc_ids[idx]),
                "cpe_id": str(cpe_ids[idx]),
                "domain": domain_label,
                "task": task_label,
                "modifier": modifier_label,
                "confidence": result.get("confidence"),
                "rationale": result.get("rationale"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            processed += 1
            if processed % 50 == 0 or processed == len(target_indices):
                print(f"Processed {processed}/{len(target_indices)} concepts...")

    np.savez(
        args.output_npz,
        vectors=vectors,
        concept_vecs=concept_vecs,
        tmd_dense=tmd_dense.astype(np.float32),
        tmd_codes=tmd_codes.astype(np.int16),
        cpe_ids=cpe_ids,
        concept_texts=concept_texts,
        doc_ids=doc_ids,
    )

    print(f"Saved updated NPZ to {args.output_npz}")
    print(f"Assignments written to {assignments_jsonl}")


if __name__ == "__main__":
    main()
