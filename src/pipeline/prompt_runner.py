from __future__ import annotations
import json
from typing import Dict, Any, List
import numpy as np

try:
    from ..tmd_encoder import pack_tmd, lane_index_from_bits, tmd16_deterministic
    from ..vectorizer import EmbeddingBackend
except ImportError:
    # Handle when run as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tmd_encoder import pack_tmd, lane_index_from_bits, tmd16_deterministic
    from vectorizer import EmbeddingBackend


class PromptRunner:
    """Stub prompt runner for extraction. In real pipeline, this calls LLM + Outlines guided JSON."""

    def __init__(self):
        self.embedder = EmbeddingBackend()

    def run_prompt_extraction(self, text: str, content_type: str = "factual") -> Dict[str, Any]:
        """Minimal placeholder extraction based on content type."""
        # Simple heuristics for demo
        if "album" in text.lower():
            domain, task, modifier = "art", "fact_retrieval", "historical"
            domain_code, task_code, modifier_code = 9, 0, 5  # art, fact_retrieval, historical
        elif "singer" in text.lower():
            domain, task, modifier = "art", "fact_retrieval", "descriptive"
            domain_code, task_code, modifier_code = 9, 0, 27  # art, fact_retrieval, descriptive
        elif "released" in text.lower():
            domain, task, modifier = "art", "fact_retrieval", "temporal"
            domain_code, task_code, modifier_code = 9, 0, 16  # art, fact_retrieval, temporal
        else:
            domain, task, modifier = "science", "fact_retrieval", "factual"
            domain_code, task_code, modifier_code = 0, 0, 0  # fallback

        # Pack TMD bits
        tmd_bits = pack_tmd(domain_code, task_code, modifier_code)
        lane_index = lane_index_from_bits(tmd_bits)
        tmd_lane = f"{domain}-{task}-{modifier}"

        # Generate embeddings
        concept_text = text.split(".")[0] + "." if "." in text else text  # First sentence
        concept_vec = self.embedder.encode([concept_text])[0]
        question_vec = self.embedder.encode([f"What is the main fact in: {concept_text}"])[0]

        # TMD dense vector (deterministic projection)
        tmd_dense = tmd16_deterministic(domain_code, task_code, modifier_code)

        return {
            "concept": concept_text,
            "domain": domain,
            "task": task,
            "modifier": modifier,
            "domain_code": domain_code,
            "task_code": task_code,
            "modifier_code": modifier_code,
            "mission": f"Extract atomic facts from: {text}",
            "probe": f"What is the main fact stated in: {concept_text}",
            "expected": concept_text,
            "relations": [],  # placeholder
            "tmd_bits": tmd_bits,
            "tmd_lane": tmd_lane,
            "lane_index": lane_index,
            "concept_vec": concept_vec.tolist(),
            "question_vec": question_vec.tolist(),
            "tmd_dense": tmd_dense.tolist(),
            "echo_score": 0.95,
            "validation_status": "passed"
        }
