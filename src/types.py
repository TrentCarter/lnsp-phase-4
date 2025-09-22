from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4


@dataclass
class CPECore:
    cpe_id: UUID
    mission_text: str
    source_chunk: str
    concept_text: str
    probe_question: str
    expected_answer: str
    domain_code: int      # 0..15
    task_code: int        # 0..31
    modifier_code: int    # 0..63
    content_type: str     # 'factual'|'math'|'instruction'|'narrative'
    dataset_source: str
    chunk_position: Dict[str, Any]   # {"doc_id": str, "start": int, "end": int}
    relations_text: List[Dict[str, str]]  # [{"subj","pred","obj"}]
    tmd_bits: int
    tmd_lane: str
    lane_index: int
    echo_score: Optional[float] = None
    validation_status: str = "pending"  # 'passed'|'failed'|'pending'


@dataclass
class CPEVectors:
    cpe_id: UUID
    fused_vec: List[float]            # 784
    question_vec: Optional[List[float]]  # 768
    # Optional (FULL mode)
    concept_vec: Optional[List[float]] = None   # 768
    tmd_dense: Optional[List[float]] = None     # 16
    fused_norm: Optional[float] = None
