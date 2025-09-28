from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, model_validator, ConfigDict


class ChunkPosition(BaseModel):
    doc_id: str
    start: int
    end: int


class RelationTriple(BaseModel):
    subj: str
    pred: str
    obj: str

Lane = Literal["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"]
Mode = Literal["DENSE", "GRAPH", "HYBRID"]

class CPESH(BaseModel):
    """Authoritative CPESH payload aligned with the PRD."""

    cpe_id: Optional[str] = None

    # Core concept triple (aliases maintain backward compatibility)
    concept: Optional[str] = Field(default=None, alias="concept_text")
    probe: Optional[str] = Field(default=None, alias="probe_question")
    expected: Optional[str] = Field(default=None, alias="expected_answer")
    soft_negative: Optional[str] = None
    hard_negative: Optional[str] = None

    # Mission/source metadata
    mission_text: Optional[str] = None
    source_chunk: Optional[str] = None
    dataset_source: Optional[str] = None
    content_type: Optional[str] = None
    chunk_position: Optional[ChunkPosition] = None
    relations_text: Optional[List[RelationTriple]] = None

    # Routing / lane metadata
    tmd_bits: Optional[int] = None
    tmd_lane: Optional[str] = None
    lane_index: Optional[int] = None

    # Quality & diagnostics
    quality: Optional[float] = None
    echo_score: Optional[float] = None
    insufficient_evidence: bool = False
    soft_sim: Optional[float] = None  # cosine vs query (only if cpesh_mode=full)
    hard_sim: Optional[float] = None  # cosine vs query (only if cpesh_mode=full)

    # Audit metadata
    created_at: Optional[str] = None  # ISO8601 timestamp when this CPESH was first created
    last_accessed: Optional[str] = None  # ISO8601 timestamp when this CPESH was last accessed
    access_count: int = 0

    model_config = ConfigDict(populate_by_name=True)

    @property
    def concept_text(self) -> Optional[str]:
        return self.concept

    @concept_text.setter
    def concept_text(self, value: Optional[str]) -> None:
        object.__setattr__(self, "concept", value)

    @property
    def probe_question(self) -> Optional[str]:
        return self.probe

    @probe_question.setter
    def probe_question(self, value: Optional[str]) -> None:
        object.__setattr__(self, "probe", value)

    @property
    def expected_answer(self) -> Optional[str]:
        return self.expected

    @expected_answer.setter
    def expected_answer(self, value: Optional[str]) -> None:
        object.__setattr__(self, "expected", value)

class CPESHDiagnostics(BaseModel):
    concept: Optional[str] = None
    probe: Optional[str] = None
    expected: Optional[str] = None
    soft_negative: Optional[str] = None
    hard_negative: Optional[str] = None
    soft_sim: Optional[float] = None
    hard_sim: Optional[float] = None
    created_at: Optional[str] = None
    last_accessed: Optional[str] = None
    quality: Optional[float] = None
    echo_score: Optional[float] = None
    insufficient_evidence: Optional[bool] = None

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=512, description="Query string (1-512 characters)")
    lane: Optional[Lane] = Field(default=None, description="Lane: L1_FACTOID, L2_GRAPH, or L3_SYNTH")
    top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return (1-100)")
    lane_index: Optional[int] = Field(default=None, ge=0, le=32767, description="Optional lane index filter (0-32767)")
    return_cpesh: Optional[bool] = Field(default=False, description="Include per-item CPESH object")
    cpesh_mode: Optional[Literal["lite","full"]] = Field(default="lite", description="CPESH detail level")
    cpesh_k: Optional[int] = Field(default=None, ge=0, le=50, description="Max hits to CPESH-enrich (overrides env)")
    compact: Optional[bool] = Field(default=False, description="Return compact hit objects (id,score,tmd,lane,cpesh)")

    @model_validator(mode="after")
    def require_one_lane(self):
        if self.lane is None and self.lane_index is None:
            # Default to L1_FACTOID if nothing provided
            self.lane = "L1_FACTOID"
        return self

class SearchItem(BaseModel):
    id: str                         # canonical: cpe_id (stable UUIDv5)
    doc_id: Optional[str]           # upstream document ID (enwiki, etc.)
    score: Optional[float] = None
    why: Optional[str] = None
    concept_text: Optional[str] = None    # hydrated concept text
    tmd_code: Optional[str] = None        # hydrated TMD codes (D.T.M format)
    lane_index: Optional[int] = None      # computed lane index
    quality: Optional[float] = None       # quality score from IQS system
    final_score: Optional[float] = None   # blended score (cosine + quality)
    cpesh: Optional[CPESH] = None
    word_count: Optional[int] = None      # number of words in the chunk
    tmd_confidence: Optional[float] = None # confidence of the TMD extraction

class SearchResponse(BaseModel):
    lane: Optional[Lane]
    mode: Mode
    items: List[SearchItem]
    trace_id: Optional[str] = None
    diagnostics: Optional[CPESHDiagnostics] = None
    insufficient_evidence: Optional[bool] = None
    quality_warning: Optional[bool] = None
