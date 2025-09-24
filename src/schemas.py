from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Lane = Literal["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"]
Mode = Literal["DENSE", "GRAPH", "HYBRID"]

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=512, description="Query string (1-512 characters)")
    lane: Lane = Field(..., description="Lane: L1_FACTOID, L2_GRAPH, or L3_SYNTH")
    top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return (1-100)")
    lane_index: Optional[int] = Field(default=None, ge=0, le=32767, description="Optional lane index filter (0-32767)")

class SearchItem(BaseModel):
    id: str                         # canonical: cpe_id (stable UUIDv5)
    doc_id: Optional[str]           # upstream document ID (enwiki, etc.)
    score: Optional[float] = None
    why: Optional[str] = None
    concept_text: Optional[str] = None    # hydrated concept text
    tmd_code: Optional[str] = None        # hydrated TMD codes (D.T.M format)
    lane_index: Optional[int] = None      # computed lane index

class SearchResponse(BaseModel):
    lane: Lane
    mode: Mode
    items: List[SearchItem]
    trace_id: Optional[str] = None