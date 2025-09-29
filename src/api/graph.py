# src/api/graph.py
from typing import List, Optional, Dict, Any
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ENABLED = os.getenv("LNSP_GRAPHRAG_ENABLED", "0") == "1"
router = APIRouter()

class GraphSearchIn(BaseModel):
    q: Optional[str] = None
    lane: Optional[int] = None
    top_k: int = 10
    seed_ids: Optional[List[str]] = None  # cpe_ids

class GraphHopIn(BaseModel):
    node_id: str  # cpe_id
    max_hops: int = 2
    top_k: int = 10
    lane: Optional[int] = None

def _require_enabled():
    if not ENABLED:
        raise HTTPException(status_code=501, detail="GraphRAG disabled (set LNSP_GRAPHRAG_ENABLED=1)")

@router.post("/graph/search")
def graph_search(body: GraphSearchIn):
    _require_enabled()
    try:
        from ..db_neo4j import Neo4jDB
    except Exception as e:
        raise HTTPException(500, f"Neo4j module unavailable: {e}")

    db = Neo4jDB()
    # priority: seed_ids → FTS(q) → empty
    if body.seed_ids:
        rows = db.search_by_seed_ids(seed_ids=body.seed_ids, lane=body.lane, top_k=body.top_k)
    elif body.q:
        rows = db.search_fulltext(q=body.q, lane=body.lane, top_k=body.top_k)
    else:
        rows = []
    return rows

@router.post("/graph/hop")
def graph_hop(body: GraphHopIn):
    _require_enabled()
    try:
        from ..db_neo4j import Neo4jDB
    except Exception as e:
        raise HTTPException(500, f"Neo4j module unavailable: {e}")
    db = Neo4jDB()
    return db.expand_hops(seed=body.node_id, max_hops=body.max_hops, top_k=body.top_k, lane=body.lane)

@router.get("/graph/health")
def graph_health():
    _require_enabled()
    try:
        from ..db_neo4j import Neo4jDB
        db = Neo4jDB()
        return db.graph_health()
    except Exception as e:
        raise HTTPException(500, f"Neo4j health error: {e}")