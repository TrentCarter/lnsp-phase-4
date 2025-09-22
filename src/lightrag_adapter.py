from typing import List, Optional, Dict, Any
from lightrag import LightRAG, QueryParam
from .enums import RetrievalMode, LightRAGMode, Lane
from .vectorizer import EmbeddingBackend
from .faiss_index import build_ivf_flat_cosine

class LightRagFacade:
    def __init__(self, data_dir: str, mode: LightRAGMode, embed_model_id: str):
        self.embedder = EmbeddingBackend(embed_model_id)
        # For now, we'll assume vectors are stored separately
        # self.faiss = FaissIVFIndex.open(data_dir)  # dense side
        self.rag = LightRAG(
            working_dir=data_dir,
            llm_model_func=None,  # we only use KG + retrieval
        )
        self.mode = mode

    def index_edges(self, edges: List[Dict[str, Any]]):
        # edges: [{"src":"X","rel":"MENTIONS","dst":"Y"}]
        for e in edges:
            self.rag.insert(e["src"] + " " + e["rel"] + " " + e["dst"])

    def dense_search(self, q: str, k: int = 8):
        # For now, return placeholder - would need vector search integration
        return []

    def hybrid_search(self, q: str, k: int = 8, hops: int = 2):
        # Use LightRAG's built-in search
        result = self.rag.query(q, param=QueryParam(mode="hybrid", max_token_for_text_unit=1000))
        return {"dense": [], "kg": result}

def get_lane_mode(lane: Lane) -> RetrievalMode:
    return {Lane.L1_FACTOID: RetrievalMode.DENSE,
            Lane.L2_GRAPH: RetrievalMode.GRAPH}.get(lane, RetrievalMode.HYBRID)
