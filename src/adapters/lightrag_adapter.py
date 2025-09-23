from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from ..enums import RetrievalMode, Status

@dataclass
class LightRAGConfig:
    working_dir: str = "artifacts/lrag"
    top_k: int = 5

class LightRAGAdapter:
    def __init__(self, cfg: LightRAGConfig):
        self.cfg = cfg
        # lazy import to keep core path clean if LRAG not installed
        try:
            from lightrag_hku.api import LightRAG  # type: ignore
            self._LightRAG = LightRAG
        except Exception as e:
            self._LightRAG = None
            self._err = e

    def available(self) -> bool:
        return self._LightRAG is not None

    def index_entities(self, entities_edges_jsonl: str) -> Status:
        if not self.available():
            return Status.FAIL
        # Implement: feed entities/relations to LRAG index in self.cfg.working_dir
        return Status.OK

    def query(self, q: str, k: Optional[int] = None) -> Dict[str, Any]:
        if not self.available():
            return {"status": Status.FAIL.value, "error": str(self._err)}
        k = k or self.cfg.top_k
        # Implement LRAG query and normalize into LNSP format
        return {"status": Status.OK.value, "mode": RetrievalMode.HYBRID_LRAG.value,
                "q": q, "k": k, "results": []}
