from __future__ import annotations
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # graceful degradation if HF stack not installed yet
    SentenceTransformer = None  # type: ignore


class EmbeddingBackend:
    """Thin wrapper around a sentence-transformers model (e.g., GTR-T5, STELLA).
    Produces 768D float32 vectors.
    """

    def __init__(self, model_name: str = "sentence-transformers/gtr-t5-base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name, device=device)
            except Exception as exc:  # pragma: no cover
                print(f"[EmbeddingBackend] Falling back to stub embeddings: {exc}")
                self.model = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            # Fallback: deterministic pseudo-embeddings (for unit tests prior to deps)
            rng = np.random.default_rng(42)
            arr = rng.standard_normal((len(texts), 768)).astype(np.float32)
            from .utils.norms import l2_normalize
            return l2_normalize(arr)
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        return emb
