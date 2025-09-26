from __future__ import annotations
from typing import List, Optional
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # graceful degradation if HF stack not installed yet
    SentenceTransformer = None  # type: ignore


def load_embedder():
    """
    Enforces local-only model loading when offline.
    Honor env:
      - LNSP_EMBEDDER_PATH: local dir of the model (preferred)
      - SENTENCE_TRANSFORMERS_HOME / HF_HOME: cache roots
      - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: '1' to forbid downloads
    """
    local = os.getenv("LNSP_EMBEDDER_PATH")
    if local and os.path.isdir(local):
        return SentenceTransformer(local)
    # If offline is requested, do not attempt network
    if os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
        raise RuntimeError(
            "Embedder is offline but LNSP_EMBEDDER_PATH not set. "
            "Place the model at ./models/gtr-t5-base and export LNSP_EMBEDDER_PATH=./models/gtr-t5-base"
        )
    # Online path (allowed only if your environment permits)
    return SentenceTransformer("sentence-transformers/gtr-t5-base")


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
                # Use the offline-aware loader instead of direct instantiation
                local_path = os.getenv("LNSP_EMBEDDER_PATH")
                if local_path and os.path.isdir(local_path):
                    self.model = SentenceTransformer(local_path, device=device)
                elif os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
                    raise RuntimeError(
                        "Embedder is offline but LNSP_EMBEDDER_PATH not set. "
                        "Place the model at ./models/gtr-t5-base and export LNSP_EMBEDDER_PATH=./models/gtr-t5-base"
                    )
                else:
                    # Online path (allowed only if your environment permits)
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
