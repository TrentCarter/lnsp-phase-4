# SPDX-License-Identifier: Apache-2.0
# Adapter for LightRAG: GTR-T5 768D, offline-only
from __future__ import annotations
import os
from typing import Iterable, List, Sequence
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(f"sentence-transformers not available: {e}")

DEFAULT_LOCAL_DIR = os.getenv("LNSP_EMBED_MODEL_DIR", "data/teacher_models/gtr-t5-base")

class GTRT5Embedder:
    """
    Minimal interface LightRAG expects:
      - dim: int
      - embed_batch(List[str]) -> np.ndarray (N,768) float32, L2-normalized
      - embed(str) -> np.ndarray (768,)  [optional but handy]
    """

    def __init__(self,
                 model_dir: str | None = None,
                 device: str | None = None,
                 batch_size: int = 64,
                 normalize: bool = True):
        # Hard-force offline
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_dir = model_dir or DEFAULT_LOCAL_DIR
        if not (os.path.isdir(self.model_dir)):
            raise RuntimeError(f"GTR model path not found: {self.model_dir}")

        self.model = SentenceTransformer(self.model_dir, device=device or "cpu")
        self.batch_size = int(batch_size)
        self.normalize = bool(normalize)
        # Smoke one encode to fail fast if weights are broken
        _ = self.model.encode(["ok"], normalize_embeddings=True)

        self._dim = 768

    @property
    def dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        # SentenceTransformer returns float32 if convert_to_numpy=True
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32, copy=False)

        # Fail-fast: no zero vectors, correct shape
        if vecs.ndim != 2 or vecs.shape[1] != self._dim:
            raise ValueError(f"Bad embedding shape {vecs.shape}; expected (N,{self._dim})")
        if not np.isfinite(vecs).all():
            raise ValueError("NaN/Inf detected in embeddings")
        norms = np.linalg.norm(vecs, axis=1)
        if (norms < 0.99).any():
            # If normalize=False in config, you can relax this; for now enforce contract.
            raise ValueError("Non-normalized vectors detected; set normalize=true.")
        return vecs

    # Add embedding_dim attribute for LightRAG compatibility
    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    # Legacy method for compatibility with existing code
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.embed_batch(list(texts))

    def encode_one(self, text: str) -> np.ndarray:
        return self.embed(text)

    # Optional convenience constructor for config-driven load
    @classmethod
    def from_config(cls, cfg: dict) -> "GTRT5Embedder":
        return cls(
            model_dir=cfg.get("local_dir") or DEFAULT_LOCAL_DIR,
            device=cfg.get("device") or "cpu",
            batch_size=int(cfg.get("batch_size", 64)),
            normalize=bool(cfg.get("normalize", True)),
        )

def get_embedder() -> GTRT5Embedder:
    """Factory used by configs/lightrag.yml to instantiate the embedder."""
    return GTRT5Embedder()

def load_embedder(cfg: dict):
    """LightRAG compatible factory function."""
    return GTRT5Embedder.from_config(cfg.get("embedder", cfg))

__all__ = ["GTRT5Embedder", "get_embedder", "load_embedder"]