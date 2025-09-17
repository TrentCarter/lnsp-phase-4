# app/adapters/gtr_t5_encoder.py
from __future__ import annotations
import numpy as np

class GTRT5Encoder:
    """
    Minimal 768D encoder using sentence-transformers 'gtr-t5-base'.
    Returns float32 numpy arrays shaped [N, 768].
    """

    def __init__(self, model_name: str = "sentence-transformers/gtr-t5-base",
                 device: str = "cpu", batch_size: int = 16):
        # Lazy import so the module doesn't hard-require ST unless used
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        # Sanity: ensure 768D (gtr-t5-base)
        test = self.model.encode(["_"], convert_to_numpy=True, normalize_embeddings=False)
        if test.shape[1] != 768:
            raise RuntimeError(f"Expected 768D embeddings, got {test.shape[1]}D from {model_name}")

    def encode(self, texts: list[str],
               normalize: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False  # we control normalization explicitly
        ).astype(np.float32)
        if normalize:
            vecs = self._l2(vecs)
        return vecs

    def encode_one(self, text: str, normalize: bool = False) -> np.ndarray:
        v = self.encode([text], normalize=normalize)[0]
        return v

    @staticmethod
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        n[n < 1e-8] = 1e-8
        return x / n
