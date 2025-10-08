from __future__ import annotations
from typing import List, Optional
import numpy as np
import os
import threading
import time

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
    # Accept both env names; docs refer to LNSP_EMBED_MODEL_DIR
    local = os.getenv("LNSP_EMBEDDER_PATH") or os.getenv("LNSP_EMBED_MODEL_DIR")
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
        # Async controls
        # Defaults: safest synchronous behavior (block until ready, no placeholders)
        self._async = os.getenv("LNSP_EMBEDDER_ASYNC", "0") == "1"  # default: synchronous init
        self._nonblocking_encode = os.getenv("LNSP_EMBEDDER_NONBLOCKING", "0") == "1"  # default: block encode if somehow not ready
        self._fallback_ok = os.getenv("LNSP_EMBEDDER_FALLBACK", "0") == "1"  # default: do NOT allow stub embeddings
        try:
            self._wait_s = float(os.getenv("LNSP_EMBEDDER_WAIT_S", "0"))  # optional small wait before fallback
        except Exception:
            self._wait_s = 0.0
        self._ready_file = os.getenv("LNSP_EMBEDDER_READY_FILE")
        # Readiness primitives
        self._ready_event = threading.Event()
        self._load_error: Optional[Exception] = None

        if SentenceTransformer is None:
            # HF stack not available; will fallback in encode()
            print("[EmbeddingBackend] sentence-transformers not available; using stub embeddings until installed")
            return

        if self._async:
            # Spawn background loader to avoid blocking demos/CLI
            print("[EmbeddingBackend] Loading embedder asynchronously...")
            t = threading.Thread(target=self._load_model_safe, name="EmbedderLoader", daemon=True)
            t.start()
        else:
            # Synchronous load (may block seconds on first run)
            self._load_model_safe()

    def _load_model_safe(self):
        try:
            # Use the offline-aware loader instead of direct instantiation
            # Accept both env names; docs refer to LNSP_EMBED_MODEL_DIR
            local_path = os.getenv("LNSP_EMBEDDER_PATH") or os.getenv("LNSP_EMBED_MODEL_DIR")
            if local_path and os.path.isdir(local_path):
                self.model = SentenceTransformer(local_path, device=self.device)
            elif os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1":
                raise RuntimeError(
                    "Embedder is offline but LNSP_EMBEDDER_PATH not set. "
                    "Place the model at ./models/gtr-t5-base and export LNSP_EMBEDDER_PATH=./models/gtr-t5-base"
                )
            else:
                # Online path (allowed only if your environment permits)
                self.model = SentenceTransformer(self.model_name, device=self.device)
            self._ready_event.set()
            if self._ready_file:
                try:
                    with open(self._ready_file, "w") as f:
                        f.write(f"READY {time.time()}\n")
                except Exception:
                    pass
            print("[EmbeddingBackend] Embedder ready.")
        except Exception as exc:  # pragma: no cover
            self._load_error = exc
            # Keep model None; encode() will fallback or raise based on policy
            print(f"[EmbeddingBackend] Embedder load failed; will fallback if allowed: {exc}")

    def is_ready(self) -> bool:
        return self._ready_event.is_set() and (self.model is not None)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # If ready, use the real model
        if self.model is not None:
            emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
            return emb

        # Not ready: optionally wait a short time, then fallback or raise
        if self._wait_s > 0 and not self._ready_event.is_set():
            self._ready_event.wait(timeout=self._wait_s)
            if self.model is not None:
                emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(emb, dtype=np.float32)

        if not self._nonblocking_encode and not self.is_ready():
            # Blocking policy requested but still not ready
            raise RuntimeError("EmbeddingBackend not ready and non-blocking encode disabled. Set LNSP_EMBEDDER_NONBLOCKING=1 or wait.")

        if not self._fallback_ok:
            raise RuntimeError("EmbeddingBackend not ready and fallback disabled. Set LNSP_EMBEDDER_FALLBACK=1 or wait for readiness.")

        # Fallback: deterministic pseudo-embeddings (unit-test-safe, demo-safe)
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((len(texts), 768)).astype(np.float32)
        from .utils.norms import l2_normalize
        return l2_normalize(arr)
