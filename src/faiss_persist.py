"""Helpers to persist IVF Faiss indexes to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Faiss is required for persistence utilities") from exc


def save_ivf(index: "faiss.Index", path: str) -> str:
    """Write a Faiss index to disk and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(target))
    return str(target)


def load_ivf(path: str) -> "faiss.Index":
    """Load a Faiss index from disk."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Faiss index not found at {path}")
    return faiss.read_index(str(target))

