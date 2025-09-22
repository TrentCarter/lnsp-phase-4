"""Integration adapters for third-party components."""

from . import lightrag
from .lightrag_adapter import Triple, ingest_triples

__all__ = [
    "lightrag",
    "Triple",
    "ingest_triples",
]
