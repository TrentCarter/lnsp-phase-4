"""LightRAG data types (MIT License)."""

# Derived from LightRAG (c) SylphAI, Inc. under the MIT License.
# Simplified to remove heavy tokenizer dependencies while preserving
# the external surface used by the LNSP pipeline.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import uuid

__all__ = ["Document"]


@dataclass
class Document:
    """Lightweight stand-in for LightRAG's Document dataclass."""

    text: str
    meta_data: Dict[str, Any] = field(default_factory=dict)
    vector: List[float] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order: Optional[int] = None
    score: Optional[float] = None
    parent_doc_id: Optional[Union[str, uuid.UUID]] = None
    estimated_num_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.estimated_num_tokens is None:
            self.estimated_num_tokens = self._estimate_tokens(self.text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return len(text.split())
