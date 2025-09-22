"""Configuration helpers for LightRAG adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional
import os


def _env_to_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_to_float(raw: Optional[str], default: float) -> float:
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _env_to_int(raw: Optional[str], default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


@dataclass
class LightRAGConfig:
    """Runtime configuration shared by LightRAG adapters."""

    graph_enabled: bool = False
    query_enabled: bool = False
    relation_confidence_floor: float = 0.5
    add_self_relation: bool = False
    relation_source_tag: str = "lightrag"
    query_top_k: int = 10
    query_metric: str = "cosine"
    query_weight: float = 0.6
    allow_fallback: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LightRAGConfig":
        """Build configuration from environment variables."""

        return cls(
            graph_enabled=_env_to_bool(os.getenv("LIGHTRAG_GRAPH"), False),
            query_enabled=_env_to_bool(os.getenv("LIGHTRAG_QUERY"), False),
            relation_confidence_floor=_env_to_float(
                os.getenv("LIGHTRAG_REL_CONF_FLOOR"), 0.5
            ),
            add_self_relation=_env_to_bool(os.getenv("LIGHTRAG_REL_SELF"), False),
            relation_source_tag=os.getenv("LIGHTRAG_REL_SOURCE", "lightrag"),
            query_top_k=_env_to_int(os.getenv("LIGHTRAG_QUERY_TOPK"), 10),
            query_metric=os.getenv("LIGHTRAG_QUERY_METRIC", "cosine"),
            query_weight=_env_to_float(os.getenv("LIGHTRAG_QUERY_WEIGHT"), 0.6),
            allow_fallback=_env_to_bool(os.getenv("LIGHTRAG_ALLOW_FALLBACK"), True),
        )

    def copy_with(self, **overrides: Any) -> "LightRAGConfig":
        """Return a copy of the config with the provided overrides applied."""

        return replace(self, **overrides)

    @property
    def any_enabled(self) -> bool:
        """Return True when at least one LightRAG adapter is active."""

        return self.graph_enabled or self.query_enabled

    def relation_metadata(self) -> Dict[str, Any]:
        """Metadata attached to LightRAG-derived relations."""

        return {
            "source": self.relation_source_tag,
            "confidence_floor": self.relation_confidence_floor,
        }
