"""High-level LightRAG adapter utilities for pipeline integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


def _lane_window_default() -> int:
    raw = os.getenv("LIGHTRAG_LANE_WINDOW")
    try:
        return int(raw) if raw is not None else 4
    except ValueError:
        return 4


@dataclass
class Triple:
    """Normalized LightRAG triple ready for Neo4j insertion."""

    src_cpe_id: str
    dst_cpe_id: str
    type: str = "related_to"
    confidence: float = 0.8
    properties: Dict[str, Any] = field(default_factory=dict)

    def with_defaults(self, lane_index: int, source: str = "lightrag") -> "Triple":
        updated = dict(self.properties)
        updated.setdefault("source", source)
        updated.setdefault("lane_index", lane_index)
        updated.setdefault("text", f"{self.src_cpe_id} {self.type} {self.dst_cpe_id}")
        return Triple(
            src_cpe_id=self.src_cpe_id,
            dst_cpe_id=self.dst_cpe_id,
            type=self.type,
            confidence=self.confidence,
            properties=updated,
        )


TripleWriter = Callable[[Triple], None]


def ingest_triples(
    triples: Iterable[Triple],
    lane_index: int,
    *,
    lane_window: Optional[int] = None,
    writer: Optional[TripleWriter] = None,
) -> int:
    """Filter and optionally write LightRAG triples.

    Returns the number of triples within the lane window that were passed to the writer
    (or simply counted when writer is None).
    """

    window = lane_window if lane_window is not None else _lane_window_default()
    count = 0

    for triple in triples:
        enriched = triple.with_defaults(lane_index=lane_index)
        neighbor_lane = enriched.properties.get("lane_index", lane_index)
        if abs(int(neighbor_lane) - lane_index) > window:
            continue
        if writer is not None:
            writer(enriched)
        count += 1

    return count

