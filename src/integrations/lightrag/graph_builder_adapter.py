"""Graph adapter that vendors LightRAG heuristics into relation assembly."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .config import LightRAGConfig

try:
    from third_party.lightrag.types import Document
except Exception:  # pragma: no cover - fallback when LightRAG snippets unavailable
    Document = None  # type: ignore


Relation = Dict[str, Any]


class LightRAGGraphBuilderAdapter:
    """Normalize and score relations using LightRAG-derived utilities."""

    def __init__(
        self,
        config: LightRAGConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self.available: bool = bool(Document) and config.graph_enabled

        if config.graph_enabled and not self.available:
            self._logger.warning(
                "LightRAG graph integration requested but Document utilities are unavailable;"
                " relations will only receive baseline normalization."
            )

    @classmethod
    def from_config(
        cls, config: LightRAGConfig, logger: Optional[logging.Logger] = None
    ) -> "LightRAGGraphBuilderAdapter":
        return cls(config=config, logger=logger)

    def enhance_relations(
        self, cpe_record: Dict[str, Any], initial_relations: Optional[Iterable[Relation]] = None
    ) -> List[Relation]:
        """Return relations enriched with confidence and provenance metadata."""

        relations = list(initial_relations if initial_relations is not None else cpe_record.get("relations_text", []))

        if not relations and self.config.add_self_relation:
            relations = [
                {
                    "subj": cpe_record.get("concept_text", "unknown"),
                    "pred": "self_related",
                    "obj": cpe_record.get("concept_text", "unknown"),
                }
            ]

        if not relations:
            return []

        seen: Set[Tuple[str, str, str]] = set()
        normalized: List[Relation] = []

        for rel in relations:
            normalized_rel = self._normalize_relation(rel, cpe_record)
            if normalized_rel is None:
                continue

            key = (
                normalized_rel["subj"].lower(),
                normalized_rel["pred"].lower(),
                normalized_rel["obj"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            normalized.append(normalized_rel)

        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_relation(
        self, relation: Relation, cpe_record: Dict[str, Any]
    ) -> Optional[Relation]:
        subj = relation.get("subj") or cpe_record.get("concept_text")
        obj = relation.get("obj")
        if not subj or not obj:
            return None

        pred = relation.get("pred", "related_to")
        base = dict(relation)

        triple_text = f"{subj} {pred} {obj}"
        token_estimate = self._estimate_tokens(triple_text)

        confidence = relation.get("confidence")
        if confidence is None:
            confidence = self._confidence_from_tokens(token_estimate)

        base.update(
            {
                "subj": subj,
                "pred": pred,
                "obj": obj,
                "confidence": float(confidence),
                "source": self.config.relation_source_tag,
                "text": triple_text,
            }
        )

        return base

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0

        if self.available and Document is not None:
            try:
                document = Document(text=text, meta_data={"origin": "lightrag-adapter"})
                return int(document.estimated_num_tokens or len(text.split()))
            except Exception as exc:  # pragma: no cover - defensive guard
                self._logger.debug("LightRAG Document token estimation failed: %s", exc)

        return len(text.split())

    def _confidence_from_tokens(self, token_estimate: int) -> float:
        if token_estimate <= 0:
            return 1.0

        # Lightweight heuristic: shorter relations trend higher confidence
        base = 1.0 / (1.0 + 0.1 * max(token_estimate - 1, 0))
        return max(self.config.relation_confidence_floor, min(1.0, base))

