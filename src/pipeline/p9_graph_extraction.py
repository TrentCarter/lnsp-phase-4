"""P9 — Graph extraction stage for the LNSP pipeline."""

from __future__ import annotations

from typing import Dict, Any, List

from ..integrations import Triple, ingest_triples
from ..integrations.lightrag import LightRAGGraphBuilderAdapter


def build_triples(
    cpe_record: Dict[str, Any],
    graph_adapter: LightRAGGraphBuilderAdapter,
) -> List[Triple]:
    """Normalize extracted relations into LightRAG triples."""

    normalized = graph_adapter.enhance_relations(cpe_record)
    triples: List[Triple] = []

    for rel in normalized:
        dst_identifier = rel.get("obj_id") or rel.get("obj") or cpe_record["cpe_id"]
        properties = {
            k: v
            for k, v in rel.items()
            if k not in {"obj_id", "obj", "pred", "confidence"}
        }
        triples.append(
            Triple(
                src_cpe_id=cpe_record["cpe_id"],
                dst_cpe_id=str(dst_identifier),
                type=rel.get("pred", "related_to"),
                confidence=float(rel.get("confidence", 0.8)),
                properties=properties,
            )
        )

    # Keep normalized view for downstream auditing
    cpe_record["relations_text"] = normalized
    return triples


def run_graph_extraction(
    cpe_record: Dict[str, Any],
    graph_adapter: LightRAGGraphBuilderAdapter,
    neo_db,
) -> int:
    """Build triples and write them via Neo4jDB."""

    triples = build_triples(cpe_record, graph_adapter)
    if not triples:
        return 0

    writer_func = getattr(neo_db, "insert_relation_triple", None)
    if writer_func is None:
        return len(triples)

    # Create wrapper to convert Triple object to Neo4jDB method signature
    def triple_writer(triple):
        writer_func(triple.src_cpe_id, triple.dst_cpe_id, triple.type)

    return ingest_triples(triples, lane_index=cpe_record["lane_index"], writer=triple_writer)

