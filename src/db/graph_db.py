from __future__ import annotations
from typing import Dict, Any, Optional
import os

try:
    from ..loaders.neo4j_writer import get_driver, upsert_concept, upsert_relation
except ImportError:
    def upsert_concept(session, core):
        print(f"[GRAPH_DB STUB] Would upsert concept: {core.get('cpe_id', 'unknown')}")
    def upsert_relation(session, src_id, dst_id, rel_type, confidence=0.6, properties=None):
        print(f"[GRAPH_DB STUB] Would upsert relation: {src_id} -> {dst_id}")
    def get_driver():
        return None


class GraphDB:
    """Graph database writer for concept relationships."""

    def __init__(self):
        self.use_neo4j = os.getenv("USE_NEO4J", "false").lower() == "true"

    def insert_graph_nodes_edges(self, cpe_id: str, extraction: Dict[str, Any]):
        """Insert concept nodes and relationships into graph database."""

        if self.use_neo4j:
            driver = get_driver()
            if driver:
                with driver.session() as session:
                    # Upsert concept node
                    concept_data = {
                        "cpe_id": cpe_id,
                        "concept_text": extraction["concept"],
                        "tmd_bits": extraction["tmd_bits"],
                        "tmd_lane": extraction["tmd_lane"],
                        "lane_index": extraction["lane_index"],
                        "domain_code": extraction["domain_code"],
                        "task_code": extraction["task_code"],
                        "modifier_code": extraction["modifier_code"],
                        "echo_score": extraction["echo_score"],
                        "validation_status": extraction["validation_status"]
                    }
                    upsert_concept(session, concept_data)

                    # Insert relations (placeholder for now)
                    relations = extraction.get("relations", [])
                    for rel in relations:
                        if "subj" in rel and "obj" in rel:
                            upsert_relation(
                                session, cpe_id, rel["obj"], rel.get("pred", "related_to"),
                                confidence=0.8, properties={"source": "extraction"}
                            )

                driver.close()

        # Fallback: print to console
        relations_count = len(extraction.get("relations", []))
        print(f"[GRAPH_DB STUB] CPE {cpe_id}: {relations_count} relations")
