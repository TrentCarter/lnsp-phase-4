from __future__ import annotations
from typing import Dict, Any
import os
from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def upsert_concept(session, core: Dict[str, Any]):
    query = (
        """
        MERGE (c:Concept {cpe_id: $cpe_id})
        SET c.text = $concept_text,
            c.tmdBits = $tmd_bits,
            c.tmdLane = $tmd_lane,
            c.laneIndex = $lane_index,
            c.domainCode = $domain_code,
            c.taskCode = $task_code,
            c.modifierCode = $modifier_code,
            c.echoScore = $echo_score,
            c.validationStatus = $validation_status
        RETURN c.cpe_id as id
        """
    )
    return session.run(query, **core).single().value()


def upsert_relation(session, src_id: str, dst_id: str, rel_type: str, confidence: float = 0.6, properties: Dict[str, Any] | None = None):
    query = (
        """
        MATCH (src:Concept {cpe_id:$src}), (dst:Concept {cpe_id:$dst})
        MERGE (src)-[r:REL {type:$rel_type}]->(dst)
        SET r.confidence = $confidence,
            r.properties = $properties
        RETURN type(r)
        """
    )
    return session.run(query, src=src_id, dst=dst_id, rel_type=rel_type, confidence=confidence, properties=properties or {}).single().value()
