import os
from typing import List, Optional, Dict, Any

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional dependency
    GraphDatabase = None

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")


def get_driver():
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not available")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def upsert_concept(session, core: dict):
    q = """
    MERGE (c:Concept {cpe_id:$cpe_id})
    SET c.text=$concept_text, c.tmdBits=$tmd_bits,
        c.tmdLane=$tmd_lane, c.laneIndex=$lane_index
    RETURN c.cpe_id as id
    """
    return session.run(q, **core).single().value()


def upsert_relation(session, src_id: str, dst_id: str, rel_type: str):
    q = """
    MATCH (s:Concept {cpe_id:$src})
    MERGE (d:Entity {name:$dst})
    MERGE (s)-[r:RELATES_TO {type:$rel_type}]->(d)
    RETURN type(r)
    """
    result = session.run(q, src=src_id, dst=dst_id, rel_type=rel_type).single()
    return result.value() if result else None


class Neo4jDB:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.driver = None
        if self.enabled and GraphDatabase is not None:
            try:
                self.driver = get_driver()
                if self.driver:
                    print("[Neo4jDB] ✓ Connected to Neo4j (REAL writes enabled)")
                else:
                    print("[Neo4jDB] ✗ Driver unavailable — using stub mode")
                    self.enabled = False
            except Exception as exc:
                print(f"[Neo4jDB] ✗ Connection error: {exc} — using stub mode")
                self.enabled = False
        elif self.enabled:
            print("[Neo4jDB] ✗ neo4j driver missing — using stub mode")
            self.enabled = False
        else:
            print("[Neo4jDB] STUB MODE (writes disabled)")

    def insert_concept(self, cpe_record: dict) -> bool:
        if not self.enabled or not self.driver:
            # Stub mode - no actual writes
            return True
        try:
            with self.driver.session() as session:
                upsert_concept(session, cpe_record)
            return True
        except Exception as exc:
            print(f"[Neo4jDB] Error inserting concept {cpe_record['cpe_id']}: {exc}")
            return False

    def insert_relation_triple(self, src_cpe_id: str, dst_cpe_id: str, rel_type: str) -> bool:
        if not self.enabled or not self.driver:
            # Stub mode - no actual writes
            return True
        try:
            with self.driver.session() as session:
                upsert_relation(session, src_cpe_id, dst_cpe_id, rel_type)
            return True
        except Exception as exc:
            print(f"[Neo4jDB] Error inserting relation {src_cpe_id} -> {dst_cpe_id}: {exc}")
            return False

    def search_fulltext(self, q: str, lane: Optional[int], top_k: int) -> List[Dict[str,Any]]:
        """Search concepts using fulltext index."""
        if not self.enabled or not self.driver:
            print(f"[Neo4jDB STUB] Fulltext search: {q}")
            return []
        try:
            cy = """
            CALL db.index.fulltext.queryNodes('concept_text_fts', $q) YIELD node, score
            WHERE $lane IS NULL OR node.laneIndex = $lane
            RETURN node.cpe_id AS cpe_id, node.text AS concept_text, node.laneIndex AS lane_index,
                   coalesce(node.tmdBits,0) AS tmd_bits, score
            ORDER BY score DESC LIMIT $top_k
            """
            with self.driver.session() as s:
                res = s.run(cy, q=q, lane=lane, top_k=top_k)
                return [r.data() for r in res]
        except Exception as exc:
            print(f"[Neo4jDB] Error in fulltext search: {exc}")
            return []

    def search_by_seed_ids(self, seed_ids: List[str], lane: Optional[int], top_k: int) -> List[Dict[str,Any]]:
        """Search concepts by specific CPE IDs."""
        if not self.enabled or not self.driver:
            print(f"[Neo4jDB STUB] Seed search: {seed_ids}")
            return []
        try:
            cy = """
            MATCH (c:Concept) WHERE c.cpe_id IN $ids
            WITH c, 1.0 AS score
            WHERE $lane IS NULL OR c.laneIndex = $lane
            RETURN c.cpe_id AS cpe_id, c.text AS concept_text, c.laneIndex AS lane_index,
                   coalesce(c.tmdBits,0) AS tmd_bits, score
            ORDER BY score DESC LIMIT $top_k
            """
            with self.driver.session() as s:
                res = s.run(cy, ids=seed_ids, lane=lane, top_k=top_k)
                return [r.data() for r in res]
        except Exception as exc:
            print(f"[Neo4jDB] Error in seed search: {exc}")
            return []

    def expand_hops(self, seed: str, max_hops: int, top_k: int, lane: Optional[int]) -> List[Dict[str,Any]]:
        """Expand from a seed concept via graph relationships."""
        if not self.enabled or not self.driver:
            print(f"[Neo4jDB STUB] Hop expansion: {seed}")
            return []
        try:
            # Support both Concept-to-Concept and Concept-to-Entity relationships
            cy = f"""
            MATCH (s:Concept {{cpe_id:$seed}})-[r:RELATES_TO*1..{max_hops}]->(n)
            WHERE (n:Concept AND ($lane IS NULL OR n.laneIndex = $lane)) OR n:Entity
            WITH n, length(r) AS path_length,
                 1.0 / length(r) AS score,
                 [rel IN r | {{pred: coalesce(rel.type, 'relates_to'), weight: coalesce(rel.weight, 1.0)}}] AS path_meta
            RETURN
                CASE WHEN n:Concept THEN n.cpe_id ELSE n.name END AS cpe_id,
                CASE WHEN n:Concept THEN n.text ELSE n.name END AS concept_text,
                CASE WHEN n:Concept THEN n.laneIndex ELSE null END AS lane_index,
                CASE WHEN n:Concept THEN coalesce(n.tmdBits,0) ELSE 0 END AS tmd_bits,
                score, path_meta
            ORDER BY score DESC LIMIT $top_k
            """
            with self.driver.session() as s:
                res = s.run(cy, seed=seed, lane=lane, top_k=top_k)
                return [r.data() for r in res]
        except Exception as exc:
            print(f"[Neo4jDB] Error in hop expansion: {exc}")
            return []

    def graph_health(self) -> Dict[str,Any]:
        """Get graph health statistics."""
        if not self.enabled or not self.driver:
            return {"concepts": 0, "entities": 0, "edges": 0, "status": "disabled"}
        try:
            cy = """
            CALL {
              MATCH (c:Concept) RETURN count(c) AS cnt_concepts
            }
            CALL {
              MATCH (e:Entity) RETURN count(e) AS cnt_entities
            }
            CALL {
              MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt_edges
            }
            RETURN cnt_concepts, cnt_entities, cnt_edges
            """
            with self.driver.session() as s:
                r = s.run(cy).single()
                return {
                    "concepts": r["cnt_concepts"],
                    "entities": r["cnt_entities"],
                    "edges": r["cnt_edges"],
                    "status": "healthy"
                }
        except Exception as exc:
            print(f"[Neo4jDB] Error getting health: {exc}")
            return {"concepts": 0, "entities": 0, "edges": 0, "status": f"error: {exc}"}

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            print("[Neo4jDB] Connection closed")
