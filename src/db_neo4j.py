import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")


def get_driver():
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
    MATCH (s:Concept {cpe_id:$src}), (d:Concept {cpe_id:$dst})
    MERGE (s)-[r:REL {type:$rel_type}]->(d)
    RETURN type(r)
    """
    return session.run(q, src=src_id, dst=dst_id, rel_type=rel_type).single().value()


class Neo4jDB:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.driver = None
        if self.enabled:
            try:
                self.driver = get_driver()
                if self.driver:
                    print("[Neo4jDB] Connected to Neo4j")
                else:
                    print("[Neo4jDB] Driver unavailable — using stub mode")
                    self.enabled = False
            except Exception as exc:
                print(f"[Neo4jDB] Connection error: {exc}")
                self.enabled = False
        else:
            print("[Neo4jDB] Running in stub mode")

    def insert_concept(self, cpe_record: dict) -> bool:
        if not self.enabled or not self.driver:
            print(f"[Neo4jDB STUB] Concept {cpe_record['cpe_id']}")
            return True
        try:
            with self.driver.session() as session:
                upsert_concept(session, cpe_record)
            return True
        except Exception as exc:
            print(f"[Neo4jDB] Error inserting concept {cpe_record['cpe_id']}: {exc}")
            return False

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            print("[Neo4jDB] Connection closed")
