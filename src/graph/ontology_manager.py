"""
Ontology Manager: Dynamic insertion and management of ontology concepts.

Implements dual-write pattern (PostgreSQL + Neo4j) with closure table maintenance.
Supports dynamic insertion, RL-based refinement, and efficient ancestor/descendant queries.

Author: LNSP Phase 4 Team
Date: 2025-10-01
"""

import uuid
from typing import List, Optional, Dict, Any, Tuple
import psycopg2
from neo4j import Session as Neo4jSession
import numpy as np


class OntologyManager:
    """
    Manages dynamic ontology graph with dual-write to Postgres + Neo4j.
    
    Features:
    - Insert concepts with position control (parent/child specification)
    - Incremental closure table updates (no full rebuild)
    - Multi-parent support (DAG structure)
    - Relation type tracking (is_a, part_of, etc.)
    - RL-friendly: confidence scores, source tracking
    """
    
    def __init__(self, pg_conn: psycopg2.extensions.connection, neo4j_session: Neo4jSession):
        self.pg = pg_conn
        self.neo4j = neo4j_session
    
    def insert_concept_between(
        self,
        concept_text: str,
        concept_vec: np.ndarray,
        parent_id: uuid.UUID,
        child_id: Optional[uuid.UUID] = None,
        relation_type: str = "is_a",
        confidence: float = 1.0,
        source: str = "manual"
    ) -> uuid.UUID:
        """
        Insert new concept between parent and optional child.
        
        Example:
            Before: A → B → C
            After:  A → E → B → C (where E is the new concept)
        
        Args:
            concept_text: Human-readable concept name
            concept_vec: 768D embedding vector (GTR-T5)
            parent_id: UUID of parent concept
            child_id: Optional UUID of child concept (if inserting in middle of chain)
            relation_type: Type of relationship ('is_a', 'part_of', etc.)
            confidence: Confidence score [0.0, 1.0]
            source: Data source ('SWO', 'GO', 'ConceptNet', 'manual', 'RL')
        
        Returns:
            UUID of newly created concept
        
        Raises:
            ValueError: If parent doesn't exist or would create cycle
        """
        with self.pg:  # Transaction context
            with self.pg.cursor() as cur:
                # 1. Validate parent exists
                cur.execute("SELECT 1 FROM cpe_entry WHERE cpe_id = %s", (str(parent_id),))
                if not cur.fetchone():
                    raise ValueError(f"Parent concept {parent_id} does not exist")
                
                # 2. If child specified, validate it exists and is descendant of parent
                if child_id:
                    cur.execute("SELECT 1 FROM cpe_entry WHERE cpe_id = %s", (str(child_id),))
                    if not cur.fetchone():
                        raise ValueError(f"Child concept {child_id} does not exist")
                    
                    # Check child is descendant of parent (prevents cycles)
                    cur.execute("""
                        SELECT 1 FROM ontology_closure
                        WHERE ancestor_id = %s AND descendant_id = %s
                    """, (str(parent_id), str(child_id)))
                    if not cur.fetchone():
                        raise ValueError(f"Child {child_id} is not a descendant of parent {parent_id}")
                
                # 3. Create CPE entry in Postgres
                concept_id = uuid.uuid4()
                cur.execute("""
                    INSERT INTO cpe_entry (
                        cpe_id, mission_text, source_chunk, concept_text,
                        probe_question, expected_answer,
                        domain_code, task_code, modifier_code,
                        content_type, dataset_source, chunk_position
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, 0, 0, 0, 'ontology', %s, '{}'::jsonb
                    )
                """, (
                    str(concept_id),
                    f"Ontology concept: {concept_text}",
                    f"Dynamic insertion from {source}",
                    concept_text,
                    f"What is {concept_text}?",
                    f"{concept_text} is a concept in the ontology",
                    source
                ))
                
                # 4. Insert vector (concept_vec only, no TMD for ontology concepts)
                fused_vec = np.concatenate([concept_vec, np.zeros(16)])  # Pad to 784D
                cur.execute("""
                    INSERT INTO cpe_vectors (cpe_id, concept_vec, tmd_vec, fused_vec)
                    VALUES (%s, %s, %s, %s)
                """, (
                    str(concept_id),
                    concept_vec.tolist(),
                    [0.0] * 16,
                    fused_vec.tolist()
                ))
                
                # 5. Add edge: parent → new_concept
                self._add_edge(cur, parent_id, concept_id, relation_type, confidence, source)
                
                # 6. If child specified, add edge: new_concept → child and remove parent → child
                if child_id:
                    self._add_edge(cur, concept_id, child_id, relation_type, confidence, source)
                    self._remove_edge(cur, parent_id, child_id, relation_type)
                
                # 7. Update closure table (incremental)
                self._update_closure_incremental(cur, concept_id, parent_id, child_id, relation_type)
        
        # 8. Sync to Neo4j (outside Postgres transaction)
        self._sync_to_neo4j(concept_id, concept_text, parent_id, child_id, relation_type, confidence)
        
        return concept_id
    
    def _add_edge(
        self,
        cur,
        parent_id: uuid.UUID,
        child_id: uuid.UUID,
        relation_type: str,
        confidence: float,
        source: str
    ):
        """Add edge to ontology_edge table."""
        cur.execute("""
            INSERT INTO ontology_edge (parent_id, child_id, relation_type, confidence, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (parent_id, child_id, relation_type) DO UPDATE
            SET confidence = EXCLUDED.confidence, updated_at = NOW()
        """, (str(parent_id), str(child_id), relation_type, confidence, source))
    
    def _remove_edge(self, cur, parent_id: uuid.UUID, child_id: uuid.UUID, relation_type: str):
        """Remove edge from ontology_edge table."""
        cur.execute("""
            DELETE FROM ontology_edge
            WHERE parent_id = %s AND child_id = %s AND relation_type = %s
        """, (str(parent_id), str(child_id), relation_type))
    
    def _update_closure_incremental(
        self,
        cur,
        new_id: uuid.UUID,
        parent_id: uuid.UUID,
        child_id: Optional[uuid.UUID],
        relation_type: str
    ):
        """
        Update closure table incrementally (no full rebuild).
        
        New paths created:
        1. Self-loop: new_concept → new_concept (distance 0)
        2. ancestors(parent) → new_concept
        3. new_concept → descendants(child) [if child exists]
        4. ancestors(parent) → descendants(child) via new_concept [if child exists]
        """
        # 1. Add self-loop
        cur.execute("""
            INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
            VALUES (%s, %s, 0, ARRAY[]::TEXT[])
        """, (str(new_id), str(new_id)))
        
        # 2. Add: ancestors(parent) → new_concept
        cur.execute("""
            INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
            SELECT ancestor_id, %s, path_length + 1, relation_chain || %s
            FROM ontology_closure
            WHERE descendant_id = %s
        """, (str(new_id), relation_type, str(parent_id)))
        
        # 3. Add: new_concept → descendants(child) [if child exists]
        if child_id:
            cur.execute("""
                INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
                SELECT %s, descendant_id, path_length + 1, %s || relation_chain
                FROM ontology_closure
                WHERE ancestor_id = %s
            """, (str(new_id), relation_type, str(child_id)))
            
            # 4. Add: ancestors(parent) → descendants(child) via new_concept
            cur.execute("""
                INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
                SELECT a.ancestor_id, d.descendant_id,
                       a.path_length + 1 + d.path_length,
                       a.relation_chain || %s || d.relation_chain
                FROM ontology_closure a, ontology_closure d
                WHERE a.descendant_id = %s AND d.ancestor_id = %s
                ON CONFLICT (ancestor_id, descendant_id) DO UPDATE
                SET path_length = LEAST(ontology_closure.path_length, EXCLUDED.path_length),
                    relation_chain = CASE
                        WHEN LENGTH(ARRAY_TO_STRING(EXCLUDED.relation_chain, '')) < 
                             LENGTH(ARRAY_TO_STRING(ontology_closure.relation_chain, ''))
                        THEN EXCLUDED.relation_chain
                        ELSE ontology_closure.relation_chain
                    END
            """, (relation_type, str(new_id), str(new_id)))
    
    def _sync_to_neo4j(
        self,
        concept_id: uuid.UUID,
        concept_text: str,
        parent_id: uuid.UUID,
        child_id: Optional[uuid.UUID],
        relation_type: str,
        confidence: float
    ):
        """Sync concept and edges to Neo4j graph."""
        # Create concept node
        self.neo4j.run("""
            MERGE (c:Concept {concept_id: $concept_id})
            SET c.concept_text = $concept_text, c.updated_at = datetime()
        """, concept_id=str(concept_id), concept_text=concept_text)
        
        # Create edge: parent → new_concept
        rel_type = relation_type.upper().replace('_', '_')
        self.neo4j.run(f"""
            MATCH (parent:Concept {{concept_id: $parent_id}})
            MATCH (child:Concept {{concept_id: $child_id}})
            MERGE (parent)-[r:{rel_type}]->(child)
            SET r.confidence = $confidence, r.updated_at = datetime()
        """, parent_id=str(parent_id), child_id=str(concept_id), confidence=confidence)
        
        # If child specified, create edge: new_concept → child and remove parent → child
        if child_id:
            self.neo4j.run(f"""
                MATCH (new:Concept {{concept_id: $new_id}})
                MATCH (child:Concept {{concept_id: $child_id}})
                MERGE (new)-[r:{rel_type}]->(child)
                SET r.confidence = $confidence, r.updated_at = datetime()
            """, new_id=str(concept_id), child_id=str(child_id), confidence=confidence)
            
            self.neo4j.run(f"""
                MATCH (parent:Concept {{concept_id: $parent_id}})-[r:{rel_type}]->(child:Concept {{concept_id: $child_id}})
                DELETE r
            """, parent_id=str(parent_id), child_id=str(child_id))
    
    def get_ancestors(self, concept_id: uuid.UUID, max_distance: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all ancestors of a concept.
        
        Args:
            concept_id: UUID of concept
            max_distance: Maximum path length (None = unlimited)
        
        Returns:
            List of dicts with keys: ancestor_id, distance, relation_path
        """
        with self.pg.cursor() as cur:
            cur.execute("""
                SELECT ancestor_id, distance, relation_path
                FROM get_ancestors(%s, %s)
            """, (str(concept_id), max_distance))
            
            return [
                {
                    'ancestor_id': uuid.UUID(row[0]),
                    'distance': row[1],
                    'relation_path': row[2]
                }
                for row in cur.fetchall()
            ]
    
    def get_descendants(self, concept_id: uuid.UUID, max_distance: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all descendants of a concept.
        
        Args:
            concept_id: UUID of concept
            max_distance: Maximum path length (None = unlimited)
        
        Returns:
            List of dicts with keys: descendant_id, distance, relation_path
        """
        with self.pg.cursor() as cur:
            cur.execute("""
                SELECT descendant_id, distance, relation_path
                FROM get_descendants(%s, %s)
            """, (str(concept_id), max_distance))
            
            return [
                {
                    'descendant_id': uuid.UUID(row[0]),
                    'distance': row[1],
                    'relation_path': row[2]
                }
                for row in cur.fetchall()
            ]
    
    def is_ancestor(self, ancestor_id: uuid.UUID, descendant_id: uuid.UUID) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id."""
        with self.pg.cursor() as cur:
            cur.execute("""
                SELECT is_ancestor(%s, %s)
            """, (str(ancestor_id), str(descendant_id)))
            return cur.fetchone()[0]
    
    def rebuild_closure_table(self):
        """
        Rebuild entire closure table from scratch.
        
        Use this when:
        - Initial ontology load
        - After bulk edge updates
        - Closure table corruption detected
        
        Warning: This is expensive (O(n³)). Use incremental updates when possible.
        """
        with self.pg:
            with self.pg.cursor() as cur:
                # Clear closure table
                cur.execute("DELETE FROM ontology_closure")
                
                # 1. Add self-loops for all concepts
                cur.execute("""
                    INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
                    SELECT cpe_id, cpe_id, 0, ARRAY[]::TEXT[]
                    FROM cpe_entry
                """)
                
                # 2. Add direct edges (path_length = 1)
                cur.execute("""
                    INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
                    SELECT parent_id, child_id, 1, ARRAY[relation_type]
                    FROM ontology_edge
                """)
                
                # 3. Iteratively add transitive paths
                # Repeat until no new paths found (fixed-point)
                while True:
                    cur.execute("""
                        INSERT INTO ontology_closure (ancestor_id, descendant_id, path_length, relation_chain)
                        SELECT DISTINCT a.ancestor_id, b.descendant_id,
                               a.path_length + b.path_length,
                               a.relation_chain || b.relation_chain
                        FROM ontology_closure a
                        JOIN ontology_closure b ON a.descendant_id = b.ancestor_id
                        WHERE NOT EXISTS (
                            SELECT 1 FROM ontology_closure c
                            WHERE c.ancestor_id = a.ancestor_id
                              AND c.descendant_id = b.descendant_id
                        )
                        ON CONFLICT (ancestor_id, descendant_id) DO UPDATE
                        SET path_length = LEAST(ontology_closure.path_length, EXCLUDED.path_length),
                            relation_chain = CASE
                                WHEN LENGTH(ARRAY_TO_STRING(EXCLUDED.relation_chain, '')) < 
                                     LENGTH(ARRAY_TO_STRING(ontology_closure.relation_chain, ''))
                                THEN EXCLUDED.relation_chain
                                ELSE ontology_closure.relation_chain
                            END
                    """)
                    
                    if cur.rowcount == 0:
                        break  # No new paths found


if __name__ == "__main__":
    # Example usage
    import psycopg2
    from neo4j import GraphDatabase
    
    pg_conn = psycopg2.connect("dbname=lnsp")
    neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    neo4j_session = neo4j_driver.session()
    
    manager = OntologyManager(pg_conn, neo4j_session)
    
    # Insert concept E between A and B
    # Before: A → B → C
    # After:  A → E → B → C
    parent_uuid = uuid.UUID("...")  # Replace with real parent UUID
    child_uuid = uuid.UUID("...")   # Replace with real child UUID
    concept_vec = np.random.randn(768).astype(np.float32)
    
    new_id = manager.insert_concept_between(
        concept_text="New Intermediate Concept",
        concept_vec=concept_vec,
        parent_id=parent_uuid,
        child_id=child_uuid,
        relation_type="is_a",
        confidence=0.95,
        source="RL"
    )
    
    print(f"Inserted concept: {new_id}")
    
    # Query ancestors
    ancestors = manager.get_ancestors(new_id, max_distance=5)
    print(f"Ancestors: {ancestors}")
    
    pg_conn.close()
    neo4j_session.close()
    neo4j_driver.close()
