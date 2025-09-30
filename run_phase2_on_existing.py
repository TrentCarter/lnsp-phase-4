#!/usr/bin/env python3
"""
Run Phase-2 entity resolution on existing CPE data in PostgreSQL.
This adds cross-document relationships to improve graph quality.
"""

import sys
from pathlib import Path

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.db_postgres import PostgresDB
from src.db_neo4j import Neo4jDB
from src.integrations.lightrag import LightRAGConfig, LightRAGGraphBuilderAdapter
from src.pipeline.p10_entity_resolution import run_two_phase_graph_extraction


def main():
    """Run Phase-2 entity resolution on existing data."""

    # Initialize database connections
    pg_db = PostgresDB(enabled=True)
    neo_db = Neo4jDB(enabled=True)

    # Initialize LightRAG adapter
    lightrag_config = LightRAGConfig.from_env()
    graph_adapter = LightRAGGraphBuilderAdapter.from_config(lightrag_config)

    print("=== Fetching existing CPE records from PostgreSQL ===")

    # Fetch all CPE records that have relations
    query = """
    SELECT cpe_id, concept_text, probe_question, expected_answer, relations_text,
           soft_negatives, hard_negatives, source_chunk
    FROM cpe_entry
    WHERE relations_text IS NOT NULL
    AND jsonb_array_length(relations_text) > 0
    ORDER BY created_at
    """

    if not pg_db.enabled or pg_db.conn is None:
        print("PostgreSQL not connected!")
        return 1

    with pg_db.conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("No CPE records with relations found!")
        return 1

    # Convert to CPE record format expected by Phase-2
    cpe_records = []
    for row in rows:
        cpe_record = {
            "cpe_id": row[0],
            "concept_text": row[1],
            "probe_question": row[2],
            "expected_answer": row[3],
            "relations_text": row[4],  # Already parsed JSON
            "soft_negatives": row[5],
            "hard_negatives": row[6],
            "source_chunk": row[7]
        }
        cpe_records.append(cpe_record)

    print(f"Found {len(cpe_records)} CPE records with relations")

    # Check current cross-document relationship count
    print("\n=== Pre-Phase-2 Neo4j Status ===")
    if neo_db.enabled and neo_db.driver:
        with neo_db.driver.session() as session:

            # Count total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            total_rels = result.single()["total_rels"]

            # Count cross-document relationships (if any exist)
            result = session.run("""
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE a.cpe_id <> b.cpe_id
                RETURN count(r) as cross_doc_rels
            """)
            cross_doc_rels = result.single()["cross_doc_rels"]

            print(f"Total relationships: {total_rels}")
            print(f"Cross-document relationships: {cross_doc_rels}")
    else:
        print("Neo4j not connected!")
        total_rels = cross_doc_rels = 0

    # Run Phase-2 entity resolution
    print("\n=== Running Phase-2 Entity Resolution ===")
    phase2_results = run_two_phase_graph_extraction(
        cpe_records,
        graph_adapter,
        neo_db,
        output_dir="artifacts"
    )

    # Check post-Phase-2 status
    print("\n=== Post-Phase-2 Neo4j Status ===")
    if neo_db.enabled and neo_db.driver:
        with neo_db.driver.session() as session:

            # Count total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            new_total_rels = result.single()["total_rels"]

            # Count cross-document relationships
            result = session.run("""
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE a.cpe_id <> b.cpe_id
                RETURN count(r) as cross_doc_rels
            """)
            new_cross_doc_rels = result.single()["cross_doc_rels"]

            print(f"Total relationships: {new_total_rels} (was {total_rels})")
            print(f"Cross-document relationships: {new_cross_doc_rels} (was {cross_doc_rels})")
            print(f"New relationships added: {new_total_rels - total_rels}")
    else:
        print("Neo4j not connected!")
        new_total_rels = new_cross_doc_rels = 0

    # Display results
    print("\n=== Phase-2 Results Summary ===")
    print(f"Total entities processed: {phase2_results['total_entities']}")
    print(f"Entity clusters formed: {phase2_results['entity_clusters']}")
    print(f"Cross-document relationships created: {phase2_results['cross_doc_relationships']}")
    print(f"Entity matches found: {phase2_results['entity_matches']}")

    if phase2_results['cross_doc_relationships'] > 0:
        print("✅ Phase-2 successfully added cross-document relationships!")
        print("📈 Graph quality improved with better entity linking")
    else:
        print("⚠️  No cross-document relationships were created")
        print("🔍 Consider adjusting similarity thresholds or adding more diverse data")

    return 0


if __name__ == "__main__":
    sys.exit(main())