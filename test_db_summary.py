#!/usr/bin/env python3
"""
Database connection status summary for LNSP pipeline.
This script tests PostgreSQL and Neo4j connections and provides a summary.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_postgresql():
    """Test PostgreSQL connection and return status."""
    try:
        import psycopg2
        print("✓ psycopg2-binary installed")

        # Test connection
        host = os.getenv("PG_HOST", "localhost")
        port = os.getenv("PG_PORT", "5432")
        dbname = os.getenv("PG_DBNAME", "lnsp")
        user = os.getenv("PG_USER", "lnsp")
        password = os.getenv("PG_PASSWORD", "lnsp")

        try:
            conn = psycopg2.connect(
                host=host, port=port, dbname=dbname, user=user, password=password
            )

            # Get table counts
            cur = conn.cursor()
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'cpe%';")
            tables = [row[0] for row in cur.fetchall()]

            counts = {}
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                counts[table] = cur.fetchone()[0]

            cur.close()
            conn.close()

            return {
                "status": "connected",
                "connection": f"{user}@{host}:{port}/{dbname}",
                "tables": tables,
                "counts": counts
            }

        except psycopg2.OperationalError as e:
            return {
                "status": "connection_failed",
                "error": str(e),
                "suggestion": "Start PostgreSQL: docker run -d --name postgres-lnsp -e POSTGRES_DB=lnsp -e POSTGRES_USER=lnsp -e POSTGRES_PASSWORD=lnsp -p 5432:5432 postgres:15"
            }

    except ImportError:
        return {
            "status": "client_missing",
            "suggestion": "Install with: pip install psycopg2-binary"
        }

def test_neo4j():
    """Test Neo4j connection and return status."""
    try:
        from neo4j import GraphDatabase
        print("✓ neo4j driver installed")

        # Test connection
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASS", "password")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))

            with driver.session() as session:
                # Get node counts
                result = session.run("MATCH (c:Concept) RETURN count(c) as concept_count")
                concept_count = result.single()["concept_count"]

                result = session.run("MATCH ()-[r:REL]->() RETURN count(r) as relation_count")
                relation_count = result.single()["relation_count"]

            driver.close()

            return {
                "status": "connected",
                "connection": f"{user}@{uri}",
                "counts": {
                    "concepts": concept_count,
                    "relations": relation_count
                }
            }

        except Exception as e:
            return {
                "status": "connection_failed",
                "error": str(e),
                "suggestion": "Start Neo4j: docker run -d --name neo4j-lnsp -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5"
            }

    except ImportError:
        return {
            "status": "client_missing",
            "suggestion": "Install with: pip install neo4j"
        }

def main():
    print("=== LNSP Database Connection Summary ===\n")

    # Test PostgreSQL
    print("PostgreSQL Status:")
    pg_status = test_postgresql()
    for key, value in pg_status.items():
        print(f"  {key}: {value}")
    print()

    # Test Neo4j
    print("Neo4j Status:")
    neo4j_status = test_neo4j()
    for key, value in neo4j_status.items():
        print(f"  {key}: {value}")
    print()

    # Summary
    pg_ready = pg_status.get("status") == "connected"
    neo4j_ready = neo4j_status.get("status") == "connected"

    if pg_ready and neo4j_ready:
        print("🎉 Both databases are connected and ready!")
        print(f"PostgreSQL tables: {pg_status.get('counts', {})}")
        print(f"Neo4j counts: {neo4j_status.get('counts', {})}")
    elif pg_ready or neo4j_ready:
        print("⚠️ Partial database connectivity")
        if pg_ready:
            print(f"✓ PostgreSQL ready with {len(pg_status.get('tables', []))} tables")
        if neo4j_ready:
            print(f"✓ Neo4j ready with {neo4j_status.get('counts', {}).get('concepts', 0)} concepts")
    else:
        print("❌ No database connections available")
        print("Running in stub mode - install/start databases for real data")

if __name__ == "__main__":
    main()