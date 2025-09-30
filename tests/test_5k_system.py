#!/usr/bin/env python3
"""
5K System Integration Tests

Tests the complete 5K vecRAG system including:
- Database connectivity and integrity
- Vector search performance
- Graph queries
- API endpoints
- CPESH quality

Usage:
    python tests/test_5k_system.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_postgres_connectivity():
    """Test PostgreSQL connection and basic queries."""
    print("\n" + "="*60)
    print("TEST 1: PostgreSQL Connectivity")
    print("="*60)

    try:
        from db_postgres import connect
        conn = connect()
        cur = conn.cursor()

        # Test basic query
        cur.execute("SELECT COUNT(*) FROM cpe_entry;")
        count = cur.fetchone()[0]
        print(f"✅ Connected to PostgreSQL")
        print(f"✅ CPE entries: {count:,}")

        # Test vector table
        cur.execute("SELECT COUNT(*) FROM cpe_vectors;")
        vector_count = cur.fetchone()[0]
        print(f"✅ Vector entries: {vector_count:,}")

        # Test CPESH
        cur.execute("""
            SELECT COUNT(*) as with_cpesh
            FROM cpe_entry
            WHERE jsonb_array_length(soft_negatives) > 0;
        """)
        cpesh_count = cur.fetchone()[0]
        print(f"✅ CPESH coverage: {cpesh_count}/{count} ({cpesh_count/count*100:.1f}%)")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")
        return False


def test_neo4j_connectivity():
    """Test Neo4j connection and graph queries."""
    print("\n" + "="*60)
    print("TEST 2: Neo4j Connectivity & Graph Queries")
    print("="*60)

    try:
        from db_neo4j import Neo4jDB
        neo = Neo4jDB(enabled=True)

        if not neo.enabled or not neo.driver:
            print("❌ Neo4j not connected")
            return False

        with neo.driver.session() as session:
            # Test concept count
            result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            concepts = result.single()["count"]
            print(f"✅ Concept nodes: {concepts:,}")

            # Test entity count
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entities = result.single()["count"]
            print(f"✅ Entity nodes: {entities:,}")

            # Test relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rels = result.single()["count"]
            print(f"✅ Relationships: {rels:,}")

            # Test a sample graph traversal
            start = time.time()
            result = session.run("""
                MATCH (c:Concept)-[r:RELATES_TO]->(e:Entity)
                RETURN c.concept_text, e.entity_name
                LIMIT 10
            """)
            records = list(result)
            elapsed = (time.time() - start) * 1000
            print(f"✅ Sample graph query: {len(records)} results in {elapsed:.1f}ms")

        neo.driver.close()
        return True

    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False


def test_faiss_search():
    """Test Faiss vector search."""
    print("\n" + "="*60)
    print("TEST 3: Faiss Vector Search Performance")
    print("="*60)

    try:
        import numpy as np
        from pathlib import Path

        # Load latest Faiss index
        artifacts = Path("artifacts")
        npz_files = list(artifacts.glob("fw*.npz"))
        if not npz_files:
            print("❌ No Faiss index found")
            return False

        latest = max(npz_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Loading: {latest.name}")

        data = np.load(str(latest))
        vectors = data["fused"]
        print(f"✅ Loaded {vectors.shape[0]:,} vectors ({vectors.shape[1]}D)")

        # Test vector search performance
        query_vector = vectors[0]  # Use first vector as query

        start = time.time()
        # Compute similarities (dot product for normalized vectors)
        similarities = np.dot(vectors, query_vector)
        top_k_indices = np.argsort(similarities)[-10:][::-1]
        elapsed = (time.time() - start) * 1000

        print(f"✅ Vector search: top-10 from {vectors.shape[0]:,} vectors in {elapsed:.1f}ms")
        print(f"   Top similarities: {similarities[top_k_indices][:3]}")

        return True

    except Exception as e:
        print(f"❌ Faiss test failed: {e}")
        return False


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "="*60)
    print("TEST 4: Embedding Generation")
    print("="*60)

    try:
        from vectorizer import EmbeddingBackend

        eb = EmbeddingBackend()

        # Test single embedding
        test_text = "What is artificial intelligence?"
        start = time.time()
        embedding = eb.encode([test_text])[0]
        elapsed = (time.time() - start) * 1000

        print(f"✅ Generated embedding: {embedding.shape} in {elapsed:.1f}ms")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.3f}")

        # Test batch embedding
        test_texts = ["Text " + str(i) for i in range(10)]
        start = time.time()
        embeddings = eb.encode(test_texts)
        elapsed = (time.time() - start) * 1000

        print(f"✅ Batch embedding: {len(test_texts)} texts in {elapsed:.1f}ms")
        print(f"   ({elapsed/len(test_texts):.1f}ms per text)")

        return True

    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


def test_cpesh_quality():
    """Analyze CPESH quality and failure patterns."""
    print("\n" + "="*60)
    print("TEST 5: CPESH Quality Analysis")
    print("="*60)

    try:
        from db_postgres import connect
        conn = connect()
        cur = conn.cursor()

        # Get items without CPESH
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN jsonb_array_length(soft_negatives) = 0 THEN 1 END) as without_cpesh,
                COUNT(CASE WHEN jsonb_array_length(soft_negatives) > 0 THEN 1 END) as with_cpesh
            FROM cpe_entry;
        """)
        stats = cur.fetchone()
        total, without, with_cpesh = stats

        print(f"✅ Total items: {total:,}")
        print(f"✅ With CPESH: {with_cpesh:,} ({with_cpesh/total*100:.1f}%)")
        print(f"⚠️  Without CPESH: {without:,} ({without/total*100:.1f}%)")

        # Analyze CPESH by batch
        cur.execute("""
            SELECT
                batch_id,
                COUNT(*) as items,
                COUNT(CASE WHEN jsonb_array_length(soft_negatives) > 0 THEN 1 END) as with_cpesh
            FROM cpe_entry
            GROUP BY batch_id
            ORDER BY MIN(created_at) DESC;
        """)

        print("\n📊 CPESH by batch:")
        for row in cur.fetchall():
            batch_id, items, cpesh = row
            pct = cpesh/items*100 if items > 0 else 0
            print(f"   {batch_id}: {cpesh}/{items} ({pct:.1f}%)")

        # Sample items without CPESH
        cur.execute("""
            SELECT concept_text, LEFT(source_chunk, 80)
            FROM cpe_entry
            WHERE jsonb_array_length(soft_negatives) = 0
            LIMIT 5;
        """)

        print("\n📝 Sample items without CPESH:")
        for row in cur.fetchall():
            concept, chunk = row
            print(f"   - {concept[:50]}...")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ CPESH analysis failed: {e}")
        return False


def test_data_integrity():
    """Check for data integrity issues."""
    print("\n" + "="*60)
    print("TEST 6: Data Integrity Checks")
    print("="*60)

    try:
        from db_postgres import connect
        conn = connect()
        cur = conn.cursor()

        # Check for duplicates
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT chunk_position->>'doc_id') as unique_docs
            FROM cpe_entry;
        """)
        total, unique = cur.fetchone()
        if total == unique:
            print(f"✅ No duplicates: {total:,} total = {unique:,} unique")
        else:
            print(f"⚠️  Duplicates found: {total:,} total vs {unique:,} unique")

        # Check for orphaned vectors
        cur.execute("""
            SELECT COUNT(*) as orphans
            FROM cpe_vectors v
            WHERE NOT EXISTS (
                SELECT 1 FROM cpe_entry e WHERE e.cpe_id = v.cpe_id
            );
        """)
        orphans = cur.fetchone()[0]
        if orphans == 0:
            print(f"✅ No orphaned vectors")
        else:
            print(f"⚠️  Orphaned vectors: {orphans}")

        # Check for missing vectors
        cur.execute("""
            SELECT COUNT(*) as missing
            FROM cpe_entry e
            WHERE NOT EXISTS (
                SELECT 1 FROM cpe_vectors v WHERE v.cpe_id = e.cpe_id
            );
        """)
        missing = cur.fetchone()[0]
        if missing == 0:
            print(f"✅ No missing vectors")
        else:
            print(f"⚠️  Missing vectors: {missing}")

        # Check vector dimensions
        cur.execute("""
            SELECT
                jsonb_array_length(fused_vec) as fused_dim,
                jsonb_array_length(concept_vec) as concept_dim,
                jsonb_array_length(question_vec) as question_dim
            FROM cpe_vectors
            LIMIT 1;
        """)
        dims = cur.fetchone()
        if dims:
            fused, concept, question = dims
            print(f"✅ Vector dimensions: fused={fused}D, concept={concept}D, question={question}D")
            if fused == 784 and concept == 768 and question == 768:
                print(f"✅ All dimensions correct")
            else:
                print(f"⚠️  Unexpected dimensions")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Integrity check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🔬" + "="*58 + "🔬")
    print("  LNSP 5K System Integration Test Suite")
    print("🔬" + "="*58 + "🔬")

    results = []

    # Run tests
    results.append(("PostgreSQL", test_postgres_connectivity()))
    results.append(("Neo4j", test_neo4j_connectivity()))
    results.append(("Faiss Search", test_faiss_search()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("CPESH Quality", test_cpesh_quality()))
    results.append(("Data Integrity", test_data_integrity()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("\n" + "="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All tests passed! System is healthy.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please investigate.")
        return 1


if __name__ == "__main__":
    import numpy as np  # Import at module level for tests
    sys.exit(main())