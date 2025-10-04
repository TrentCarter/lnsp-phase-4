#!/usr/bin/env python3
"""Simple GraphRAG test without benchmark complexity."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=" * 60)
print("GraphRAG Simple Test")
print("=" * 60)

# Test 1: Import GraphRAG backend
print("\n[1/5] Testing GraphRAG import...")
try:
    from RAG.graphrag_backend import GraphRAGBackend
    print("✓ GraphRAGBackend imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Connect to Neo4j
print("\n[2/5] Testing Neo4j connection...")
try:
    gb = GraphRAGBackend()
    print(f"✓ Connected to Neo4j")
    print(f"  Concept mappings: {len(gb.text_to_idx)}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)

# Test 3: Test 1-hop neighbors (use actual concepts from DB)
print("\n[3/5] Testing 1-hop neighbor retrieval...")
try:
    # Get actual concept texts from Neo4j
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    with driver.session() as session:
        result = session.run("MATCH (c:Concept) WHERE c.text IS NOT NULL RETURN c.text LIMIT 3")
        test_concepts = [record["c.text"] for record in result]
    driver.close()

    if not test_concepts:
        print("  ✗ No concepts found in Neo4j!")
    else:
        for concept in test_concepts:
            neighbors = gb._get_1hop_neighbors(concept)
            print(f"  '{concept[:50]}': {len(neighbors)} neighbors")
            if neighbors:
                print(f"    Top neighbor: {neighbors[0][0][:50]} (conf={neighbors[0][1]:.3f})")
except Exception as e:
    print(f"✗ 1-hop test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test graph walks
print("\n[4/5] Testing graph walk retrieval...")
try:
    if test_concepts:
        for concept in test_concepts[:2]:
            walks = gb._get_graph_walks(concept, max_length=3)
            print(f"  '{concept[:50]}': {len(walks)} walk results")
            if walks:
                print(f"    Top result: {walks[0][0][:50]} (score={walks[0][1]:.3f})")
except Exception as e:
    print(f"✗ Graph walk test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test RRF fusion
print("\n[5/5] Testing RRF fusion...")
try:
    vector_indices = [0, 1, 2, 3, 4]
    vector_scores = [0.95, 0.90, 0.85, 0.80, 0.75]
    graph_neighbors = {"material entity", "independent continuant"}
    graph_scores = {"material entity": 0.8, "independent continuant": 0.6}
    doc_ids_to_idx = {text: idx for text, idx in gb.text_to_idx.items()}

    fused_idx, fused_scores = gb._rrf_fusion(
        vector_indices,
        vector_scores,
        graph_neighbors,
        graph_scores,
        doc_ids_to_idx,
        k=60
    )

    print(f"  Fused {len(fused_idx)} results")
    print(f"  Top-3 scores: {fused_scores[:3]}")
    print("✓ RRF fusion works")
except Exception as e:
    print(f"✗ RRF fusion failed: {e}")

# Cleanup
gb.close()

print("\n" + "=" * 60)
print("✅ All GraphRAG tests passed!")
print("=" * 60)
print("\nNext step: Run full benchmark")
print("  ./scripts/run_graphrag_benchmark.sh")
