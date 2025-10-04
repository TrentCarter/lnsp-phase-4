#!/usr/bin/env bash
# Run GraphRAG vs vecRAG benchmark comparison
# Tests pure vector retrieval vs graph-augmented retrieval

set -e

echo "=== GraphRAG vs vecRAG Benchmark ==="
echo ""

# Check Neo4j is running
if ! cypher-shell -u neo4j -p password "RETURN 1" &>/dev/null; then
    echo "ERROR: Neo4j not running. Start with: brew services start neo4j"
    exit 1
fi

# Check graph has data
CONCEPT_COUNT=$(cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c) as count" --format plain 2>/dev/null | tail -1 || echo "0")
if [ "$CONCEPT_COUNT" -lt 100 ]; then
    echo "ERROR: Neo4j graph has insufficient data ($CONCEPT_COUNT concepts)"
    echo "Run data ingestion first: ./scripts/ingest_10k.sh"
    exit 1
fi

EDGE_COUNT=$(cypher-shell -u neo4j -p password "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count" --format plain 2>/dev/null | tail -1 || echo "0")
echo "✓ Neo4j running with $CONCEPT_COUNT concepts, $EDGE_COUNT edges"
echo "  NOTE: Graph may have limited connectivity (Entity nodes with NULL text)"

# Find NPZ and index
if [ -z "$FAISS_NPZ_PATH" ]; then
    if [ -f "artifacts/fw9k_vectors_tmd_fixed.npz" ]; then
        export FAISS_NPZ_PATH="artifacts/fw9k_vectors_tmd_fixed.npz"
    elif [ -f "artifacts/fw10k_vectors.npz" ]; then
        export FAISS_NPZ_PATH="artifacts/fw10k_vectors.npz"
    else
        echo "ERROR: No NPZ file found. Set FAISS_NPZ_PATH or run vectorization first."
        exit 1
    fi
fi

echo "✓ Using NPZ: $FAISS_NPZ_PATH"

# Install neo4j driver if needed
if ! ./.venv/bin/python -c "import neo4j" 2>/dev/null; then
    echo "Installing neo4j driver..."
    ./.venv/bin/pip install neo4j
fi

# Run benchmark with 3 modes
TIMESTAMP=$(date +%s)
OUTFILE="RAG/results/graphrag_benchmark_${TIMESTAMP}.jsonl"

echo ""
echo "Running benchmark (500 queries, top-10 retrieval)..."
echo "Backends: vec (baseline), graphrag_local (1-hop), graphrag_global (walks), graphrag_hybrid (both)"
echo ""

PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,bm25,graphrag_local,graphrag_global,graphrag_hybrid \
  --out "$OUTFILE"

echo ""
echo "✅ Benchmark complete!"
echo ""
echo "Results saved to: $OUTFILE"
echo "Summary saved to: RAG/results/summary_${TIMESTAMP}.md"
echo ""
echo "Expected improvements over vecRAG baseline:"
echo "  - graphrag_local: +5-10% P@1 (1-hop context)"
echo "  - graphrag_global: +3-8% P@1 (graph walks)"
echo "  - graphrag_hybrid: +10-15% P@1 (combined)"
