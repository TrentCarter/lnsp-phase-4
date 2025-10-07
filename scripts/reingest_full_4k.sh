#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Full 4K Ontology Re-Ingestion
#
# Re-ingests the complete 4,484-concept ontology dataset with:
# - PostgreSQL storage (CPESH data)
# - Neo4j graph (concepts + edges + vectors)
# - FAISS index (768D GTR-T5 vectors)
#
# Estimated time: ~6-8 hours
# Output: artifacts/ontology_4k_full.{npz,index}
################################################################################

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     FULL 4K ONTOLOGY RE-INGESTION                         ║"
echo "║     PostgreSQL + Neo4j + FAISS                            ║"
echo "║     Estimated time: 6-8 hours                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "=== Checking Prerequisites ==="

# Check Ollama LLM
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama LLM not running. Start with: ollama serve"
    exit 1
fi
echo "✓ Ollama LLM running"

# Check Neo4j
if ! cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1; then
    echo "❌ Neo4j not accessible. Check connection."
    exit 1
fi
echo "✓ Neo4j accessible"

# Check PostgreSQL
if ! psql lnsp -c "SELECT 1" >/dev/null 2>&1; then
    echo "❌ PostgreSQL 'lnsp' database not accessible."
    exit 1
fi
echo "✓ PostgreSQL accessible"

echo ""
echo "=== Clearing Previous Data ==="

# Clear Neo4j graph
echo "Clearing Neo4j graph..."
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n" >/dev/null 2>&1 || true
echo "  ✓ Neo4j graph cleared"

# Clear PostgreSQL tables
echo "Clearing PostgreSQL tables..."
psql lnsp -c "TRUNCATE TABLE cpe_entry CASCADE" >/dev/null 2>&1 || echo "  (cpe_entry already empty)"
psql lnsp -c "TRUNCATE TABLE cpe_vectors CASCADE" >/dev/null 2>&1 || echo "  (cpe_vectors already empty)"

# Remove old FAISS artifacts
echo "Removing old FAISS artifacts..."
rm -f artifacts/ontology_4k_full.npz artifacts/ontology_4k_full.index

echo ""
echo "=== Starting Re-Ingestion ==="
echo "Start time: $(date)"
echo ""

# Configure environment
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Ingest each ontology dataset
DATASETS=(
    "artifacts/ontology_chains/swo_chains.jsonl"
    "artifacts/ontology_chains/go_chains.jsonl"
    "artifacts/ontology_chains/dbpedia_chains.jsonl"
    "artifacts/ontology_chains/conceptnet_chains.jsonl"
)

TOTAL_PROCESSED=0

for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "$dataset" ]]; then
        echo "⚠️  Skipping $dataset (not found)"
        continue
    fi

    dataset_name=$(basename "$dataset" .jsonl)
    echo "───────────────────────────────────────────────────────────"
    echo "Processing: $dataset_name"
    echo "───────────────────────────────────────────────────────────"

    # Run ingestion (no limit = all records)
    ./.venv/bin/python -m src.ingest_ontology_simple \
        --input "$dataset" \
        --write-pg \
        --write-neo4j \
        --write-faiss \
        2>&1 | grep -E "INFO|✓|processed|Error|CRITICAL" | tail -20

    # Count processed
    COUNT=$(psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry")
    echo "  ✓ Total concepts in PostgreSQL: $COUNT"
    TOTAL_PROCESSED=$COUNT
    echo ""
done

echo ""
echo "=== Building FAISS Index ==="

# Save FAISS index and metadata
./.venv/bin/python -c "
from src.db_faiss import FaissDB
import numpy as np
import psycopg2

# Load vectors from PostgreSQL
conn = psycopg2.connect('dbname=lnsp')
cur = conn.cursor()
cur.execute('SELECT cpe_id, concept_vec FROM cpe_vectors ORDER BY id')
rows = cur.fetchall()
cur.close()
conn.close()

cpe_ids = [row[0] for row in rows]
vectors = np.array([row[1] for row in rows], dtype=np.float32)

print(f'Loaded {len(vectors)} vectors from PostgreSQL')
print(f'Vector shape: {vectors.shape}')

# Save to NPZ
np.savez('artifacts/ontology_4k_full.npz',
         concept_vecs=vectors,
         cpe_ids=np.array(cpe_ids),
         doc_ids=np.array(cpe_ids))  # Alias for compatibility

# Build FAISS index
db = FaissDB(index_path='artifacts/ontology_4k_full.index',
             meta_npz_path='artifacts/ontology_4k_full.npz')
db.build_from_npz('artifacts/ontology_4k_full.npz',
                  index_type='ivf_flat',
                  metric='ip',
                  nlist=128)
db.save('artifacts/ontology_4k_full.index')

print('✓ FAISS index saved')
"

echo ""
echo "=== Re-Ingestion Complete ==="
echo "End time: $(date)"
echo ""

# Summary statistics
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    FINAL SUMMARY                           ║"
echo "╚════════════════════════════════════════════════════════════╝"

echo ""
echo "PostgreSQL:"
psql lnsp -c "SELECT
    COUNT(*) as concepts,
    COUNT(DISTINCT dataset_source) as datasets
FROM cpe_entry"

echo ""
echo "Neo4j:"
cypher-shell -u neo4j -p password "
MATCH (c:Concept)
WITH count(c) as concepts
MATCH ()-[r:RELATES_TO]->()
RETURN concepts, count(r) as edges
"

echo ""
echo "FAISS:"
ls -lh artifacts/ontology_4k_full.{npz,index}

echo ""
echo "✅ Re-ingestion complete!"
echo "   To run benchmarks:"
echo "   ./scripts/run_lightrag_benchmark.sh"
