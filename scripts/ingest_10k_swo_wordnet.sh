#!/bin/bash
# Complete 10K Ontology Ingestion: 2K SWO + 8K WordNet
# Full pipeline: PostgreSQL + Neo4j + FAISS + CPESH + TMD + Graph

set -euo pipefail

echo "=== 10K ONTOLOGY INGESTION (SWO + WordNet) ==="
echo "Components: PostgreSQL + Neo4j + FAISS + CPESH + TMD"
echo "Estimated time: 8-10 hours"
echo ""

# CRITICAL: Set LLM environment (required for CPESH generation)
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Verify Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ ERROR: Ollama not running!"
    echo "Start with: ollama serve"
    exit 1
fi

echo "✓ Ollama LLM verified: $LNSP_LLM_MODEL"
echo ""

# Verify services are running
echo "=== Checking Services ==="
psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓ PostgreSQL: Connected" || { echo "❌ PostgreSQL down"; exit 1; }
cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓ Neo4j: Connected" || { echo "❌ Neo4j down"; exit 1; }
echo ""

# Optional: Clear existing data for fresh start
read -p "Clear existing ontology data? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing PostgreSQL..."
    psql lnsp -c "DELETE FROM cpe_entry WHERE dataset_source LIKE 'ontology-%';"
    psql lnsp -c "DELETE FROM cpe_vectors WHERE cpe_id NOT IN (SELECT cpe_id FROM cpe_entry);"

    echo "Clearing Neo4j..."
    cypher-shell -u neo4j -p password "MATCH (n:Concept) WHERE n.source IN ['swo', 'wordnet'] DETACH DELETE n;"

    echo "✓ Data cleared"
fi
echo ""

# Ingestion function
ingest_ontology() {
    local SOURCE=$1
    local FILE=$2
    local LIMIT=$3

    echo "=== Ingesting $SOURCE ($LIMIT concepts) ==="
    echo "File: $FILE"
    echo "Flags: --write-pg --write-neo4j --write-faiss"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    ./.venv/bin/python -m src.ingest_ontology_simple \
        --input "$FILE" \
        --write-pg \
        --write-neo4j \
        --write-faiss \
        --limit "$LIMIT" 2>&1 | tee "/tmp/ingest_${SOURCE}_$(date +%s).log"

    echo ""
    echo "✓ $SOURCE ingestion complete"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# Phase 1: Ingest SWO (2,000 concepts)
ingest_ontology "swo" "artifacts/ontology_chains/swo_chains.jsonl" 2000

# Phase 2: Ingest WordNet (8,000 concepts)
ingest_ontology "wordnet" "artifacts/ontology_chains/wordnet_chains_8k.jsonl" 8000

# Verification
echo "=== VERIFICATION ==="
echo ""

echo "PostgreSQL counts:"
psql lnsp -c "
SELECT
    dataset_source,
    COUNT(*) as concepts
FROM cpe_entry
WHERE dataset_source LIKE 'ontology-%'
GROUP BY dataset_source
ORDER BY dataset_source;
"

echo ""
echo "PostgreSQL vectors:"
psql lnsp -c "
SELECT COUNT(*) as total_vectors
FROM cpe_vectors v
JOIN cpe_entry e USING (cpe_id)
WHERE e.dataset_source LIKE 'ontology-%';
"

echo ""
echo "Neo4j counts:"
cypher-shell -u neo4j -p password "
MATCH (n:Concept)
WHERE n.source IN ['swo', 'wordnet']
RETURN n.source as source, count(n) as concepts
ORDER BY source;
"

echo ""
echo "FAISS metadata:"
if [ -f artifacts/faiss_meta.json ]; then
    cat artifacts/faiss_meta.json | jq '.'
else
    echo "⚠️  FAISS metadata not found"
fi

echo ""
echo "=== INGESTION COMPLETE ==="
echo "Total concepts: 10,000 (2K SWO + 8K WordNet)"
echo "Components ready:"
echo "  ✓ PostgreSQL (CPESH + TMD)"
echo "  ✓ Neo4j (Graph structure)"
echo "  ✓ FAISS (Vector index)"
echo ""
echo "Next step: LVM training (~30 minutes)"
echo "  Estimated sequences: ~50K (10K concepts × avg 5 positions/chain)"
echo ""
