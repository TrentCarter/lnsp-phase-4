#!/bin/bash
#
# Limited Ontology Ingestion Script
# Max 2K chains per dataset = ~6K total
# Estimated time: ~7-8 hours
#

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     LIMITED ONTOLOGY INGESTION (Max 2K per dataset)       ║"
echo "║     Total: ~6K chains                                      ║"
echo "║     Estimated time: ~7-8 hours                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Error: Ollama not running"
    echo "   Start with: ollama serve"
    exit 1
fi

echo "✓ Ollama running"
echo ""

# Set environment
# REMOVED: export LNSP_TEST_MODE=1 (was preventing database writes)
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Start timestamp
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run ingestion with 2K limit per dataset
echo "Starting limited ingestion (max 2K per dataset)..."
echo "🚨 CRITICAL: Writing to PostgreSQL + Neo4j + FAISS atomically!"
./.venv/bin/python -m src.ingest_ontology_simple \
    --ingest-all \
    --write-pg \
    --write-neo4j \
    --write-faiss \
    --limit 2000

# End timestamp
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     INGESTION COMPLETE                                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "End time: $(date)"

# Show final stats
echo ""
echo "Database statistics:"
psql lnsp -c "SELECT COUNT(*) as total_cpe_entries FROM cpe_entry;"

