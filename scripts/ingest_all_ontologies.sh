#!/bin/bash
#
# Full Ontology Ingestion Script
# Ingests all 173K chains from SWO, GO, and DBpedia
# Estimated time: ~20 hours
#

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     FULL ONTOLOGY INGESTION (173,029 chains)              ║"
echo "║     Estimated time: ~20 hours                             ║"
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
export LNSP_TEST_MODE=1
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Start timestamp
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run ingestion
echo "Starting full ingestion..."
./.venv/bin/python -m src.ingest_ontology_simple \
    --ingest-all \
    --write-pg

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
psql lnsp -c "SELECT source_type, COUNT(*) FROM cpe_entry GROUP BY source_type ORDER BY source_type;"

