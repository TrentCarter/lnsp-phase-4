#!/bin/bash
##
# Resume Wikipedia Ingestion from Checkpoint
#
# Last completed: 2025-10-23 (Bulk ingestion)
# Articles in DB: 8,447 / 8,470 attempted (99.73% success)
# Next article: 8,471
# Missing articles: 421, 7151, 7691 (JSON encoding failures)
#
# Usage:
#   ./tools/resume_wikipedia_ingestion.sh [NUM_ARTICLES]
#
# Example:
#   ./tools/resume_wikipedia_ingestion.sh 10000  # Resume 10k more articles
#   ./tools/resume_wikipedia_ingestion.sh        # Default: 10000 articles
##

set -euo pipefail

# Configuration
INPUT_FILE="data/datasets/wikipedia/wikipedia_500k.jsonl"
SKIP_OFFSET=8470  # Next article to process (last attempted was 8,470)
LIMIT=${1:-10000}  # Default: 10,000 articles (~9.4 hours)

# Verify source data exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ ERROR: Source data not found!"
    echo "   Expected: $INPUT_FILE"
    exit 1
fi

# Check database connectivity
if ! psql lnsp -c "SELECT 1" >/dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to PostgreSQL database 'lnsp'"
    echo "   Make sure PostgreSQL is running"
    exit 1
fi

# Get current database state
echo "=========================================="
echo "Wikipedia Ingestion Resume Script"
echo "=========================================="
echo ""
echo "Current database state:"
CURRENT_ARTICLES=$(psql lnsp -t -c "SELECT COUNT(DISTINCT chunk_position->>'article_index') FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
CURRENT_CHUNKS=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
echo "  Articles: $CURRENT_ARTICLES"
echo "  Chunks: $CURRENT_CHUNKS"
echo ""

# Confirm parameters
END_ARTICLE=$((SKIP_OFFSET + LIMIT))
ESTIMATED_HOURS=$(echo "scale=1; $LIMIT * 3.4 / 3600" | bc)
echo "Resumption parameters:"
echo "  Start article: $((SKIP_OFFSET + 1))"
echo "  End article: $END_ARTICLE"
echo "  Number to process: $LIMIT"
echo "  Estimated time: ~$ESTIMATED_HOURS hours"
echo "=========================================="
echo ""

# Prompt for confirmation
read -p "Continue with ingestion? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/wikipedia_resume_${SKIP_OFFSET}_$(date +%Y%m%d_%H%M%S).log"

# Set environment variables
export LNSP_TMD_MODE=heuristic  # Don't need full CPESH for Wikipedia
export OMP_NUM_THREADS=8        # Optimize for multi-core

# Start ingestion
echo ""
echo "Starting Wikipedia bulk ingestion..."
echo "Log file: $LOG_FILE"
echo ""

nohup ./.venv/bin/python tools/ingest_wikipedia_bulk.py \
  --input "$INPUT_FILE" \
  --skip-offset "$SKIP_OFFSET" \
  --limit "$LIMIT" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > /tmp/wikipedia_ingestion.pid

echo "=========================================="
echo "✅ Ingestion started!"
echo "=========================================="
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check article count:"
echo "  psql lnsp -c \"SELECT COUNT(DISTINCT chunk_position->>'article_index') FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';\""
echo ""
echo "Check chunk count:"
echo "  psql lnsp -c \"SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';\""
echo ""
echo "Stop ingestion:"
echo "  kill $PID"
echo "=========================================="
