#!/bin/bash
##
# Resume Wikipedia Ingestion from Checkpoint
#
# Last checkpoint: 2025-10-16
# Articles processed: 3,425
# Next article: 3,426
#
# Usage:
#   ./tools/resume_wikipedia_ingestion.sh [NUM_ARTICLES]
#
# Example:
#   ./tools/resume_wikipedia_ingestion.sh 7000  # Resume 7k more articles
#   ./tools/resume_wikipedia_ingestion.sh       # Default: 7000 articles
##

set -e

# Configuration
CHECKPOINT_FILE="WIKIPEDIA_INGESTION_CHECKPOINT.md"
INPUT_FILE="data/datasets/wikipedia/wikipedia_500k.jsonl"
SKIP_OFFSET=3426  # Next article to process
LIMIT=${1:-7000}  # Default: 7,000 articles (30-40 hours)

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "⚠️  Warning: Checkpoint file not found"
    echo "Looking for: $CHECKPOINT_FILE"
fi

# Get current database state
echo "================================"
echo "Wikipedia Ingestion Resume"
echo "================================"
echo ""
echo "Current database state:"
psql lnsp -t -c "
SELECT
  'Concepts: ' || COUNT(*) ||
  ' | Batches: ' || COUNT(DISTINCT batch_id) ||
  ' | Max Article: ' || MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER))
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k';"
echo ""

# Confirm parameters
echo "Resumption parameters:"
echo "  Starting from article: $SKIP_OFFSET"
echo "  Number to process: $LIMIT"
echo "  Estimated time: ~$(echo "scale=1; $LIMIT * 18 / 3600" | bc) hours"
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
LOG_FILE="$LOG_DIR/wikipedia_ingestion_$(date +%Y%m%d_%H%M%S).log"

# Start ingestion
echo ""
echo "Starting Wikipedia ingestion..."
echo "Log file: $LOG_FILE"
echo ""

LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input "$INPUT_FILE" \
  --skip-offset "$SKIP_OFFSET" \
  --limit "$LIMIT" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > /tmp/wikipedia_ingestion.pid

echo "✓ Ingestion started!"
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop ingestion:"
echo "  kill -SIGTERM $PID"
echo ""
echo "Check database:"
echo "  psql lnsp -c \"SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';\""
