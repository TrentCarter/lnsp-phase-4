#!/bin/bash
# Auto-start Phase 1 ingestion after 100-article test completes
# This script waits for the current test to finish, cleans the database, and starts Phase 1

set -e

echo "========================================="
echo "Wikipedia Phase 1 Auto-Start Script"
echo "========================================="
echo ""

# Wait for current 100-article test to complete
echo "⏳ Waiting for 100-article validation test to complete..."
while pgrep -f "ingest_wikipedia_pipeline.py.*--limit 100" > /dev/null; do
    sleep 30
    echo "  Still running... ($(date +%H:%M:%S))"
done

echo "✅ 100-article test completed!"
echo ""

# Give it a moment for final writes
sleep 5

# Clean database
echo "🧹 Cleaning database (removing wikipedia_500k test data)..."
psql lnsp -c "
DELETE FROM cpe_vectors WHERE cpe_id IN (SELECT cpe_id FROM cpe_entry WHERE dataset_source = 'wikipedia_500k');
DELETE FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';
" 2>&1 | grep -E "DELETE|ERROR" || true

# Verify clean
count=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
echo "  Database entries remaining: $count"

if [ "$count" != "0" ]; then
    echo "❌ ERROR: Database cleanup failed! Still has $count entries."
    exit 1
fi

echo "✅ Database cleaned successfully"
echo ""

# Start Phase 1 (1,031 articles for 10 hours)
echo "🚀 Starting Phase 1: 1,031 articles (~10 hours)"
echo "  Start time: $(date)"
echo "  Expected completion: $(date -v+10H)"
echo "  Log file: /tmp/wikipedia_phase1_1031.log"
echo ""

cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 1031 \
  > /tmp/wikipedia_phase1_1031.log 2>&1 &

PIPELINE_PID=$!
echo "✅ Phase 1 started! PID: $PIPELINE_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f /tmp/wikipedia_phase1_1031.log"
echo "  ps aux | grep $PIPELINE_PID"
echo ""
echo "Check database progress:"
echo "  psql lnsp -c \"SELECT count(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';\""
echo ""
echo "========================================="
echo "Phase 1 ingestion running in background"
echo "========================================="
