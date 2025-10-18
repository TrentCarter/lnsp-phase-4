#!/bin/bash
# Batch Wikipedia Ingestion with Progress Tracking
# Processes 500k articles in safe batches with checkpointing
#
# Usage:
#   ./tools/ingest_wikipedia_batched.sh [batch_size] [total_articles]
#
# Example:
#   ./tools/ingest_wikipedia_batched.sh 10000 500000  # 50 batches of 10k each
#   ./tools/ingest_wikipedia_batched.sh 50000 500000  # 10 batches of 50k each

set -e

BATCH_SIZE=${1:-10000}
TOTAL_ARTICLES=${2:-500000}
DATASET="data/datasets/wikipedia/wikipedia_500k.jsonl"
LOG_DIR="logs/wikipedia_ingestion"
METRICS_DIR="artifacts/ingestion_metrics"

# Create directories
mkdir -p "$LOG_DIR" "$METRICS_DIR"

# Calculate batches
NUM_BATCHES=$(( (TOTAL_ARTICLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "🚀 Wikipedia Batch Ingestion"
echo "================================"
echo "Dataset: $DATASET"
echo "Total articles: $TOTAL_ARTICLES"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"
echo "TMD Mode: ${LNSP_TMD_MODE:-hybrid}"
echo ""
echo "Logs: $LOG_DIR"
echo "Metrics: $METRICS_DIR"
echo ""

# Record initial state
INITIAL_COUNT=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')
START_TIME=$(date +%s)
START_DATE=$(date +"%Y-%m-%d %H:%M:%S")

echo "Starting concept count: $INITIAL_COUNT"
echo "Start time: $START_DATE"
echo ""

# Create checkpoint file
CHECKPOINT_FILE="$METRICS_DIR/checkpoint.txt"
if [ -f "$CHECKPOINT_FILE" ]; then
    LAST_BATCH=$(cat "$CHECKPOINT_FILE")
    echo "📍 Resuming from batch $((LAST_BATCH + 1))"
    START_BATCH=$((LAST_BATCH + 1))
else
    START_BATCH=1
fi

# Process batches
for BATCH in $(seq $START_BATCH $NUM_BATCHES); do
    BATCH_START_TIME=$(date +%s)
    BATCH_LOG="$LOG_DIR/batch_${BATCH}.log"

    # Calculate article range
    OFFSET=$(( (BATCH - 1) * BATCH_SIZE ))
    LIMIT=$BATCH_SIZE

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Batch $BATCH / $NUM_BATCHES"
    echo "   Articles: $OFFSET - $((OFFSET + LIMIT))"
    echo "   Log: $BATCH_LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Get concept count before batch
    COUNT_BEFORE=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')

    # Run ingestion for this batch
    LNSP_TMD_MODE=${LNSP_TMD_MODE:-hybrid} \
    ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
        --input "$DATASET" \
        --limit $LIMIT \
        --skip-offset $OFFSET \
        2>&1 | tee "$BATCH_LOG"

    # Check for errors
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ Batch $BATCH failed! Check log: $BATCH_LOG"
        echo "Last successful batch: $((BATCH - 1))" > "$CHECKPOINT_FILE"
        exit 1
    fi

    # Get concept count after batch
    COUNT_AFTER=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')
    BATCH_ADDED=$((COUNT_AFTER - COUNT_BEFORE))

    # Calculate batch timing
    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
    BATCH_RATE=$(echo "scale=2; $BATCH_ADDED / $BATCH_DURATION" | bc)

    # Update checkpoint
    echo "$BATCH" > "$CHECKPOINT_FILE"

    # Calculate overall progress
    TOTAL_ELAPSED=$((BATCH_END_TIME - START_TIME))
    TOTAL_ADDED=$((COUNT_AFTER - INITIAL_COUNT))
    OVERALL_RATE=$(echo "scale=2; $TOTAL_ADDED / $TOTAL_ELAPSED" | bc)

    # Calculate ETA
    REMAINING_BATCHES=$((NUM_BATCHES - BATCH))
    if [ $REMAINING_BATCHES -gt 0 ] && [ $BATCH -gt 0 ]; then
        AVG_BATCH_TIME=$(echo "scale=0; $TOTAL_ELAPSED / $BATCH" | bc)
        ETA_SECONDS=$((AVG_BATCH_TIME * REMAINING_BATCHES))
        ETA_HOURS=$(echo "scale=1; $ETA_SECONDS / 3600" | bc)
    else
        ETA_HOURS="0"
    fi

    echo ""
    echo "✅ Batch $BATCH Complete!"
    echo "   Duration: $((BATCH_DURATION / 60))m $((BATCH_DURATION % 60))s"
    echo "   Concepts added: $BATCH_ADDED"
    echo "   Batch rate: $BATCH_RATE chunks/sec"
    echo ""
    echo "📊 Overall Progress:"
    echo "   Batches: $BATCH / $NUM_BATCHES ($((BATCH * 100 / NUM_BATCHES))%)"
    echo "   Total concepts: $COUNT_AFTER (+$TOTAL_ADDED)"
    echo "   Overall rate: $OVERALL_RATE chunks/sec"
    echo "   Elapsed: $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m"
    echo "   ETA: ${ETA_HOURS}h"
    echo ""

    # Save batch metrics
    cat > "$METRICS_DIR/batch_${BATCH}_metrics.json" <<EOF
{
  "batch": $BATCH,
  "offset": $OFFSET,
  "limit": $LIMIT,
  "duration_seconds": $BATCH_DURATION,
  "concepts_added": $BATCH_ADDED,
  "rate_chunks_per_sec": $BATCH_RATE,
  "total_concepts": $COUNT_AFTER,
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

    # Optional: pause between batches to prevent overload
    if [ $BATCH -lt $NUM_BATCHES ]; then
        sleep 2
    fi
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
FINAL_COUNT=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')
TOTAL_ADDED=$((FINAL_COUNT - INITIAL_COUNT))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 INGESTION COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Start time: $START_DATE"
echo "End time: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Duration: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "Initial concepts: $INITIAL_COUNT"
echo "Final concepts: $FINAL_COUNT"
echo "Added: $TOTAL_ADDED"
echo ""
echo "Average rate: $(echo "scale=2; $TOTAL_ADDED / $TOTAL_DURATION" | bc) chunks/sec"
echo ""

# Save final summary
cat > "$METRICS_DIR/final_summary.json" <<EOF
{
  "start_time": "$START_DATE",
  "end_time": "$(date +"%Y-%m-%d %H:%M:%S")",
  "duration_seconds": $TOTAL_DURATION,
  "initial_concepts": $INITIAL_COUNT,
  "final_concepts": $FINAL_COUNT,
  "concepts_added": $TOTAL_ADDED,
  "batches_completed": $NUM_BATCHES,
  "batch_size": $BATCH_SIZE,
  "average_rate_chunks_per_sec": $(echo "scale=2; $TOTAL_ADDED / $TOTAL_DURATION" | bc)
}
EOF

# Clean up checkpoint
rm -f "$CHECKPOINT_FILE"

echo "📊 Metrics saved to: $METRICS_DIR"
echo "📝 Logs saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Build FAISS index: make build-faiss"
echo "  2. Verify data: ./scripts/verify_data_sync.sh"
echo "  3. Run benchmarks: make slo-grid"
