#!/usr/bin/env bash
#
# Overnight Wikipedia Ingestion - Continue from article 3,432
#
# Goal: Ingest 5,000-10,000 more Wikipedia articles overnight (~8 hours)
# This will add ~50k-100k new concepts/vectors to the database
# Tomorrow: Re-export training data → more sequences for Phase-3.5 retry
#
# Usage:
#   ./tools/overnight_wikipedia_ingestion.sh
#

set -euo pipefail

# Configuration
WIKI_DATA="data/datasets/wikipedia/wikipedia_500k.jsonl"
START_ARTICLE=3432  # Continue from last checkpoint
BATCH_SIZE=1000     # Articles per batch
NUM_BATCHES=8       # 8 batches × 1,000 articles = 8,000 articles total
TOTAL_ARTICLES=$((BATCH_SIZE * NUM_BATCHES))

# API endpoints must be running
REQUIRED_APIS=(
    "http://localhost:8900/health"  # Episode Chunker
    "http://localhost:8001/health"  # Semantic Chunker
    "http://localhost:8767/health"  # GTR-T5 Embeddings
    "http://localhost:8004/health"  # Ingest API
)

# Logging
LOG_DIR="/tmp/lnsp_overnight_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/overnight_ingestion_$(date +%Y%m%d_%H%M%S).log"

echo "🌙 OVERNIGHT WIKIPEDIA INGESTION" | tee -a "$MAIN_LOG"
echo "=================================" | tee -a "$MAIN_LOG"
echo "Start time: $(date)" | tee -a "$MAIN_LOG"
echo "Starting article: $START_ARTICLE" | tee -a "$MAIN_LOG"
echo "Total articles to ingest: $TOTAL_ARTICLES (in $NUM_BATCHES batches of $BATCH_SIZE)" | tee -a "$MAIN_LOG"
echo "Expected completion: ~8 hours" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Check all APIs are running
echo "🔍 Checking API health..." | tee -a "$MAIN_LOG"
for api in "${REQUIRED_APIS[@]}"; do
    if ! curl -s "$api" > /dev/null 2>&1; then
        echo "❌ ERROR: API not responding: $api" | tee -a "$MAIN_LOG"
        echo "Please start all FastAPI services first:" | tee -a "$MAIN_LOG"
        echo "  ./scripts/start_all_fastapi_services.sh" | tee -a "$MAIN_LOG"
        exit 1
    fi
    echo "  ✓ $(echo $api | cut -d'/' -f3) responding" | tee -a "$MAIN_LOG"
done
echo "" | tee -a "$MAIN_LOG"

# Run batches sequentially
echo "🚀 Starting batch ingestion..." | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

CURRENT_OFFSET=$START_ARTICLE
for batch_num in $(seq 1 $NUM_BATCHES); do
    BATCH_LOG="$LOG_DIR/batch_${batch_num}_$(date +%Y%m%d_%H%M%S).log"

    echo "📦 BATCH $batch_num/$NUM_BATCHES" | tee -a "$MAIN_LOG"
    echo "  Articles: $CURRENT_OFFSET to $((CURRENT_OFFSET + BATCH_SIZE - 1))" | tee -a "$MAIN_LOG"
    echo "  Start time: $(date)" | tee -a "$MAIN_LOG"
    echo "  Log: $BATCH_LOG" | tee -a "$MAIN_LOG"

    # Run ingestion for this batch
    LNSP_TMD_MODE=hybrid \
    LNSP_LLM_ENDPOINT="http://localhost:11434" \
    LNSP_LLM_MODEL="llama3.1:8b" \
    ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
        --input "$WIKI_DATA" \
        --skip-offset $CURRENT_OFFSET \
        --limit $BATCH_SIZE \
        > "$BATCH_LOG" 2>&1

    BATCH_EXIT_CODE=$?

    if [ $BATCH_EXIT_CODE -eq 0 ]; then
        echo "  ✅ Batch $batch_num completed successfully" | tee -a "$MAIN_LOG"
        echo "  Completion time: $(date)" | tee -a "$MAIN_LOG"

        # Show batch stats from log
        if grep -q "Pipeline complete" "$BATCH_LOG" 2>/dev/null; then
            echo "  Stats:" | tee -a "$MAIN_LOG"
            tail -20 "$BATCH_LOG" | grep -E "(articles processed|concepts added|avg.*ms)" | sed 's/^/    /' | tee -a "$MAIN_LOG"
        fi
    else
        echo "  ❌ Batch $batch_num FAILED (exit code: $BATCH_EXIT_CODE)" | tee -a "$MAIN_LOG"
        echo "  Check log for details: $BATCH_LOG" | tee -a "$MAIN_LOG"

        # Save checkpoint
        echo "  💾 Saving checkpoint at article $CURRENT_OFFSET" | tee -a "$MAIN_LOG"
        echo "$CURRENT_OFFSET" > "$LOG_DIR/last_successful_article.txt"

        echo "" | tee -a "$MAIN_LOG"
        echo "⚠️  INGESTION STOPPED due to batch failure" | tee -a "$MAIN_LOG"
        echo "To resume, update START_ARTICLE to $CURRENT_OFFSET in this script" | tee -a "$MAIN_LOG"
        exit 1
    fi

    echo "" | tee -a "$MAIN_LOG"

    # Update offset for next batch
    CURRENT_OFFSET=$((CURRENT_OFFSET + BATCH_SIZE))

    # Brief pause between batches (5 seconds)
    if [ $batch_num -lt $NUM_BATCHES ]; then
        echo "  ⏸️  Pausing 5 seconds before next batch..." | tee -a "$MAIN_LOG"
        sleep 5
    fi
done

# Final summary
echo "=" | tee -a "$MAIN_LOG"
echo "🎉 OVERNIGHT INGESTION COMPLETE!" | tee -a "$MAIN_LOG"
echo "=================================" | tee -a "$MAIN_LOG"
echo "End time: $(date)" | tee -a "$MAIN_LOG"
echo "Total articles processed: $TOTAL_ARTICLES" | tee -a "$MAIN_LOG"
echo "Final article index: $((START_ARTICLE + TOTAL_ARTICLES - 1))" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Query database for updated stats
echo "📊 Database stats after ingestion:" | tee -a "$MAIN_LOG"
if command -v psql &> /dev/null; then
    psql lnsp -c "
        SELECT
            COUNT(*) as total_vectors,
            COUNT(DISTINCT id) as unique_concepts
        FROM cpe_vectors
        WHERE id IN (
            SELECT id FROM cpe_entry WHERE dataset_source = 'wikipedia-500k'
        );
    " 2>/dev/null | tee -a "$MAIN_LOG" || echo "  (Database query failed)" | tee -a "$MAIN_LOG"
fi
echo "" | tee -a "$MAIN_LOG"

echo "✅ Next steps:" | tee -a "$MAIN_LOG"
echo "  1. Verify data quality: ./tools/verify_ingestion_quality.sh" | tee -a "$MAIN_LOG"
echo "  2. Re-export training data with larger dataset:" | tee -a "$MAIN_LOG"
echo "     # 2000-context export (should get ~1,500-1,800 sequences now!)" | tee -a "$MAIN_LOG"
echo "     ./.venv/bin/python tools/export_lvm_training_data_extended.py \\" | tee -a "$MAIN_LOG"
echo "       --input artifacts/wikipedia_500k_corrected_vectors.npz \\" | tee -a "$MAIN_LOG"
echo "       --context-length 2000 \\" | tee -a "$MAIN_LOG"
echo "       --overlap 1000 \\" | tee -a "$MAIN_LOG"
echo "       --output-dir artifacts/lvm/data_phase3.5_retry/" | tee -a "$MAIN_LOG"
echo "  3. Retry Phase-3.5 training with new data" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "📁 All logs saved to: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"
