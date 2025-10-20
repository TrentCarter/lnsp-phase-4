#!/bin/bash
# 10-Hour Wikipedia Ingestion with Small Batches
# Strategy: 100 articles per batch, checkpoint between batches
# Created: 2025-10-18

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
START_OFFSET=3932  # Start after last successful batch (3,431 + 499 + 1)
BATCH_SIZE=100     # Small batches for safety
TARGET_HOURS=10
LOG_DIR="logs/wikipedia_10hr"
CHECKPOINT_FILE="artifacts/wikipedia_10hr_checkpoint.txt"

# Calculate total articles to process
# Previous rate: ~9.2 articles/minute = 552 articles/hour
TARGET_ARTICLES=$((552 * TARGET_HOURS))

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}10-Hour Wikipedia Ingestion (Batched)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Start offset: $START_OFFSET"
echo "Batch size: $BATCH_SIZE articles"
echo "Target duration: $TARGET_HOURS hours"
echo "Estimated total: ~$TARGET_ARTICLES articles"
echo "Checkpoint file: $CHECKPOINT_FILE"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize checkpoint
echo "$START_OFFSET" > "$CHECKPOINT_FILE"

# Start time
START_TIME=$(date +%s)
END_TIME=$((START_TIME + TARGET_HOURS * 3600))

# Restart services (CRITICAL: clear memory before long run)
echo -e "${YELLOW}Step 1: Restarting FastAPI services...${NC}"
./scripts/stop_all_fastapi_services.sh
sleep 5
./scripts/start_all_fastapi_services.sh
sleep 10
echo ""

echo -e "${GREEN}✓ Services restarted${NC}"
echo ""

# Batch counter
BATCH_NUM=0
CURRENT_OFFSET=$START_OFFSET
TOTAL_PROCESSED=0
TOTAL_ERRORS=0

echo -e "${YELLOW}Starting ingestion batches...${NC}"
echo ""

while [ $(date +%s) -lt $END_TIME ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_START=$(date +%s)

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Batch #$BATCH_NUM - Articles $CURRENT_OFFSET to $((CURRENT_OFFSET + BATCH_SIZE - 1))${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Calculate time remaining
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((TARGET_HOURS * 3600 - ELAPSED))
    HOURS_LEFT=$((REMAINING / 3600))
    MINS_LEFT=$(((REMAINING % 3600) / 60))

    echo "Time remaining: ${HOURS_LEFT}h ${MINS_LEFT}m"
    echo "Progress: $TOTAL_PROCESSED articles processed, $TOTAL_ERRORS errors"
    echo ""

    # Run batch
    LOG_FILE="$LOG_DIR/batch_${BATCH_NUM}_offset_${CURRENT_OFFSET}.log"

    LNSP_TMD_MODE=hybrid \
    LNSP_LLM_ENDPOINT="http://localhost:11434" \
    LNSP_LLM_MODEL="llama3.1:8b" \
    ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
        --input data/datasets/wikipedia/wikipedia_500k.jsonl \
        --skip-offset "$CURRENT_OFFSET" \
        --limit "$BATCH_SIZE" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        # Success
        ARTICLES_PROCESSED=$(grep "Articles processed:" "$LOG_FILE" | tail -n 1 | awk '{print $3}')
        CHUNKS_INGESTED=$(grep "Chunks ingested:" "$LOG_FILE" | tail -n 1 | awk '{print $3}')

        echo -e "${GREEN}✓ Batch #$BATCH_NUM completed${NC}"
        echo "  Articles: $ARTICLES_PROCESSED"
        echo "  Chunks: $CHUNKS_INGESTED"

        TOTAL_PROCESSED=$((TOTAL_PROCESSED + ARTICLES_PROCESSED))
        CURRENT_OFFSET=$((CURRENT_OFFSET + BATCH_SIZE))

        # Update checkpoint
        echo "$CURRENT_OFFSET" > "$CHECKPOINT_FILE"

    else
        # Error
        echo -e "${RED}✗ Batch #$BATCH_NUM failed (exit code: $EXIT_CODE)${NC}"
        echo "  Check log: $LOG_FILE"
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))

        # Continue to next batch (don't stop on single failure)
        CURRENT_OFFSET=$((CURRENT_OFFSET + BATCH_SIZE))
    fi

    # Batch stats
    BATCH_DURATION=$(($(date +%s) - BATCH_START))
    echo "  Duration: ${BATCH_DURATION}s"
    echo ""

    # Memory check every 5 batches
    if [ $((BATCH_NUM % 5)) -eq 0 ]; then
        echo -e "${YELLOW}Memory check (batch $BATCH_NUM)...${NC}"
        ps aux | grep -E "(episode_chunker|semantic_chunker|gtr_t5|ingest_chunks)" | grep -v grep | awk '{print $2, $3, $4, $11}'
        echo ""
    fi

    # Restart services every 50 batches (prevent memory leaks)
    if [ $((BATCH_NUM % 50)) -eq 0 ]; then
        echo -e "${YELLOW}Restarting services (preventive maintenance)...${NC}"
        ./scripts/stop_all_fastapi_services.sh
        sleep 5
        ./scripts/start_all_fastapi_services.sh
        sleep 10
        echo ""
    fi

    # Brief pause between batches (prevent service overload)
    sleep 2
done

# End time
TOTAL_TIME=$(($(date +%s) - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}10-Hour Ingestion Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Total batches: $BATCH_NUM"
echo "Total articles: $TOTAL_PROCESSED"
echo "Total errors: $TOTAL_ERRORS"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Final offset: $CURRENT_OFFSET"
echo ""
echo "Checkpoint saved to: $CHECKPOINT_FILE"
echo "Logs saved to: $LOG_DIR/"
echo ""

# Final database check
echo -e "${YELLOW}Final database state:${NC}"
psql lnsp -c "SELECT COUNT(*) as total_concepts FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"
echo ""

echo -e "${GREEN}Next steps:${NC}"
echo "1. Export new training data:"
echo "   ./.venv/bin/python tools/export_lvm_training_data.py"
echo ""
echo "2. Retrain models with larger dataset"
echo ""
