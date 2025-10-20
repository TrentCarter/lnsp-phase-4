#!/bin/bash
# 8-Hour Wikipedia Ingestion Continuation
# Starts automatically after 10-hour run completes
# Created: 2025-10-19 1:35 AM

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Wait for first ingestion to complete
echo -e "${YELLOW}Waiting for 10-hour ingestion to complete...${NC}"
while pgrep -f "ingest_wikipedia_10hr_batched.sh" > /dev/null; do
    sleep 60
    echo "  Still running... checking again in 60s"
done

echo -e "${GREEN}✓ Previous ingestion completed!${NC}"
echo ""
sleep 10

# Configuration
CHECKPOINT_FILE="artifacts/wikipedia_10hr_checkpoint.txt"
START_OFFSET=$(cat "$CHECKPOINT_FILE" 2>/dev/null || echo "11632")
BATCH_SIZE=100
TARGET_HOURS=8
LOG_DIR="logs/wikipedia_8hr_continuation"
NEW_CHECKPOINT="artifacts/wikipedia_8hr_checkpoint.txt"

# Calculate target articles
TARGET_ARTICLES=$((14 * 60 * TARGET_HOURS))  # 14 articles/min based on actual performance

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}8-Hour Continuation Ingestion${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Start offset: $START_OFFSET (from previous run)"
echo "Batch size: $BATCH_SIZE articles"
echo "Target duration: $TARGET_HOURS hours"
echo "Estimated total: ~$TARGET_ARTICLES articles"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize checkpoint
echo "$START_OFFSET" > "$NEW_CHECKPOINT"

# Start time
START_TIME=$(date +%s)
END_TIME=$((START_TIME + TARGET_HOURS * 3600))

# Restart services (CRITICAL: fresh start after 10 hours)
echo -e "${YELLOW}Restarting FastAPI services...${NC}"
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

echo -e "${YELLOW}Starting continuation batches...${NC}"
echo ""

while [ $(date +%s) -lt $END_TIME ]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    BATCH_START=$(date +%s)

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Batch #$BATCH_NUM - Articles $CURRENT_OFFSET to $((CURRENT_OFFSET + BATCH_SIZE - 1))${NC}"
    echo -e "${YELLOW}========================================${NC}"

    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((TARGET_HOURS * 3600 - ELAPSED))
    HOURS_LEFT=$((REMAINING / 3600))
    MINS_LEFT=$(((REMAINING % 3600) / 60))

    echo "Time remaining: ${HOURS_LEFT}h ${MINS_LEFT}m"
    echo "Progress: $TOTAL_PROCESSED articles processed, $TOTAL_ERRORS errors"
    echo ""

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
        ARTICLES_PROCESSED=$(grep "Articles processed:" "$LOG_FILE" | tail -n 1 | awk '{print $3}')
        CHUNKS_INGESTED=$(grep "Chunks ingested:" "$LOG_FILE" | tail -n 1 | awk '{print $3}')

        echo -e "${GREEN}✓ Batch #$BATCH_NUM completed${NC}"
        echo "  Articles: $ARTICLES_PROCESSED"
        echo "  Chunks: $CHUNKS_INGESTED"

        TOTAL_PROCESSED=$((TOTAL_PROCESSED + ARTICLES_PROCESSED))
        CURRENT_OFFSET=$((CURRENT_OFFSET + BATCH_SIZE))
        echo "$CURRENT_OFFSET" > "$NEW_CHECKPOINT"
    else
        echo -e "${RED}✗ Batch #$BATCH_NUM failed (exit code: $EXIT_CODE)${NC}"
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
        CURRENT_OFFSET=$((CURRENT_OFFSET + BATCH_SIZE))
    fi

    BATCH_DURATION=$(($(date +%s) - BATCH_START))
    echo "  Duration: ${BATCH_DURATION}s"
    echo ""

    # Memory check every 5 batches
    if [ $((BATCH_NUM % 5)) -eq 0 ]; then
        echo -e "${YELLOW}Memory check (batch $BATCH_NUM)...${NC}"
        ps aux | grep -E "(episode_chunker|semantic_chunker|gtr_t5|ingest_chunks)" | grep -v grep | awk '{print $2, $3, $4, $11}'
        echo ""
    fi

    # Restart services every 50 batches
    if [ $((BATCH_NUM % 50)) -eq 0 ]; then
        echo -e "${YELLOW}Restarting services...${NC}"
        ./scripts/stop_all_fastapi_services.sh
        sleep 5
        ./scripts/start_all_fastapi_services.sh
        sleep 10
        echo ""
    fi

    sleep 2
done

# Summary
TOTAL_TIME=$(($(date +%s) - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}8-Hour Continuation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Total batches: $BATCH_NUM"
echo "Total articles: $TOTAL_PROCESSED"
echo "Total errors: $TOTAL_ERRORS"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Final offset: $CURRENT_OFFSET"
echo ""

# Final database check
echo -e "${YELLOW}Final database state:${NC}"
psql lnsp -c "SELECT COUNT(*) as total_concepts FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"
echo ""

echo -e "${GREEN}COMBINED TOTAL (10hr + 8hr):${NC}"
echo "  Duration: 18 hours"
echo "  Estimated articles: ~15,000+"
echo "  Estimated concepts: ~600,000+"
echo ""
