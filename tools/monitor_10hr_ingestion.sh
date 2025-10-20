#!/bin/bash
# Monitor 10-Hour Wikipedia Ingestion Progress
# Created: 2025-10-18

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CHECKPOINT_FILE="artifacts/wikipedia_10hr_checkpoint.txt"
LOG_DIR="logs/wikipedia_10hr"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}10-Hour Ingestion Progress Monitor${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if running
if pgrep -f "ingest_wikipedia_10hr_batched.sh" > /dev/null; then
    echo -e "${GREEN}✓ Ingestion is RUNNING${NC}"
    RUNNING_PID=$(pgrep -f "ingest_wikipedia_10hr_batched.sh")
    echo "  PID: $RUNNING_PID"
else
    echo -e "${YELLOW}✗ Ingestion is NOT running${NC}"
fi

echo ""

# Check checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    CURRENT_OFFSET=$(cat "$CHECKPOINT_FILE")
    echo "Current offset: $CURRENT_OFFSET"
    echo "Progress: $((CURRENT_OFFSET - 3932)) articles processed since start"
else
    echo "No checkpoint file found"
fi

echo ""

# Database stats
echo -e "${YELLOW}Database Stats:${NC}"
psql lnsp -c "SELECT COUNT(*) as total_concepts FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"

echo ""

# Recent batches
echo -e "${YELLOW}Recent Batches (last 5):${NC}"
if [ -d "$LOG_DIR" ]; then
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -n 5 | while read line; do
        LOG_FILE=$(echo "$line" | awk '{print $NF}')
        BATCH_NUM=$(basename "$LOG_FILE" | sed 's/batch_\([0-9]*\).*/\1/')

        if [ -f "$LOG_FILE" ]; then
            ARTICLES=$(grep "Articles processed:" "$LOG_FILE" 2>/dev/null | tail -n 1 | awk '{print $3}')
            CHUNKS=$(grep "Chunks ingested:" "$LOG_FILE" 2>/dev/null | tail -n 1 | awk '{print $3}')

            if [ -n "$ARTICLES" ]; then
                echo "  Batch #$BATCH_NUM: $ARTICLES articles, $CHUNKS chunks"
            fi
        fi
    done
else
    echo "No log directory found yet"
fi

echo ""

# Service health
echo -e "${YELLOW}Service Health:${NC}"
for port in 8900 8001 8767 8004; do
    if lsof -ti ":$port" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Port $port (active)"
    else
        echo -e "  ${YELLOW}✗${NC} Port $port (down)"
    fi
done

echo ""
echo "To view live logs:"
echo "  tail -f logs/wikipedia_10hr/*.log"
echo ""
