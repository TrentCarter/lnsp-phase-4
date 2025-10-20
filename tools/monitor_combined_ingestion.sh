#!/bin/bash
# Monitor Combined 18-Hour Ingestion (10hr + 8hr)
# Created: 2025-10-19 1:40 AM

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Combined 18-Hour Ingestion Status${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Phase 1 Status
echo -e "${YELLOW}Phase 1: 10-Hour Run${NC}"
if pgrep -f "ingest_wikipedia_10hr_batched.sh" > /dev/null; then
    echo -e "  Status: ${GREEN}🔄 RUNNING${NC}"
    OFFSET1=$(cat artifacts/wikipedia_10hr_checkpoint.txt 2>/dev/null || echo "Unknown")
    echo "  Current offset: $OFFSET1"
    echo "  Progress: $((OFFSET1 - 3932)) articles from start"
else
    echo -e "  Status: ${GREEN}✓ COMPLETE${NC}"
    FINAL_OFFSET1=$(cat artifacts/wikipedia_10hr_checkpoint.txt 2>/dev/null || echo "Unknown")
    echo "  Final offset: $FINAL_OFFSET1"
    echo "  Total articles: $((FINAL_OFFSET1 - 3932))"
fi

echo ""

# Phase 2 Status
echo -e "${YELLOW}Phase 2: 8-Hour Continuation${NC}"
if pgrep -f "ingest_wikipedia_8hr_continuation.sh" > /dev/null; then
    PID2=$(pgrep -f "ingest_wikipedia_8hr_continuation.sh")

    # Check if it's waiting or running
    if grep -q "Waiting for 10-hour ingestion" logs/wikipedia_8hr_continuation_main.log 2>/dev/null; then
        echo -e "  Status: ${YELLOW}⏳ WAITING for Phase 1${NC}"
        echo "  Will start when Phase 1 completes"
    else
        echo -e "  Status: ${GREEN}🔄 RUNNING${NC}"
        OFFSET2=$(cat artifacts/wikipedia_8hr_checkpoint.txt 2>/dev/null)
        START2=$(head -n 1 logs/wikipedia_8hr_continuation_main.log 2>/dev/null | grep "Start offset" | awk '{print $3}')
        if [ -n "$OFFSET2" ] && [ -n "$START2" ]; then
            echo "  Current offset: $OFFSET2"
            echo "  Progress: $((OFFSET2 - START2)) articles in Phase 2"
        fi
    fi
else
    if [ -f "artifacts/wikipedia_8hr_checkpoint.txt" ]; then
        echo -e "  Status: ${GREEN}✓ COMPLETE${NC}"
        FINAL_OFFSET2=$(cat artifacts/wikipedia_8hr_checkpoint.txt)
        echo "  Final offset: $FINAL_OFFSET2"
    else
        echo -e "  Status: ${YELLOW}⏳ NOT STARTED${NC}"
    fi
fi

echo ""

# Database Stats
echo -e "${YELLOW}Database Status:${NC}"
psql lnsp -c "SELECT COUNT(*) as total_concepts FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" 2>/dev/null

echo ""

# Service Health
echo -e "${YELLOW}Service Health:${NC}"
for port in 8900 8001 8767 8004; do
    if lsof -ti ":$port" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Port $port (active)"
    else
        echo -e "  ${RED}✗${NC} Port $port (down)"
    fi
done

echo ""

# Combined Progress
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Combined Progress${NC}"
echo -e "${YELLOW}========================================${NC}"

CURRENT_CONCEPTS=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" 2>/dev/null | tr -d ' ')
START_CONCEPTS=367378
GROWTH=$((CURRENT_CONCEPTS - START_CONCEPTS))

echo "Starting concepts:  367,378"
echo "Current concepts:   $CURRENT_CONCEPTS"
echo "Growth:            +$GROWTH concepts"
echo ""

# Estimate completion
if pgrep -f "ingest_wikipedia_10hr_batched.sh" > /dev/null; then
    echo -e "${YELLOW}Estimated Completion:${NC}"
    echo "  Phase 1: ~2:49 AM (Oct 19)"
    echo "  Phase 2: ~10:49 AM (Oct 19)"
    echo "  Total: 18 hours from 4:49 PM (Oct 18)"
fi

echo ""
echo "Logs:"
echo "  Phase 1: tail -f logs/wikipedia_10hr_main.log"
echo "  Phase 2: tail -f logs/wikipedia_8hr_continuation_main.log"
echo ""
