#!/bin/bash
# Monitor arXiv download progress

LOG_FILE="logs/arxiv_download_50k.log"
OUTPUT_FILE="data/datasets/arxiv/arxiv_full_50k.jsonl.gz"
TARGET=50000
PID=$(ps aux | grep "download_arxiv.py" | grep -v grep | awk '{print $2}')

echo "==================================================="
echo "arXiv Download Progress Monitor"
echo "==================================================="
echo ""

if [ -z "$PID" ]; then
    echo "❌ Download process NOT running!"
    echo ""
    echo "Last log entry:"
    tail -5 "$LOG_FILE"
    exit 1
fi

echo "✅ Process running (PID: $PID)"
echo ""

# Get current count from log
CURRENT=$(tail -1 "$LOG_FILE" | grep -oE "[0-9]+ records" | grep -oE "[0-9]+" || echo "0")
PERCENT=$(echo "scale=2; $CURRENT * 100 / $TARGET" | bc)

echo "Progress: $CURRENT / $TARGET papers ($PERCENT%)"
echo ""

# Estimate time remaining
if [ "$CURRENT" -gt "0" ]; then
    START_TIME=$(stat -f %B "$LOG_FILE")
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    RATE=$(echo "scale=2; $CURRENT / $ELAPSED" | bc)
    REMAINING=$((TARGET - CURRENT))
    ETA_SEC=$(echo "scale=0; $REMAINING / $RATE" | bc)
    ETA_HOURS=$(echo "scale=1; $ETA_SEC / 3600" | bc)

    echo "Elapsed time: $(($ELAPSED / 3600))h $(($ELAPSED % 3600 / 60))m"
    echo "Download rate: $(printf '%.2f' $RATE) papers/sec"
    echo "Estimated remaining: ${ETA_HOURS}h"
fi

echo ""
echo "Disk usage:"
du -sh data/datasets/arxiv/pdfs/ 2>/dev/null || echo "N/A"

echo ""
echo "Recent papers downloaded:"
ls -lt data/datasets/arxiv/pdfs/*.txt 2>/dev/null | head -5 | awk '{print $9, "(" $5 ")"}'

echo ""
echo "==================================================="
echo "Monitor: tail -f $LOG_FILE"
echo "Stop: kill $PID"
echo "==================================================="
