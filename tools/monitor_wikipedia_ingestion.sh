#!/bin/bash
# Monitor Wikipedia ingestion progress in real-time
# Usage: ./tools/monitor_wikipedia_ingestion.sh

echo "🔍 Wikipedia Ingestion Monitor"
echo "================================"
echo ""

INITIAL_COUNT=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')
START_TIME=$(date +%s)

echo "Starting count: $INITIAL_COUNT concepts"
echo "Started at: $(date)"
echo ""
echo "Monitoring progress (Ctrl+C to stop)..."
echo ""

PREV_COUNT=$INITIAL_COUNT

while true; do
    sleep 30  # Update every 30 seconds

    CURRENT_COUNT=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry;" 2>/dev/null | tr -d ' ')
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Calculate stats
    TOTAL_ADDED=$((CURRENT_COUNT - INITIAL_COUNT))
    RECENT_ADDED=$((CURRENT_COUNT - PREV_COUNT))

    # Calculate rate (concepts per second)
    if [ $ELAPSED -gt 0 ]; then
        RATE=$(echo "scale=2; $TOTAL_ADDED / $ELAPSED" | bc)
        ARTICLES_RATE=$(echo "scale=2; ($TOTAL_ADDED / 13) / $ELAPSED" | bc)
    else
        RATE="0"
        ARTICLES_RATE="0"
    fi

    # Calculate ETA for 500k articles (6.5M chunks)
    TARGET_CHUNKS=$((500000 * 13))
    REMAINING_CHUNKS=$((TARGET_CHUNKS - CURRENT_COUNT))

    if [ $(echo "$RATE > 0" | bc) -eq 1 ]; then
        ETA_SECONDS=$(echo "scale=0; $REMAINING_CHUNKS / $RATE" | bc)
        ETA_HOURS=$(echo "scale=1; $ETA_SECONDS / 3600" | bc)
        ETA_DAYS=$(echo "scale=1; $ETA_HOURS / 24" | bc)
    else
        ETA_HOURS="∞"
        ETA_DAYS="∞"
    fi

    # Display progress
    clear
    echo "🔍 Wikipedia Ingestion Monitor"
    echo "================================"
    echo ""
    echo "Time: $(date)"
    echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
    echo ""
    echo "📊 Current Status:"
    echo "   Total concepts: $CURRENT_COUNT"
    echo "   Added this session: $TOTAL_ADDED (+$RECENT_ADDED in last 30s)"
    echo ""
    echo "📈 Performance:"
    echo "   Rate: $RATE chunks/sec (~$ARTICLES_RATE articles/sec)"
    echo ""
    echo "🎯 Progress to 500k articles (6.5M chunks):"
    PERCENT=$(echo "scale=1; ($CURRENT_COUNT * 100) / $TARGET_CHUNKS" | bc)
    echo "   $PERCENT% complete"
    echo "   Remaining: $REMAINING_CHUNKS chunks"
    echo "   ETA: ${ETA_DAYS} days (${ETA_HOURS} hours)"
    echo ""
    echo "💾 Storage check:"
    FAISS_SIZE=$(du -sh artifacts/*.npz 2>/dev/null | awk '{print $1}' || echo "N/A")
    echo "   FAISS vectors: $FAISS_SIZE"
    echo ""

    # Check if target reached
    if [ $CURRENT_COUNT -ge $TARGET_CHUNKS ]; then
        echo "✅ Target reached! Ingestion complete."
        break
    fi

    PREV_COUNT=$CURRENT_COUNT
done
