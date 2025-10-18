#!/bin/bash
# Monitor Wikipedia Ingestion Progress
# Usage: ./tools/monitor_current_ingestion.sh

echo "=== Wikipedia Ingestion Progress Monitor ==="
echo ""

# Check if ingestion process is running
if pgrep -f "ingest_wikipedia_pipeline.py" > /dev/null; then
    echo "✅ Ingestion process: RUNNING"
else
    echo "⚠️  Ingestion process: NOT RUNNING"
fi

echo ""

# Show latest progress from log
echo "📊 Latest Progress:"
tail -20 logs/wikipedia_ingestion_20251016/ingestion.log 2>/dev/null | grep -E "Articles:|ERROR|✓|✗" | tail -10

echo ""

# Database statistics
echo "📈 Database Statistics:"
psql lnsp -c "SELECT COUNT(*) as total_concepts,
              COUNT(DISTINCT batch_id) as batches,
              MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER)) as max_article_idx
              FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" -t

echo ""
echo "Target: 11,032 articles total (currently processing articles 1,032-11,031)"
echo "Expected completion: ~8 hours from start"
echo ""
echo "To view live log: tail -f logs/wikipedia_ingestion_20251016/ingestion.log"
