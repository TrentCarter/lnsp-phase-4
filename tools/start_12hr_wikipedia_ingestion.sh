#!/bin/bash
##
# 12-Hour Wikipedia Ingestion
#
# Resumes from article 3,432 and runs for approximately 12 hours
# At ~20s/article average, this should process ~2,000-2,500 articles
##

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
START_ARTICLE=3432
LIMIT=3000  # Target ~2,500 articles in 12 hours (with buffer)
LOG_DIR="logs/wikipedia_12hr_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/ingestion.log"

mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "Starting 12-Hour Wikipedia Ingestion"
echo "================================================================================"
echo ""
echo "Start time: $(date)"
echo "Start article: ${START_ARTICLE}"
echo "Target articles: ${LIMIT}"
echo "Estimated runtime: ~12 hours"
echo "Log file: ${LOG_FILE}"
echo ""

# Check current database state
echo "Current database state:"
psql lnsp -c "SELECT COUNT(*) as concepts FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"
echo ""

# Start ingestion in background
echo "Starting ingestion process..."
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset ${START_ARTICLE} \
  --limit ${LIMIT} \
  > "${LOG_FILE}" 2>&1 &

INGESTION_PID=$!
echo ${INGESTION_PID} > /tmp/wikipedia_12hr_ingestion.pid

echo "✓ Ingestion started with PID: ${INGESTION_PID}"
echo ""
echo "================================================================================"
echo "Monitoring Commands"
echo "================================================================================"
echo ""
echo "# Watch progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "# Check database growth:"
echo "  watch -n 60 'psql lnsp -c \"SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = '\"'\"'wikipedia_500k'\"'\"'\"'"
echo ""
echo "# Check process status:"
echo "  ps -p ${INGESTION_PID}"
echo ""
echo "# Stop ingestion:"
echo "  kill ${INGESTION_PID}"
echo ""
echo "================================================================================"
echo "Expected Completion: $(date -v+12H 2>/dev/null || date -d '+12 hours' 2>/dev/null || echo 'in ~12 hours')"
echo "================================================================================"
echo ""

# Create monitoring script
cat > /tmp/monitor_wikipedia_ingestion.sh << 'EOF'
#!/bin/bash
LOG_FILE="$1"
START_COUNT="$2"

while true; do
    sleep 300  # Check every 5 minutes

    CURRENT_COUNT=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
    NEW_CONCEPTS=$((CURRENT_COUNT - START_COUNT))

    # Get last log line
    LAST_LINE=$(tail -1 "${LOG_FILE}" 2>/dev/null || echo "N/A")

    clear
    echo "================================================================================"
    echo "Wikipedia Ingestion Monitor - $(date)"
    echo "================================================================================"
    echo ""
    echo "Concepts: ${CURRENT_COUNT} (started at ${START_COUNT}, +${NEW_CONCEPTS} new)"
    echo ""
    echo "Last log entry:"
    echo "  ${LAST_LINE}"
    echo ""
    echo "Press Ctrl+C to exit monitor (ingestion will continue)"
    echo "================================================================================"
done
EOF

chmod +x /tmp/monitor_wikipedia_ingestion.sh

CURRENT_COUNT=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
echo "To monitor progress interactively:"
echo "  /tmp/monitor_wikipedia_ingestion.sh \"${LOG_FILE}\" ${CURRENT_COUNT}"
echo ""
