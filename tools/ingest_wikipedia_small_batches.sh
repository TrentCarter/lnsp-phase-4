#!/bin/bash
##
# Wikipedia Ingestion in Small, Proven-Sized Batches
#
# SOLUTION 2: Split large ingestion jobs into small, reliable batches.
#
# KEY FEATURES:
# - Processes Wikipedia in batches of 250-500 articles (proven reliable size)
# - Each batch runs independently and completes within 1-2 hours
# - Failed batches can be retried individually without losing progress
# - Automatic progress tracking via database queries
#
# USAGE:
#   # Ingest articles 3,432 to 6,431 in batches of 500
#   ./tools/ingest_wikipedia_small_batches.sh --start 3432 --total 3000 --batch-size 500
#
#   # Resume from where it left off
#   ./tools/ingest_wikipedia_small_batches.sh --resume --total 3000
##

set -e

# Configuration
BATCH_SIZE=500          # Articles per batch (can override with --batch-size)
START_ARTICLE=0         # Starting article number (can override with --start)
TOTAL_ARTICLES=3000     # Total articles to ingest (can override with --total)
RESUME_MODE=0           # Resume from last completed batch (--resume)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --start)
      START_ARTICLE="$2"
      shift 2
      ;;
    --total)
      TOTAL_ARTICLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --resume)
      RESUME_MODE=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--start N] [--total N] [--batch-size N] [--resume]"
      exit 1
      ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/wikipedia_batched_${TIMESTAMP}"
PROGRESS_FILE="${LOG_DIR}/progress.txt"

mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "Wikipedia Ingestion - Small Batch Mode"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Batch size: ${BATCH_SIZE} articles"
echo "  Total target: ${TOTAL_ARTICLES} articles"
echo "  Log directory: ${LOG_DIR}"
echo ""

# Resume mode: query database for last completed article
if [ "${RESUME_MODE}" -eq 1 ]; then
  echo "🔄 Resume mode enabled - querying database for last article..."

  LAST_ARTICLE=$(psql lnsp -t -c "
    SELECT MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER))
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k' AND batch_id LIKE 'wikipedia_%'
  " | xargs)

  if [ -z "${LAST_ARTICLE}" ] || [ "${LAST_ARTICLE}" = "NULL" ]; then
    echo "  ⚠️  No articles found in database, starting from 0"
    START_ARTICLE=0
  else
    START_ARTICLE=$((LAST_ARTICLE + 1))
    echo "  ✓ Last completed article: ${LAST_ARTICLE}"
    echo "  ✓ Resuming from article: ${START_ARTICLE}"
  fi
fi

# Calculate number of batches needed
END_ARTICLE=$((START_ARTICLE + TOTAL_ARTICLES))
NUM_BATCHES=$(( (TOTAL_ARTICLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo ""
echo "Batch Plan:"
echo "  Start article: ${START_ARTICLE}"
echo "  End article: ${END_ARTICLE}"
echo "  Number of batches: ${NUM_BATCHES}"
echo "  Estimated time: $((NUM_BATCHES * 2)) hours (at ~2 hours per batch)"
echo ""

# Initialize progress file
echo "# Wikipedia Ingestion Progress" > "${PROGRESS_FILE}"
echo "# Started: $(date)" >> "${PROGRESS_FILE}"
echo "# Start article: ${START_ARTICLE}" >> "${PROGRESS_FILE}"
echo "# Target: ${TOTAL_ARTICLES} articles in ${NUM_BATCHES} batches" >> "${PROGRESS_FILE}"
echo "" >> "${PROGRESS_FILE}"

# Process batches
CURRENT_ARTICLE=${START_ARTICLE}
BATCH_NUM=1

while [ ${CURRENT_ARTICLE} -lt ${END_ARTICLE} ]; do
  # Calculate this batch's size
  REMAINING=$((END_ARTICLE - CURRENT_ARTICLE))
  THIS_BATCH_SIZE=$((REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE))

  echo "================================================================================"
  echo "BATCH ${BATCH_NUM}/${NUM_BATCHES}"
  echo "================================================================================"
  echo "  Articles: ${CURRENT_ARTICLE} to $((CURRENT_ARTICLE + THIS_BATCH_SIZE - 1)) (${THIS_BATCH_SIZE} total)"
  echo "  Started: $(date)"
  echo ""

  BATCH_LOG="${LOG_DIR}/batch_$(printf "%03d" ${BATCH_NUM})_articles_${CURRENT_ARTICLE}.log"

  # Run ingestion for this batch
  echo "  Running ingestion..."
  BATCH_START_TIME=$(date +%s)

  if LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
    --input data/datasets/wikipedia/wikipedia_500k.jsonl \
    --skip-offset ${CURRENT_ARTICLE} \
    --limit ${THIS_BATCH_SIZE} \
    > "${BATCH_LOG}" 2>&1; then

    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
    BATCH_MINUTES=$((BATCH_DURATION / 60))

    echo "  ✅ Batch ${BATCH_NUM} completed successfully (${BATCH_MINUTES} minutes)"
    echo "  Log: ${BATCH_LOG}"
    echo ""

    # Record progress
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch ${BATCH_NUM} DONE: articles ${CURRENT_ARTICLE}-$((CURRENT_ARTICLE + THIS_BATCH_SIZE - 1)) (${BATCH_MINUTES}m)" >> "${PROGRESS_FILE}"

    # Verify database state
    DB_COUNT=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
    echo "  📊 Current database: ${DB_COUNT} total concepts"

  else
    echo "  ❌ Batch ${BATCH_NUM} FAILED!"
    echo "  Log: ${BATCH_LOG}"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch ${BATCH_NUM} FAILED: articles ${CURRENT_ARTICLE}-$((CURRENT_ARTICLE + THIS_BATCH_SIZE - 1))" >> "${PROGRESS_FILE}"

    # Ask user if they want to continue or abort
    echo "================================================================================"
    echo "Batch failed. Options:"
    echo "  1) Continue to next batch (skip failed batch)"
    echo "  2) Retry this batch"
    echo "  3) Abort ingestion"
    echo ""
    read -p "Choice (1/2/3): " CHOICE

    case ${CHOICE} in
      1)
        echo "  Continuing to next batch..."
        ;;
      2)
        echo "  Retrying batch ${BATCH_NUM}..."
        continue
        ;;
      3)
        echo "  Aborting ingestion."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABORTED by user" >> "${PROGRESS_FILE}"
        exit 1
        ;;
      *)
        echo "  Invalid choice, aborting."
        exit 1
        ;;
    esac
  fi

  # Move to next batch
  CURRENT_ARTICLE=$((CURRENT_ARTICLE + THIS_BATCH_SIZE))
  BATCH_NUM=$((BATCH_NUM + 1))

  # Small delay between batches to let services stabilize
  if [ ${CURRENT_ARTICLE} -lt ${END_ARTICLE} ]; then
    echo "  ⏸  Waiting 10 seconds before next batch..."
    sleep 10
    echo ""
  fi
done

# Final summary
echo ""
echo "================================================================================"
echo "ALL BATCHES COMPLETE"
echo "================================================================================"
echo "  Completed: $(date)"
echo "  Batches processed: ${NUM_BATCHES}"
echo ""

# Final database verification
FINAL_COUNT=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | xargs)
FINAL_MAX=$(psql lnsp -t -c "SELECT MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER)) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k' AND batch_id LIKE 'wikipedia_%';" | xargs)

echo "  📊 Final database state:"
echo "     Total concepts: ${FINAL_COUNT}"
echo "     Max article: ${FINAL_MAX}"
echo ""
echo "  📁 Progress log: ${PROGRESS_FILE}"
echo "  📁 All logs: ${LOG_DIR}"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL BATCHES COMPLETE - Final: ${FINAL_COUNT} concepts, max article ${FINAL_MAX}" >> "${PROGRESS_FILE}"
echo ""
