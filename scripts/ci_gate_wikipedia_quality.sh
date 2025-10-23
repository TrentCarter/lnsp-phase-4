#!/bin/bash
# CI Gate: Wikipedia Data Quality Checks
# Enforces production-ready data quality standards
# Exit codes: 0 = pass, 1 = fail

set -e

DATASET_SOURCE="${1:-wikipedia_500k}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Wikipedia Data Quality CI Gate ==="
echo "Dataset: $DATASET_SOURCE"
echo ""

FAILED=0

# Check 1: Giant chunks (>500 chars)
echo -n "Check 1: Giant chunks (>500 chars)... "
GIANTS=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM cpe_entry
WHERE dataset_source='$DATASET_SOURCE'
  AND LENGTH(concept_text) > 500;
")
if [ "$GIANTS" -gt 0 ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  Found $GIANTS giant chunks (>500 chars)"
    echo "  These are moved to catalog lane (not used for training)"
else
    echo -e "${GREEN}PASS${NC}"
fi

# Check 2: Microscopic chunks (<8 chars)
echo -n "Check 2: Microscopic chunks (<8 chars)... "
MICROSCOPIC=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM cpe_entry
WHERE dataset_source='$DATASET_SOURCE'
  AND LENGTH(concept_text) < 8;
")
if [ "$MICROSCOPIC" -gt 0 ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  Found $MICROSCOPIC microscopic chunks (<8 chars)"
    echo "  These are moved to catalog lane (not used for training)"
else
    echo -e "${GREEN}PASS${NC}"
fi

# Check 3: Duplicate keys (article_index, chunk_index)
echo -n "Check 3: Duplicate (article_index, chunk_index) keys... "
DUPLICATE_KEYS=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM (
  SELECT
    chunk_position->>'article_index' AS article_idx,
    chunk_position->>'chunk_index' AS chunk_idx,
    COUNT(*) AS n
  FROM cpe_entry
  WHERE dataset_source='$DATASET_SOURCE'
  GROUP BY 1, 2
  HAVING COUNT(*) > 1
) dups;
")
if [ "$DUPLICATE_KEYS" -gt 0 ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  Found $DUPLICATE_KEYS duplicate key pairs"
    echo "  This breaks data integrity - run flatten operation"
    FAILED=1
else
    echo -e "${GREEN}PASS${NC}"
fi

# Check 4: Duplicate text within same article
echo -n "Check 4: Duplicate text within articles... "
DUPLICATE_TEXT=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM (
  SELECT
    chunk_position->>'article_index' AS article_idx,
    concept_text,
    COUNT(*) AS n
  FROM cpe_entry
  WHERE dataset_source='$DATASET_SOURCE'
  GROUP BY 1, 2
  HAVING COUNT(*) > 1
) dups;
")
if [ "$DUPLICATE_TEXT" -gt 0 ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  Found $DUPLICATE_TEXT duplicate text instances within articles"
    echo "  May indicate redundant chunks (acceptable in small quantities)"
else
    echo -e "${GREEN}PASS${NC}"
fi

# Check 5: Train lane size
echo -n "Check 5: Train lane size (8-500 chars)... "
TRAIN_COUNT=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM cpe_entry
WHERE dataset_source='$DATASET_SOURCE'
  AND LENGTH(concept_text) BETWEEN 8 AND 500;
")
TOTAL_COUNT=$(psql lnsp -tAc "
SELECT COUNT(*)
FROM cpe_entry
WHERE dataset_source='$DATASET_SOURCE';
")
TRAIN_PCT=$(echo "scale=2; 100.0 * $TRAIN_COUNT / $TOTAL_COUNT" | bc)
echo -e "${GREEN}PASS${NC}"
echo "  Train lane: $TRAIN_COUNT / $TOTAL_COUNT chunks ($TRAIN_PCT%)"

# Summary
echo ""
echo "=== Summary ==="
echo "Total chunks: $TOTAL_COUNT"
echo "Train lane (8-500 chars): $TRAIN_COUNT ($TRAIN_PCT%)"
echo "Catalog lane: $((TOTAL_COUNT - TRAIN_COUNT)) ($((100 - ${TRAIN_PCT%.*}))%)"
echo "  - Giants (>500 chars): $GIANTS"
echo "  - Microscopic (<8 chars): $MICROSCOPIC"
echo ""

if [ $FAILED -eq 1 ]; then
    echo -e "${RED}CI GATE: FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}CI GATE: PASSED${NC}"
    exit 0
fi
