#!/bin/bash
# Validate Batch 1 Data Quality
#
# Run this after Batch 1 completes to decide if we should continue
#
# Usage:
#   ./tools/validate_batch1.sh

set -e

echo "📊 Batch 1 Data Quality Validation"
echo "===================================="
echo ""

# Check if batch 1 is complete
CHUNK_COUNT=$(psql lnsp -t -c "SELECT count(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | tr -d ' ')

if [ "$CHUNK_COUNT" -lt 100000 ]; then
    echo "⚠️  Warning: Only $CHUNK_COUNT chunks found"
    echo "   Expected: ~250,000 chunks (10,000 articles × ~25 chunks/article)"
    echo "   Batch 1 may still be running. Check logs/wikipedia_batch1_test.log"
    exit 1
fi

echo "✓ Found $CHUNK_COUNT chunks"
echo ""

# Test 1: Dataset Labeling
echo "Test 1: Dataset Labeling"
echo "-------------------------"
psql lnsp -c "
SELECT
  dataset_source,
  COUNT(*) as chunks,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cpe_entry), 2) as pct
FROM cpe_entry
GROUP BY dataset_source
ORDER BY chunks DESC;
" | head -10

CORRECT_LABEL=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';" | tr -d ' ')
if [ "$CORRECT_LABEL" -eq "$CHUNK_COUNT" ]; then
    echo "✅ PASS: All chunks correctly labeled as 'wikipedia_500k'"
else
    echo "❌ FAIL: Some chunks have incorrect dataset_source"
    exit 1
fi
echo ""

# Test 2: Chunk Size Distribution
echo "Test 2: Chunk Size Distribution"
echo "--------------------------------"
psql lnsp -c "
SELECT
  CASE
    WHEN length(concept_text) < 40 THEN '< 40 chars (too small)'
    WHEN length(concept_text) BETWEEN 40 AND 200 THEN '40-200 chars (IDEAL ✅)'
    WHEN length(concept_text) BETWEEN 201 AND 500 THEN '201-500 chars (ok)'
    ELSE '> 500 chars (TOO LARGE ❌)'
  END as size_bucket,
  COUNT(*) as chunks,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k'), 2) as pct
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
GROUP BY 1
ORDER BY
  CASE
    WHEN CASE
      WHEN length(concept_text) < 40 THEN '< 40 chars (too small)'
      WHEN length(concept_text) BETWEEN 40 AND 200 THEN '40-200 chars (IDEAL ✅)'
      WHEN length(concept_text) BETWEEN 201 AND 500 THEN '201-500 chars (ok)'
      ELSE '> 500 chars (TOO LARGE ❌)'
    END = '< 40 chars (too small)' THEN 1
    WHEN CASE
      WHEN length(concept_text) < 40 THEN '< 40 chars (too small)'
      WHEN length(concept_text) BETWEEN 40 AND 200 THEN '40-200 chars (IDEAL ✅)'
      WHEN length(concept_text) BETWEEN 201 AND 500 THEN '201-500 chars (ok)'
      ELSE '> 500 chars (TOO LARGE ❌)'
    END = '40-200 chars (IDEAL ✅)' THEN 2
    WHEN CASE
      WHEN length(concept_text) < 40 THEN '< 40 chars (too small)'
      WHEN length(concept_text) BETWEEN 40 AND 200 THEN '40-200 chars (IDEAL ✅)'
      WHEN length(concept_text) BETWEEN 201 AND 500 THEN '201-500 chars (ok)'
      ELSE '> 500 chars (TOO LARGE ❌)'
    END = '201-500 chars (ok)' THEN 3
    ELSE 4
  END;
"

# Check if ideal range is >= 75%
IDEAL_PCT=$(psql lnsp -t -c "
SELECT ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k'), 2)
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
  AND length(concept_text) BETWEEN 40 AND 200;
" | tr -d ' ')

if (( $(echo "$IDEAL_PCT >= 75" | bc -l) )); then
    echo "✅ PASS: ${IDEAL_PCT}% of chunks in ideal range (40-200 chars)"
else
    echo "❌ FAIL: Only ${IDEAL_PCT}% in ideal range (need >= 75%)"
    exit 1
fi

# Check if too-large is <= 5%
TOO_LARGE_PCT=$(psql lnsp -t -c "
SELECT ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k'), 2)
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
  AND length(concept_text) > 500;
" | tr -d ' ')

if (( $(echo "$TOO_LARGE_PCT <= 5" | bc -l) )); then
    echo "✅ PASS: Only ${TOO_LARGE_PCT}% chunks too large (> 500 chars)"
else
    echo "❌ FAIL: ${TOO_LARGE_PCT}% chunks too large (threshold: 5%)"
    exit 1
fi

echo ""

# Test 3: Sample Quality
echo "Test 3: Sample Quality (10 random chunks)"
echo "------------------------------------------"
psql lnsp -c "
SELECT
  substring(concept_text, 1, 80) as sample,
  length(concept_text) as len,
  domain_code,
  task_code,
  modifier_code
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
ORDER BY random()
LIMIT 10;
"

echo ""
echo "===================================="
echo "✅ ALL TESTS PASSED!"
echo "===================================="
echo ""
echo "Summary:"
echo "  Chunks: $CHUNK_COUNT"
echo "  Ideal range (40-200): ${IDEAL_PCT}%"
echo "  Too large (>500): ${TOO_LARGE_PCT}%"
echo ""
echo "Next steps:"
echo "  1. Review the sample quality above"
echo "  2. If satisfied, continue to Batch 2-50:"
echo "     ./tools/continue_wikipedia_from_batch2.sh"
echo ""
