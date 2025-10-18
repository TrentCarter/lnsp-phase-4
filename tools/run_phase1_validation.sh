#!/bin/bash
# Phase 1: Data Quality Validation
# Run this IMMEDIATELY after Batch 1 completes

set -e

echo "========================================="
echo "Phase 1: Data Quality Validation"
echo "========================================="
echo ""

# Create validation directories
mkdir -p artifacts/validation
mkdir -p artifacts/reports

echo "📊 Test 1: Sequential Coherence"
echo "-------------------------------"
PYTHONPATH=. ./.venv/bin/python tools/test_sequential_coherence.py \
  --dataset wikipedia_500k \
  --limit 1000 \
  --output artifacts/validation/coherence_batch1.json

COHERENCE_STATUS=$?
echo ""

echo "📊 Test 2: TMD Distribution"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/tmd_distribution_batch1.txt
SELECT 
  domain_code,
  count(*) as chunks,
  round(100.0 * count(*) / sum(count(*)) OVER (), 2) as pct
FROM cpe_entry 
WHERE dataset_source = 'wikipedia_500k'
GROUP BY domain_code 
ORDER BY count DESC;
\o
SQL

cat artifacts/validation/tmd_distribution_batch1.txt
echo ""

echo "📊 Test 3: Chunk Statistics"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/chunk_stats_batch1.txt
SELECT
  count(*) as total_chunks,
  round(avg(length(concept_text))) as avg_length,
  min(length(concept_text)) as min_length,
  max(length(concept_text)) as max_length,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY length(concept_text)) as median_length
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k';
\o
SQL

cat artifacts/validation/chunk_stats_batch1.txt
echo ""

echo "📊 Test 4: Sample Chunks (Quality Check)"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/sample_chunks_batch1.txt
SELECT 
  concept_text,
  length(concept_text) as len,
  domain_code
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
ORDER BY random()
LIMIT 10;
\o
SQL

cat artifacts/validation/sample_chunks_batch1.txt
echo ""

echo "========================================="
echo "Phase 1 Validation Complete!"
echo "========================================="
echo ""
echo "📊 Results saved to: artifacts/validation/"
echo ""

if [ $COHERENCE_STATUS -eq 0 ]; then
  echo "✅ STATUS: PASS"
  echo "   All tests passed. Safe to continue ingestion."
  echo ""
  echo "📝 Next Steps:"
  echo "   1. Review results in artifacts/validation/"
  echo "   2. Continue ingestion to Batch 5-10"
  echo "   3. Prepare for Phase 2 training (see docs/Iterative_Training_Strategy.md)"
  exit 0
elif [ $COHERENCE_STATUS -eq 2 ]; then
  echo "⚠️  STATUS: WARN"
  echo "   Tests passed with warnings. Review results carefully."
  echo ""
  echo "📝 Next Steps:"
  echo "   1. Review artifacts/validation/coherence_batch1.json"
  echo "   2. If concerns persist, consider adjusting chunking parameters"
  echo "   3. Safe to continue ingestion, but monitor quality"
  exit 0
else
  echo "❌ STATUS: FAIL"
  echo "   Critical issues detected. DO NOT CONTINUE ingestion!"
  echo ""
  echo "📝 Next Steps:"
  echo "   1. Review artifacts/validation/coherence_batch1.json"
  echo "   2. Investigate root cause of low coherence"
  echo "   3. Fix chunking strategy before proceeding"
  exit 1
fi
