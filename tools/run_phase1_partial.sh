#!/bin/bash
# Phase 1: Partial Validation (works without article tracking)

set -e

echo "==========================================="
echo "Phase 1: Partial Data Quality Validation"
echo "==========================================="
echo ""
echo "⚠️  Note: Running partial validation due to dataset labeling issue"
echo "   Article-level tests skipped (sequential coherence)"
echo "   Chunk-level tests running (TMD, size, samples)"
echo ""

mkdir -p artifacts/validation

echo "📊 Test 1: Chunk Statistics"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/chunk_stats_today.txt
SELECT
  count(*) as total_chunks,
  round(avg(length(concept_text))) as avg_length,
  min(length(concept_text)) as min_length,
  max(length(concept_text)) as max_length,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY length(concept_text)) as p25_length,
  percentile_cont(0.50) WITHIN GROUP (ORDER BY length(concept_text)) as median_length,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY length(concept_text)) as p75_length
FROM cpe_entry
WHERE created_at > '2025-10-14';
\o
SQL

cat artifacts/validation/chunk_stats_today.txt
echo ""

echo "📊 Test 2: TMD Distribution"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/tmd_distribution_today.txt
SELECT 
  domain_code,
  count(*) as chunks,
  round(100.0 * count(*) / sum(count(*)) OVER (), 2) as pct
FROM cpe_entry 
WHERE created_at > '2025-10-14'
GROUP BY domain_code 
ORDER BY count DESC
LIMIT 15;
\o
SQL

cat artifacts/validation/tmd_distribution_today.txt
echo ""

echo "📊 Test 3: Sample Quality (Random 20 chunks)"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/samples_today.txt
SELECT 
  left(concept_text, 100) as chunk_preview,
  length(concept_text) as len,
  domain_code,
  task_code,
  modifier_code
FROM cpe_entry
WHERE created_at > '2025-10-14'
AND length(concept_text) > 50
ORDER BY random()
LIMIT 20;
\o
SQL

cat artifacts/validation/samples_today.txt
echo ""

echo "📊 Test 4: Chunk Length Distribution"
echo "-------------------------------"
psql lnsp << SQL
\o artifacts/validation/length_distribution_today.txt
SELECT 
  CASE 
    WHEN length(concept_text) < 50 THEN '< 50 chars (too short)'
    WHEN length(concept_text) < 100 THEN '50-100 chars'
    WHEN length(concept_text) < 200 THEN '100-200 chars (ideal)'
    WHEN length(concept_text) < 300 THEN '200-300 chars (good)'
    WHEN length(concept_text) < 500 THEN '300-500 chars'
    ELSE '> 500 chars (too long)'
  END as length_bucket,
  count(*) as chunks,
  round(100.0 * count(*) / sum(count(*)) OVER (), 2) as pct
FROM cpe_entry
WHERE created_at > '2025-10-14'
GROUP BY length_bucket
ORDER BY min(length(concept_text));
\o
SQL

cat artifacts/validation/length_distribution_today.txt
echo ""

echo "==========================================="
echo "Partial Validation Complete"
echo "==========================================="
echo ""
echo "✅ RESULTS:"
cat artifacts/validation/chunk_stats_today.txt | grep -E "total|avg|median"
echo ""
echo "⚠️  LIMITATIONS:"
echo "   - Cannot test sequential coherence (no article tracking)"
echo "   - Cannot validate article boundaries"
echo "   - Cannot prepare training sequences"
echo ""
echo "📝 NEXT STEPS:"
echo "   1. Fix dataset_source labeling in ingestion pipeline"
echo "   2. Add article_id and chunk_index to chunk_position"
echo "   3. Re-run full Phase 1 validation"
echo ""
echo "📊 Results saved to: artifacts/validation/"

exit 0
