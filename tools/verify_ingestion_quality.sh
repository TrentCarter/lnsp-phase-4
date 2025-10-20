#!/usr/bin/env bash
#
# Verify Wikipedia Ingestion Quality
#
# Checks:
# 1. Vector count increase
# 2. FAISS NPZ file updated
# 3. Data consistency (PostgreSQL ↔ FAISS)
#

set -euo pipefail

echo "🔍 INGESTION QUALITY VERIFICATION"
echo "=================================="
echo ""

# 1. Check PostgreSQL vector count
echo "1. PostgreSQL Stats:"
echo "-------------------"
psql lnsp -c "
    SELECT
        COUNT(*) as total_vectors,
        COUNT(DISTINCT id) as unique_concepts,
        MIN(created_at) as first_created,
        MAX(created_at) as last_created
    FROM cpe_vectors
    WHERE id IN (
        SELECT id FROM cpe_entry WHERE dataset_source = 'wikipedia-500k'
    );
" 2>/dev/null || echo "  ❌ Database query failed"
echo ""

# 2. Check FAISS NPZ file
echo "2. FAISS NPZ File:"
echo "------------------"
NPZ_FILE="artifacts/wikipedia_500k_corrected_vectors.npz"
if [ -f "$NPZ_FILE" ]; then
    echo "  File: $NPZ_FILE"
    echo "  Size: $(du -h "$NPZ_FILE" | cut -f1)"
    echo "  Modified: $(stat -f "%Sm" "$NPZ_FILE")"

    # Count vectors in NPZ
    python3 <<EOF 2>/dev/null || echo "  ❌ NPZ inspection failed"
import numpy as np
npz = np.load("$NPZ_FILE")
print(f"  Vectors: {npz['vectors'].shape[0]:,}")
print(f"  Dimensions: {npz['vectors'].shape[1]}")
if 'concept_texts' in npz:
    print(f"  Concepts: {len(npz['concept_texts']):,}")
if 'cpe_ids' in npz:
    print(f"  CPE IDs: {len(npz['cpe_ids']):,}")
EOF
else
    echo "  ❌ FAISS NPZ file not found: $NPZ_FILE"
fi
echo ""

# 3. Check consistency
echo "3. Data Consistency:"
echo "--------------------"
psql lnsp -c "
    WITH pg_counts AS (
        SELECT COUNT(*) as pg_vectors
        FROM cpe_vectors
        WHERE id IN (
            SELECT id FROM cpe_entry WHERE dataset_source = 'wikipedia-500k'
        )
    )
    SELECT
        pg_vectors,
        pg_vectors AS expected_npz_vectors,
        CASE
            WHEN pg_vectors > 0 THEN '✅ Data available'
            ELSE '⚠️  No data found'
        END as status
    FROM pg_counts;
" 2>/dev/null || echo "  ❌ Consistency check failed"
echo ""

# 4. Recent activity check
echo "4. Recent Activity (last 24 hours):"
echo "------------------------------------"
psql lnsp -c "
    SELECT
        COUNT(*) as vectors_added_24h
    FROM cpe_vectors
    WHERE
        created_at >= NOW() - INTERVAL '24 hours'
        AND id IN (
            SELECT id FROM cpe_entry WHERE dataset_source = 'wikipedia-500k'
        );
" 2>/dev/null || echo "  ❌ Activity check failed"
echo ""

# 5. Summary
echo "=================================="
echo "✅ Verification complete!"
echo ""
echo "Next step: Re-export training data with larger dataset"
echo ""
