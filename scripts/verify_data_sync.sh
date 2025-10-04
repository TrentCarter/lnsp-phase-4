#!/usr/bin/env bash
# Verify PostgreSQL, Neo4j, and FAISS are synchronized
# CRITICAL: GraphRAG will NOT work if these stores are out of sync!

set -e

echo "=" | head -c 70; echo
echo "GraphRAG Data Synchronization Verification"
echo "=" | head -c 70; echo
echo

# Get counts
echo "[1/4] Checking data store counts..."
PG_COUNT=$(psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;" 2>/dev/null || echo "0")
NEO_COUNT=$(cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)" --format plain 2>/dev/null | tail -1 || echo "0")

# Find NPZ file
NPZ_FILE=""
for candidate in artifacts/fw9k_vectors_tmd_fixed.npz artifacts/fw10k_vectors.npz; do
    if [ -f "$candidate" ]; then
        NPZ_FILE="$candidate"
        break
    fi
done

if [ -z "$NPZ_FILE" ]; then
    echo "❌ No NPZ file found!"
    NPZ_COUNT=0
else
    NPZ_COUNT=$(python3 -c "import numpy as np; print(len(np.load('$NPZ_FILE')['vectors']))" 2>/dev/null || echo "0")
fi

echo "  PostgreSQL: $PG_COUNT concepts"
echo "  Neo4j:      $NEO_COUNT concepts"
echo "  FAISS NPZ:  $NPZ_COUNT vectors ($NPZ_FILE)"
echo

# Check if counts match
if [ "$PG_COUNT" = "0" ] || [ "$NEO_COUNT" = "0" ] || [ "$NPZ_COUNT" = "0" ]; then
    echo "❌ CRITICAL: One or more data stores are empty!"
    echo "   Run: ./scripts/ingest_10k.sh"
    exit 1
fi

if [ "$PG_COUNT" != "$NEO_COUNT" ] || [ "$PG_COUNT" != "$NPZ_COUNT" ]; then
    echo "❌ CRITICAL: Data stores have different counts!"
    echo "   This means they are OUT OF SYNC!"
    echo
    echo "   PostgreSQL: $PG_COUNT"
    echo "   Neo4j:      $NEO_COUNT"
    echo "   FAISS:      $NPZ_COUNT"
    echo
    echo "   To fix: RE-INGEST everything"
    echo "   1. Clear all data: make clean-data"
    echo "   2. Re-ingest: ./scripts/ingest_10k.sh"
    echo "   3. Build FAISS: make build-faiss"
    exit 1
fi

echo "✅ Counts match: $PG_COUNT concepts in all stores"
echo

# Sample 5 concepts from PostgreSQL and verify they exist in Neo4j
echo "[2/4] Verifying sample concepts exist in all stores..."
MISMATCH=0
SAMPLE_CONCEPTS=$(psql lnsp -tAc "SELECT concept_text FROM cpe_entry ORDER BY RANDOM() LIMIT 5;" 2>/dev/null)

if [ -z "$SAMPLE_CONCEPTS" ]; then
    echo "❌ Could not retrieve sample concepts from PostgreSQL"
    exit 1
fi

while IFS= read -r concept; do
    # Check Neo4j
    NEO_CHECK=$(cypher-shell -u neo4j -p password "MATCH (c:Concept {text: \"$concept\"}) RETURN c.text LIMIT 1" --format plain 2>/dev/null | tail -1)

    # Check NPZ
    NPZ_CHECK=$(python3 -c "import numpy as np; npz=np.load('$NPZ_FILE', allow_pickle=True); print('FOUND' if '$concept' in list(npz['concept_texts']) else 'NOT_FOUND')" 2>/dev/null)

    if [ -z "$NEO_CHECK" ]; then
        echo "  ❌ '$concept' exists in PostgreSQL but NOT in Neo4j!"
        MISMATCH=1
    elif [ "$NPZ_CHECK" != "FOUND" ]; then
        echo "  ❌ '$concept' exists in PostgreSQL but NOT in FAISS NPZ!"
        MISMATCH=1
    else
        echo "  ✅ '$concept'"
    fi
done <<< "$SAMPLE_CONCEPTS"

if [ "$MISMATCH" = "1" ]; then
    echo
    echo "❌ CRITICAL: Concept mismatches detected!"
    echo "   PostgreSQL, Neo4j, and FAISS contain DIFFERENT data!"
    echo
    echo "   This likely means:"
    echo "   - Data was ingested separately to different stores"
    echo "   - Or someone ran tools/regenerate_*_vectors.py (DON'T DO THIS!)"
    echo
    echo "   To fix: RE-INGEST everything"
    echo "   ./scripts/ingest_10k.sh"
    exit 1
fi

echo "✅ Sample concepts verified in all stores"
echo

# Check Neo4j has relationships
echo "[3/4] Checking Neo4j graph connectivity..."
REL_COUNT=$(cypher-shell -u neo4j -p password "MATCH ()-[r:RELATES_TO]->() RETURN count(r)" --format plain 2>/dev/null | tail -1 || echo "0")
echo "  Relationships: $REL_COUNT edges"

if [ "$REL_COUNT" -lt "100" ]; then
    echo "  ⚠️  WARNING: Low relationship count"
    echo "     GraphRAG will have limited graph connectivity"
else
    echo "  ✅ Good graph connectivity"
fi
echo

# Check FAISS index exists
echo "[4/4] Checking FAISS index..."
if [ -f "artifacts/faiss_meta.json" ]; then
    INDEX_PATH=$(python3 -c "import json; print(json.load(open('artifacts/faiss_meta.json')).get('index_path', ''))" 2>/dev/null)
    if [ -n "$INDEX_PATH" ] && [ -f "$INDEX_PATH" ]; then
        echo "  Index: $INDEX_PATH"
        INDEX_DIM=$(python3 -c "import faiss; print(faiss.read_index('$INDEX_PATH').d)" 2>/dev/null)
        NPZ_DIM=$(python3 -c "import numpy as np; print(np.load('$NPZ_FILE')['vectors'].shape[1])" 2>/dev/null)

        echo "  Index dim: $INDEX_DIM, NPZ dim: $NPZ_DIM"

        if [ "$INDEX_DIM" != "$NPZ_DIM" ]; then
            echo "  ❌ CRITICAL: Dimension mismatch!"
            echo "     Index and NPZ have different dimensions"
            echo "     Rebuild index: make build-faiss"
            exit 1
        fi

        echo "  ✅ FAISS index matches NPZ vectors"
    else
        echo "  ❌ Index file not found: $INDEX_PATH"
        echo "     Build index: make build-faiss"
        exit 1
    fi
else
    echo "  ❌ No faiss_meta.json found"
    echo "     Build index: make build-faiss"
    exit 1
fi

echo
echo "=" | head -c 70; echo
echo "✅ ALL CHECKS PASSED - Data stores are synchronized!"
echo "=" | head -c 70; echo
echo
echo "Summary:"
echo "  - $PG_COUNT concepts in PostgreSQL"
echo "  - $NEO_COUNT concepts in Neo4j"
echo "  - $NPZ_COUNT vectors in FAISS NPZ"
echo "  - $REL_COUNT graph relationships"
echo "  - All sample concepts verified"
echo "  - FAISS index exists and matches"
echo
echo "✅ GraphRAG is ready to use!"
echo "   Run: ./scripts/run_graphrag_benchmark.sh"
