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

# Resolve NPZ and Index from configs/lightrag.yml if present
NPZ_FILE=""
INDEX_PATH=""

if [ -f "configs/lightrag.yml" ]; then
    NPZ_FROM_CFG=$(python3 -c "import yaml,sys; d=yaml.safe_load(open('configs/lightrag.yml')); print((d or {}).get('vector_store',{}).get('meta_npz',''))" 2>/dev/null || echo "")
    IDX_FROM_CFG=$(python3 -c "import yaml,sys; d=yaml.safe_load(open('configs/lightrag.yml')); print((d or {}).get('vector_store',{}).get('index_path',''))" 2>/dev/null || echo "")
    if [ -n "$NPZ_FROM_CFG" ] && [ -f "$NPZ_FROM_CFG" ]; then NPZ_FILE="$NPZ_FROM_CFG"; fi
    if [ -n "$IDX_FROM_CFG" ] && [ -f "$IDX_FROM_CFG" ]; then INDEX_PATH="$IDX_FROM_CFG"; fi
fi

# Fallback NPZ candidates
if [ -z "$NPZ_FILE" ]; then
    for candidate in artifacts/ontology_4k_full.npz artifacts/ontology_4k_tmd_llm.npz artifacts/fw10k_vectors.npz artifacts/fw9k_vectors_tmd_fixed.npz; do
        if [ -f "$candidate" ]; then NPZ_FILE="$candidate"; break; fi
    done
fi

# Fallback Index candidates via index_meta.json
if [ -z "$INDEX_PATH" ] && [ -f "artifacts/index_meta.json" ]; then
    INDEX_PATH=$(python3 -c "import json; m=json.load(open('artifacts/index_meta.json')); print(sorted(m.keys())[-1] if m else '')" 2>/dev/null || echo "")
fi

# Final NPZ count
if [ -z "$NPZ_FILE" ]; then
    echo "❌ No NPZ file found!"
    NPZ_COUNT=0
else
    NPZ_COUNT=$(python3 -c "import numpy as np; npz=np.load('$NPZ_FILE'); print(len(npz['vectors']) if 'vectors' in npz else (len(npz['fused']) if 'fused' in npz else 0))" 2>/dev/null || echo "0")
fi

echo "  PostgreSQL: $PG_COUNT concepts"
echo "  Neo4j:      $NEO_COUNT concepts"
echo "  FAISS NPZ:  $NPZ_COUNT vectors ($NPZ_FILE)"
echo

# Check if counts match
if [ "$PG_COUNT" = "0" ] || [ "$NEO_COUNT" = "0" ] || [ "$NPZ_COUNT" = "0" ]; then
    echo "❌ CRITICAL: One or more data stores are empty!"
    echo "   Run: ./scripts/ingest_ontologies.sh && make build-faiss"
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
    echo "   2. Re-ingest: ./scripts/ingest_ontologies.sh"
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
    echo "   ./scripts/ingest_ontologies.sh && make build-faiss"
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
else
    echo "  ✅ Good graph connectivity"
fi
echo

# Check FAISS index exists (prefer resolved INDEX_PATH)
echo "[4/4] Checking FAISS index..."
if [ -n "$INDEX_PATH" ] && [ -f "$INDEX_PATH" ]; then
    echo "  Index: $INDEX_PATH"
    INDEX_DIM=$(python3 -c "import faiss; print(faiss.read_index('$INDEX_PATH').d)" 2>/dev/null)
    NPZ_DIM=$(python3 -c "import numpy as np; npz=np.load('$NPZ_FILE'); print(npz['vectors'].shape[1] if 'vectors' in npz else (npz['fused'].shape[1] if 'fused' in npz else -1))" 2>/dev/null)

    echo "  Index dim: $INDEX_DIM, NPZ dim: $NPZ_DIM"

    if [ "$INDEX_DIM" != "$NPZ_DIM" ]; then
        echo "  ❌ CRITICAL: Dimension mismatch!"
        echo "     Index and NPZ have different dimensions"
        echo "     Rebuild index: make build-faiss"
        exit 1
    fi

    echo "  ✅ FAISS index matches NPZ vectors"
else
    # Legacy path: try to read meta file for index
    if [ -f "artifacts/index_meta.json" ]; then
        echo "  ❌ Index path from config missing or invalid"
        echo "     Check artifacts/index_meta.json and configs/lightrag.yml"
        exit 1
    else
        echo "  ❌ No index path resolved"
        echo "     Build index: make build-faiss"
        exit 1
    fi
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
