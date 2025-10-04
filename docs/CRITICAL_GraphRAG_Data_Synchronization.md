# 🚨 CRITICAL: GraphRAG Data Synchronization

**Date**: October 3, 2025
**Priority**: CRITICAL - System Integrity
**Impact**: GraphRAG completely broken if violated

---

## The Golden Rule

> **NEVER update PostgreSQL, Neo4j, or FAISS independently.**
> **ALWAYS ingest data to all three stores atomically in a single run.**

---

## Why This Matters

GraphRAG depends on **three data stores being EXACTLY synchronized**:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │     │     Neo4j       │     │     FAISS       │
│                 │     │                 │     │                 │
│  concept_text   │────▶│  Concept.text   │────▶│  vector[i]      │
│  cpe_id         │     │  cpe_id         │     │  position=i     │
│  TMD codes      │     │  relationships  │     │  embedding      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       ▲                       ▲
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                    MUST ALL MATCH EXACTLY!
```

**If any ONE gets out of sync**:
- vecRAG returns concept `i`
- GraphRAG looks up `concept[i]` in Neo4j
- Neo4j returns neighbors for **different concept** from previous run
- Result: **0% accuracy, total failure**

---

## What Happened (Oct 2, 2025 Incident)

### The Mistake

1. **Oct 1**: Ran `./scripts/ingest_10k.sh` → Wrote FactoidWiki to PostgreSQL + Neo4j + FAISS
2. **Oct 2**: Ran `./tools/regenerate_all_tmd_vectors.py` → **ONLY updated PostgreSQL**
3. **Result**:
   - PostgreSQL: "oxidoreductase activity"
   - Neo4j: "Moroccan royal family" (STALE!)
   - FAISS: vectors for "Moroccan royal family" (STALE!)
4. **GraphRAG**: 0 neighbors found, 0% improvement

### The Damage

```bash
# What the test showed:
GraphRAG test: 10 queries
- vecRAG:    5/10 = 50% P@1
- GraphRAG:  5/10 = 50% P@1  ← NO IMPROVEMENT!
- Reason: All queries had 0 graph neighbors (concepts didn't exist in Neo4j)
```

---

## The ONLY Correct Ingestion Process

### Step 1: Clear Everything

```bash
# Clear PostgreSQL
psql lnsp -c "TRUNCATE cpe_entry, cpe_vectors CASCADE;"

# Clear Neo4j
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"

# Clear FAISS artifacts
rm -f artifacts/*.index artifacts/*_vectors*.npz
```

### Step 2: Atomic Ingestion (writes to ALL three stores)

```bash
# Run ingestion with ALL flags enabled
./scripts/ingest_10k.sh

# This internally calls:
python -m src.ingest_factoid \
  --file-path artifacts/fw10k_chunks.jsonl \
  --num-samples 10000 \
  --write-pg \           # ← PostgreSQL
  --write-neo4j \        # ← Neo4j
  --faiss-out artifacts/fw10k_vectors.npz  # ← FAISS NPZ
```

### Step 3: Build FAISS Index

```bash
# Build searchable index from NPZ vectors
make build-faiss ARGS="--type ivf_flat --metric ip"

# Or manually:
FAISS_NPZ_PATH=artifacts/fw10k_vectors.npz \
  python -m src.faiss_index \
  --type ivf_flat \
  --metric ip \
  --nlist 512 \
  --nprobe 16
```

### Step 4: Verify Synchronization

```bash
# Check counts match
echo "PostgreSQL:" && psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;"
echo "Neo4j:" && cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)" --format plain | tail -1
echo "FAISS NPZ:" && python -c "import numpy as np; print(len(np.load('artifacts/fw10k_vectors.npz')['vectors']))"

# Check sample concept exists in all three
SAMPLE="oxidoreductase activity"
psql lnsp -c "SELECT concept_text FROM cpe_entry WHERE concept_text='$SAMPLE' LIMIT 1;"
cypher-shell -u neo4j -p password "MATCH (c:Concept {text: \"$SAMPLE\"}) RETURN c.text LIMIT 1"
python -c "import numpy as np; npz=np.load('artifacts/fw10k_vectors.npz', allow_pickle=True); print('$SAMPLE' in npz['concept_texts'])"
```

---

## Forbidden Operations

### ❌ NEVER Do This

```bash
# DON'T update vectors separately!
python tools/regenerate_all_tmd_vectors.py  # ❌ WRONG - only updates PostgreSQL

# DON'T vectorize from PostgreSQL without updating Neo4j!
python tools/fix_ontology_tmd_real.py      # ❌ WRONG - creates NPZ without Neo4j sync

# DON'T manually update Neo4j without PostgreSQL!
cypher-shell "CREATE (c:Concept {text: ...})"  # ❌ WRONG - PostgreSQL won't match
```

### ✅ ALWAYS Do This

```bash
# RE-INGEST everything atomically
./scripts/ingest_10k.sh  # ✓ CORRECT - writes to all three stores at once
```

---

## How to Detect Synchronization Issues

### Quick Check Script

```bash
#!/usr/bin/env bash
# Save as: scripts/verify_data_sync.sh

# Get concept counts
PG_COUNT=$(psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")
NEO_COUNT=$(cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)" --format plain 2>/dev/null | tail -1)
NPZ_COUNT=$(python -c "import numpy as np; print(len(np.load('artifacts/fw10k_vectors.npz')['vectors']))" 2>/dev/null || echo 0)

echo "Data store counts:"
echo "  PostgreSQL: $PG_COUNT"
echo "  Neo4j:      $NEO_COUNT"
echo "  FAISS NPZ:  $NPZ_COUNT"

if [ "$PG_COUNT" = "$NEO_COUNT" ] && [ "$PG_COUNT" = "$NPZ_COUNT" ]; then
    echo "✅ Counts match!"
else
    echo "❌ MISMATCH DETECTED! Data stores are out of sync!"
    exit 1
fi

# Sample 5 random concepts from PostgreSQL and check they exist in Neo4j
psql lnsp -tAc "SELECT concept_text FROM cpe_entry ORDER BY RANDOM() LIMIT 5;" | while read -r concept; do
    NEO_CHECK=$(cypher-shell -u neo4j -p password "MATCH (c:Concept {text: \"$concept\"}) RETURN c.text LIMIT 1" --format plain 2>/dev/null | tail -1)
    if [ -z "$NEO_CHECK" ]; then
        echo "❌ Concept '$concept' exists in PostgreSQL but NOT in Neo4j!"
        exit 1
    fi
done

echo "✅ Sample concepts verified in both PostgreSQL and Neo4j"
```

---

## Updated Workflow for GraphRAG

### Before Running GraphRAG Benchmark

```bash
# 1. Verify data is synchronized
./scripts/verify_data_sync.sh

# 2. If out of sync, re-ingest
if [ $? -ne 0 ]; then
    echo "Re-ingesting data..."
    ./scripts/ingest_10k.sh
    make build-faiss
fi

# 3. Run GraphRAG benchmark
./scripts/run_graphrag_benchmark.sh
```

### After Any Data Changes

```bash
# If you modify data in ANY way:
# 1. Clear all stores
# 2. Re-ingest from source
# 3. Rebuild FAISS index
# 4. Verify sync

# NO SHORTCUTS - it's faster to re-ingest than debug sync issues!
```

---

## Code Changes Made (Oct 3, 2025)

### 1. Updated `scripts/ingest_10k.sh`
- ✅ Added warnings about atomic writes
- ✅ Added automatic sync verification after ingestion
- ✅ Checks sample concept exists in PostgreSQL + Neo4j + FAISS

### 2. Created `tools/README_VECTOR_REGENERATION_WARNING.md`
- ✅ Documents why regeneration scripts are dangerous
- ✅ Explains the Oct 2 incident
- ✅ Provides correct re-ingestion procedure

### 3. Updated `RAG/bench.py`
- ✅ Fixed NPZ detection to verify 2D vectors exist
- ✅ Prioritizes `fw9k_vectors_tmd_fixed.npz` (correct file)

### 4. Created this document
- ✅ Establishes synchronization as CRITICAL requirement
- ✅ Documents correct procedures
- ✅ Warns against forbidden operations

---

## Testing Procedure After Re-Ingestion

```bash
# 1. Re-ingest with sync verification
./scripts/ingest_10k.sh

# 2. Build FAISS index
make build-faiss ARGS="--type ivf_flat --metric ip"

# 3. Run quick GraphRAG test (10 queries)
python graphrag_quick_test.py

# Expected results:
# - All queries should find 1-10 graph neighbors
# - GraphRAG should show +3-10% P@1 improvement over vecRAG
# - No "0 neighbors" warnings
```

---

## Action Items

- [x] Document root cause of Oct 2 incident
- [x] Update `ingest_10k.sh` with sync verification
- [x] Create warning docs for regeneration scripts
- [x] Fix NPZ detection in bench.py
- [ ] **DEPRECATE** `tools/regenerate_*.py` scripts (move to `_deprecated/`)
- [ ] **RE-INGEST** FactoidWiki data with proper synchronization
- [ ] **RE-RUN** GraphRAG benchmark to get real results
- [ ] **ADD** CI check to prevent future desync

---

## Summary

**The Problem**: Someone ran a vector regeneration script that only updated PostgreSQL, breaking synchronization with Neo4j and FAISS.

**The Solution**: **NEVER update stores independently**. Always use `./scripts/ingest_10k.sh` which writes to all three stores atomically.

**The Rule**: If you need to change vectors, **RE-INGEST EVERYTHING**. There are no shortcuts.

**The Verification**: Run `./scripts/verify_data_sync.sh` after any data operation.

---

**Remember**: GraphRAG is only as good as its data synchronization. Treat this as a **database integrity constraint** that must NEVER be violated.
