# Wikipedia Ingestion Pipeline - Detailed Performance Analysis

**Date**: October 15, 2025
**Test Dataset**: 10 Wikipedia articles (validated)
**Configuration**: CPESH **DISABLED**, TMD hybrid mode, Batch processing

---

## Executive Summary

### Current Performance (CPESH Disabled)

| Metric | Value |
|--------|-------|
| **Time per chunk/vector** | **244ms** |
| **Time per article** | 34.9 seconds |
| **Throughput** | 0.029 articles/sec (2.5 articles/min) |
| **Chunks per article** | 143 chunks (avg) |
| **Episodes per article** | 29.2 episodes (avg) |

### Projected Time for 500k Articles (Current Config)

| Scale | Time Estimate | Notes |
|-------|---------------|-------|
| **500k articles** | **201.4 days** | At 34.9s per article (CPESH disabled) |
| **100k articles** | **40.3 days** | Proportional scaling |
| **10k articles** | **4.0 days** | Recommended batch size |

---

## ⚠️ CRITICAL: CPESH Status

### Current State: **CPESH EXTRACTION IS DISABLED**

The Ingest API (`app/api/ingest_chunks.py:124`) checks environment variable:

```python
self.enable_cpesh = os.getenv("LNSP_ENABLE_CPESH", "false").lower() == "true"
```

**Current runtime**: Ingest API started **WITHOUT** `LNSP_ENABLE_CPESH=true`

### What This Means:

**CPESH Disabled (Current)**:
- ✅ Fast ingestion: 244ms per chunk
- ❌ No concept extraction: Uses raw chunk text as "concept"
- ❌ No probe questions generated
- ❌ No expected answers generated
- ❌ No soft/hard negatives for contrastive learning
- ❌ **NOT suitable for LVM training** (missing contrastive pairs)

**CPESH Enabled (Required for Training)**:
- ❌ Slower ingestion: ~700-2,500ms per chunk (estimated)
- ✅ Real concept extraction via TinyLlama LLM
- ✅ Probe questions + expected answers
- ✅ Soft negatives (3-5 similar concepts)
- ✅ Hard negatives (3-5 confusable concepts)
- ✅ **Production-ready data for LVM training**

---

## Detailed Time Breakdown (Per Chunk/Vector)

Based on 10-article test (1,430 chunks processed, 0 errors):

### Stage-by-Stage Timing

| Stage | Total Time | Avg per Article | **Avg per Chunk** | % of Total |
|-------|------------|-----------------|-------------------|------------|
| **1. Episode Chunking** | 22,152ms | 2,215ms | **15.5ms** | 6.4% |
| **2. Semantic Chunking** | 14,040ms | 1,404ms | **9.8ms** | 4.0% |
| **3. GTR-T5 Embeddings** | 27,785ms | 2,779ms | **19.4ms** | 8.0% |
| **4. Ingestion (TMD+DB)** | 284,564ms | 28,456ms | **199.0ms** | 81.6% |
| **Total Pipeline** | 348,542ms | 34,854ms | **243.7ms** | 100% |

### What Each Stage Does:

**1. Episode Chunking (15.5ms per chunk)**
- Coherence-based document segmentation
- Identifies natural episode boundaries
- Creates ~29 episodes per article
- Service: `http://localhost:8900`

**2. Semantic Chunking (9.8ms per chunk)**
- Fine-grained semantic boundary detection
- Splits episodes into chunks (min=10, max=500 chars)
- Creates ~143 chunks per article (~5 chunks per episode)
- Service: `http://localhost:8001`

**3. GTR-T5 Embeddings (19.4ms per chunk)**
- Vec2text-compatible encoder
- Generates 768D semantic vectors
- Uses in-process GTR-T5 model (no API calls)
- Service: `http://localhost:8767`

**4. Ingestion (199.0ms per chunk) - THE BOTTLENECK**
- **TMD Extraction** (hybrid mode):
  - Domain: Heuristic first, LLM fallback (~5-20ms)
  - Task: Heuristics only (~1ms)
  - Modifier: Heuristics only (~1ms)
- **TMD Encoding** (16D dense vector): ~0.1ms
- **Vector Fusion** (768D + 16D = 784D): ~0.1ms
- **PostgreSQL Writes** (~150-180ms):
  - Insert into `cpe_entry` table
  - Insert into `cpe_vectors` table
  - Transactional (ACID guarantees)
- **NO CPESH extraction** (disabled, would add 500-2000ms)

---

## Performance Analysis

### Bottleneck Identification

**Primary Bottleneck: Ingestion Stage (81.6% of time)**

Breaking down the 199ms ingestion time:
- PostgreSQL writes: **~150-180ms** (75-90% of ingestion time)
- TMD extraction: **~10-20ms** (5-10% of ingestion time)
- Vector operations: **~1-5ms** (negligible)

**Secondary Bottleneck: GTR-T5 Embeddings (8.0% of time)**
- Currently using in-process model
- Could benefit from batch optimizations

### Optimization Opportunities

#### 1. Database Write Optimization (Highest Impact)

**Current**: Sequential PostgreSQL writes per chunk

**Optimization**: Batch writes with transaction pooling

```python
# Current (slow):
for chunk in chunks:
    insert_cpe_entry(conn, chunk)      # Individual INSERT
    upsert_cpe_vectors(conn, chunk)    # Individual UPSERT

# Optimized (fast):
batch_insert_cpe_entries(conn, chunks)      # Single transaction with COPY
batch_upsert_cpe_vectors(conn, vectors)     # Batched UPSERT
```

**Expected Speedup**: 3-5x (PostgreSQL writes: 150ms → 30-50ms per chunk)

**Implementation**: Already supported in Ingest API via `LNSP_ENABLE_BATCH_EMBEDDINGS=true`

**Estimated new time per chunk**: 244ms → 110-130ms (55% improvement)

---

#### 2. Enable Batch Embeddings (Medium Impact)

**Current**: Individual embedding calls per chunk

**Optimization**: Batch embedding with GPU parallelization

```bash
# Enable batch mode
export LNSP_ENABLE_BATCH_EMBEDDINGS=true
export LNSP_MAX_PARALLEL_WORKERS=10
```

**Expected Speedup**: 1.5-2x (embeddings: 19.4ms → 10-13ms per chunk)

**Estimated new time per chunk**: 244ms → 225-235ms (4% improvement)

---

#### 3. CPESH Impact Analysis (When Enabled)

**CPESH extraction time estimate** (based on TinyLlama 1.1B):
- Fast case (simple concepts): ~500ms per chunk
- Average case: ~1,000ms per chunk
- Complex case (technical concepts): ~2,000ms per chunk

**Projected time per chunk WITH CPESH enabled**:
- Best case: 244ms + 500ms = **744ms per chunk**
- Average case: 244ms + 1,000ms = **1,244ms per chunk** ← Most likely
- Worst case: 244ms + 2,000ms = **2,244ms per chunk**

**This means**:
- Current (no CPESH): 244ms per chunk = **201 days for 500k articles**
- With CPESH: ~1,244ms per chunk = **1,027 days for 500k articles** (2.8 years!)

**Solution**: Must enable CPESH + optimize database writes simultaneously

---

## Optimized Pipeline Projections

### Configuration 1: Current (CPESH Disabled, No Batch)
```bash
# Current running config
LNSP_ENABLE_CPESH=false
LNSP_ENABLE_BATCH_EMBEDDINGS=false
LNSP_TMD_MODE=hybrid
```

| Metric | Value |
|--------|-------|
| Time per chunk | 244ms |
| Time per article | 34.9s |
| 500k articles | **201 days** |
| 10k article batch | **4.0 days** |

**Status**: ❌ Not production-ready (missing CPESH for training)

---

### Configuration 2: CPESH Enabled, No Batch Optimization
```bash
# NOT recommended - too slow
export LNSP_ENABLE_CPESH=true
export LNSP_ENABLE_BATCH_EMBEDDINGS=false
export LNSP_TMD_MODE=hybrid
```

| Metric | Value |
|--------|-------|
| Time per chunk | ~1,244ms (estimated) |
| Time per article | ~178s |
| 500k articles | **1,027 days** |
| 10k article batch | **20.6 days** |

**Status**: ❌ Too slow for large-scale ingestion

---

### Configuration 3: CPESH Enabled + Batch Optimization (RECOMMENDED)
```bash
# Recommended for production
export LNSP_ENABLE_CPESH=true
export LNSP_ENABLE_BATCH_EMBEDDINGS=true
export LNSP_ENABLE_PARALLEL=true
export LNSP_MAX_PARALLEL_WORKERS=10
export LNSP_TMD_MODE=hybrid
```

**Optimizations Applied**:
1. Batch PostgreSQL writes (150ms → 30ms) = -120ms
2. Batch embeddings (19.4ms → 10ms) = -9.4ms
3. Parallel TMD extraction (no change, already fast)

**Estimated time per chunk**:
- Base: 244ms
- Add CPESH: +1,000ms
- Subtract batch savings: -129ms
- **Total: ~1,115ms per chunk**

| Metric | Value |
|--------|-------|
| Time per chunk | ~1,115ms |
| Time per article | ~159s (2.7 min) |
| **500k articles** | **920 days (2.5 years)** |
| **10k article batch** | **18.4 days** |

**Status**: ⚠️ Production-ready but very slow

---

### Configuration 4: Aggressive Optimization (ASPIRATIONAL)

**Additional optimizations needed**:
1. Batch CPESH extraction (10 chunks → 1 LLM call with batch prompt)
2. GPU-accelerated GTR-T5 with larger batch sizes
3. PostgreSQL connection pooling + prepared statements
4. Parallel article processing (8 workers)

**Estimated time per chunk**: ~400ms

| Metric | Value |
|--------|-------|
| Time per chunk | ~400ms |
| Time per article | ~57s |
| **500k articles** | **330 days (11 months)** |
| **10k article batch** | **6.6 days** |

**Status**: 🎯 Target performance (requires significant engineering)

---

## Recommendations for Big Run

### Option A: Fast Prototype (CPESH Disabled)
**Use Case**: Testing pipeline stability, infrastructure validation

```bash
# Run 10k articles as proof-of-concept
export LNSP_ENABLE_CPESH=false
export LNSP_ENABLE_BATCH_EMBEDDINGS=true
export LNSP_TMD_MODE=hybrid

./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 10000
```

**Time**: ~4.0 days
**Output**: 1.43M vectors (no CPESH data)
**Training-ready**: ❌ No (missing contrastive pairs)

---

### Option B: Production Data (CPESH Enabled, Batch 10k)
**Use Case**: Generate training-ready data for LVM

```bash
# Restart Ingest API with CPESH enabled
pkill -f "uvicorn.*ingest_chunks"

LNSP_ENABLE_CPESH=true \
LNSP_ENABLE_BATCH_EMBEDDINGS=true \
LNSP_ENABLE_PARALLEL=true \
LNSP_MAX_PARALLEL_WORKERS=10 \
LNSP_TMD_MODE=hybrid \
./.venv/bin/uvicorn app.api.ingest_chunks:app \
  --host 127.0.0.1 --port 8004 &

# Run 10k article batch
./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 10000
```

**Time**: ~18.4 days
**Output**: 1.43M vectors with full CPESH data
**Training-ready**: ✅ Yes

---

### Option C: Hybrid Approach (Recommended for Testing)
**Use Case**: Validate CPESH quality on small sample, then scale

**Phase 1: Small CPESH Sample (100 articles)**
```bash
# Enable CPESH, test 100 articles
export LNSP_ENABLE_CPESH=true
export LNSP_ENABLE_BATCH_EMBEDDINGS=true

./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --limit 100
```

**Time**: ~4.4 hours
**Output**: ~14.3k vectors with CPESH
**Purpose**: Validate CPESH quality, check database, verify training data format

**Phase 2: Review CPESH Quality**
```sql
-- Check CPESH completeness
SELECT
  count(*) as total,
  avg(jsonb_array_length(soft_negatives)) as avg_soft_negs,
  avg(jsonb_array_length(hard_negatives)) as avg_hard_negs,
  avg(CASE WHEN probe_question != '' THEN 1 ELSE 0 END) as probe_pct
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k';

-- Sample some CPESH data
SELECT
  concept_text,
  probe_question,
  expected_answer,
  soft_negatives,
  hard_negatives
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
LIMIT 5;
```

**Phase 3: Scale to 10k if Quality Good**
```bash
# Continue with larger batch
./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip 100 \
  --limit 9900
```

---

## Time Estimates Summary

### Per-Chunk Time Comparison

| Configuration | Time per Chunk | 10k Articles | 500k Articles |
|---------------|----------------|--------------|---------------|
| **Current (no CPESH)** | 244ms | 4.0 days | 201 days |
| **CPESH, no batch** | 1,244ms | 20.6 days | 1,027 days |
| **CPESH + batch** | 1,115ms | 18.4 days | 920 days |
| **Aggressive opt** | 400ms | 6.6 days | 330 days |

### Critical Numbers for Big Run Planning

**If you care about time per chunk/vector** (current best config):
- **Without CPESH**: 244ms per vector
- **With CPESH + batch optimization**: ~1,115ms per vector

**My recommendation**: Start with Option C (Hybrid Approach)
1. Test 100 articles with CPESH enabled (~4.4 hours)
2. Validate data quality
3. If good, scale to 10k (~18 days)
4. Pause, assess, then decide on full 500k

**Rationale**: You need CPESH for training, but 920 days is too long. Better to validate quality first, then potentially invest in aggressive optimizations (Option 4) before committing to full 500k ingestion.

---

## Current 100-Article Test Status

The running test is at **40% complete** (40/100 articles).

**Progress**:
- Articles processed: 40/100
- Estimated chunks so far: ~5,720
- Time elapsed: ~19.4 minutes
- Estimated completion: ~30 more minutes

**This test is running with CPESH disabled**, so it's validating pipeline stability at the fast 244ms/chunk rate.

---

## Next Steps

1. **Let 100-article test complete** - Validates pipeline stability
2. **Decide on CPESH**: Required for training, but adds 4.5x time
3. **If enabling CPESH**:
   - Restart Ingest API with `LNSP_ENABLE_CPESH=true`
   - Test 100 articles with CPESH (~4.4 hours)
   - Validate CPESH data quality via SQL queries
4. **For big run**:
   - Start with 10k articles (4 days without CPESH, 18 days with CPESH+batch)
   - Monitor disk space (PostgreSQL + FAISS storage)
   - Implement checkpointing for resumability

**Question for you**: Do you want training-quality data (with CPESH) or fast ingestion (without CPESH)?

- **Training-ready**: Must enable CPESH, accept 18+ days for 10k articles
- **Infrastructure test**: Keep CPESH disabled, run 10k in 4 days, but data won't be suitable for LVM training
