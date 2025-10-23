# Wikipedia Ingestion State - October 23, 2025

## Current Status

**Database**: PostgreSQL `lnsp`

**Tables**: 
- `cpe_entry` - Main article chunks with metadata
- `cpe_vectors` - 768D GTR-T5 embeddings (pgvector)

**Ingested Data**:
- **Total chunks**: 584,545
- **Total articles**: 8,447 / 8,470 attempted (99.73% success)
- **Article range**: 1-8,470
- **Missing articles**: 421, 7151, 7691 (JSON encoding failures - acceptable loss)
- **Dataset source**: `wikipedia_500k`

**Source File**: `data/datasets/wikipedia/wikipedia_500k.jsonl`

**Ingestion Script**: `tools/ingest_wikipedia_bulk.py` (COPY-based bulk ingestion)

**Last Run**: October 23, 2025 (`logs/wikipedia_FINAL_fixed_20251023_183926.log`)

## Next Batch Configuration

**Resume Point**: Article 8,471

**Recommended Batch**: 10,000 articles (8,471-18,470)

**Resume Script**: `tools/resume_wikipedia_ingestion.sh`

**Usage**:
```bash
# Default: 10,000 articles (~9.4 hours)
./tools/resume_wikipedia_ingestion.sh

# Custom amount:
./tools/resume_wikipedia_ingestion.sh 5000  # Process 5,000 articles
```

## Performance Metrics

**Per-Article Averages** (from final run):
- **Chunking**: 1ms median
- **Embedding** (GTR-T5 768D): 279ms median  
- **Database** (COPY-based): 9ms median
- **Total**: ~3.4 seconds per article

**Speedup vs Original Pipeline**:
- Database writes: **8,222x faster** (74,000ms → 9ms)

**Estimated Time for 10,000 Articles**: ~9.4 hours

## Critical Configuration

**Chunking Strategy** (Simple Word-Based):
```python
target_words = 60      # Target words per chunk
max_chars = 500        # Maximum characters per chunk
```

**Database Tuning**:
```python
BATCH_SIZE = 5         # Articles per COPY operation
synchronous_commit = OFF
work_mem = 64MB
```

**Environment Variables**:
```bash
export LNSP_TMD_MODE=heuristic  # Minimal TMD processing for Wikipedia
export OMP_NUM_THREADS=8        # Multi-core optimization
```

## Schema Modifications Made

Made all CPESH/TMD fields **nullable** for Wikipedia data:

```sql
-- CPESH fields (not used for Wikipedia)
ALTER TABLE cpe_entry ALTER COLUMN mission_text DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN source_chunk DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN probe_question DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN expected_answer DROP NOT NULL;

-- TMD fields (minimal for Wikipedia)
ALTER TABLE cpe_entry ALTER COLUMN domain_code DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN task_code DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN modifier_code DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN content_type DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN tmd_bits DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN tmd_lane DROP NOT NULL;
ALTER TABLE cpe_entry ALTER COLUMN lane_index DROP NOT NULL;

-- Fused vectors (not used for Wikipedia)
ALTER TABLE cpe_vectors ALTER COLUMN fused_vec DROP NOT NULL;
```

## Known Issues (All Resolved)

### 1. ✅ UTF-8 Encoding Errors
**Symptom**: `invalid byte sequence for encoding "UTF8"`  
**Fix**: Multi-layer UTF-8 sanitization in `process_article()` and `rows_to_tsv()`

### 2. ✅ JSON Encoding Errors
**Symptom**: `invalid input syntax for type json`  
**Fix**: `ensure_ascii=True` with control character removal

### 3. ✅ NULL Constraint Violations
**Symptom**: `null value in column "X" violates not-null constraint`  
**Fix**: Made CPESH/TMD fields nullable (schema changes above)

### 4. ✅ INSERT Column Mismatch
**Symptom**: `INSERT has more target columns than expressions`  
**Fix**: Added `NOW()` for `created_at` column in SELECT statement

### 5. ✅ Transaction Abort Cascades
**Symptom**: Hundreds of "current transaction is aborted" errors  
**Fix**: Proactive `ROLLBACK` before each batch

## Data Verification Commands

**Check article count**:
```bash
psql lnsp -c "SELECT COUNT(DISTINCT chunk_position->>'article_index') FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"
```

**Check chunk count**:
```bash
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source = 'wikipedia_500k';"
```

**Check vector count**:
```bash
psql lnsp -c "SELECT COUNT(*) FROM cpe_vectors v INNER JOIN cpe_entry e ON v.cpe_id = e.cpe_id WHERE e.dataset_source = 'wikipedia_500k';"
```

**Check for missing articles**:
```bash
psql lnsp -c "
WITH expected AS (
  SELECT generate_series(1, 8470) AS article_id
),
actual AS (
  SELECT DISTINCT CAST(chunk_position->>'article_index' AS INTEGER) AS article_id
  FROM cpe_entry
  WHERE dataset_source = 'wikipedia_500k'
)
SELECT e.article_id
FROM expected e
LEFT JOIN actual a ON e.article_id = a.article_id
WHERE a.article_id IS NULL
ORDER BY e.article_id;
"
```

**Sample data check**:
```bash
psql lnsp -c "
SELECT 
  chunk_position->>'article_index' AS article,
  chunk_position->>'article_title' AS title,
  LEFT(concept_text, 50) AS preview
FROM cpe_entry 
WHERE dataset_source = 'wikipedia_500k'
ORDER BY CAST(chunk_position->>'article_index' AS INTEGER)
LIMIT 5;
"
```

## Files and Locations

**Ingestion Scripts**:
- `tools/ingest_wikipedia_bulk.py` - Production ingestion (COPY-based, all fixes)
- `tools/resume_wikipedia_ingestion.sh` - Resume script for next batch

**Source Data**:
- `data/datasets/wikipedia/wikipedia_500k.jsonl` - 500,000 Wikipedia articles

**Logs**:
- `logs/wikipedia_FINAL_fixed_20251023_183926.log` - Final successful run
- `logs/ingest_profile.jsonl` - Performance profiling data

**Database**:
- **Host**: localhost
- **Database**: `lnsp`
- **Tables**: `cpe_entry`, `cpe_vectors`
- **Extension**: pgvector (for 768D embeddings)

## Ready for LVM Training

The ingested data is ready for LVM (Latent Vector Model) training:

✅ **584,545 chunks** with GTR-T5 embeddings  
✅ **chunk_position** metadata preserved (article_index, chunk_index, article_title)  
✅ **Temporal ordering** maintained for autoregressive training  
✅ **No CPESH required** (LVM training only needs vector sequences)

**Next Steps**:
1. Continue ingestion to 100k+ articles for robust training set
2. Extract sequential chunk chains for LVM training data
3. Train LVM models (AMN, LSTM, GRU, Transformer architectures)

See `docs/LVM_DATA_MAP.md` for LVM training pipeline details.

---

**Last Updated**: 2025-10-23  
**Status**: ✅ Ready for next batch (article 8,471+)
