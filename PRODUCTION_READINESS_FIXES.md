# Production Readiness Fixes - Wikipedia Ingestion

**Date**: 2025-10-22
**Dataset**: wikipedia_500k (996 articles, 38,505 chunks)

## Issues Identified

### 1. Duplicate (article_index, chunk_index) Keys ❌
**Problem**: Episode-based chunking reused chunk_index within articles, creating 8,684 duplicate key pairs across 436 articles (43.8%).

**Example (Article 3596)**:
```
sequential_id | article_index | chunk_index | episode
--------------+---------------+-------------+----------
2498          | 3596          | 0           | 1
2499          | 3596          | 1           | 1
...
16597         | 3596          | 0           | 2  <- DUPLICATE KEY!
16598         | 3596          | 1           | 2  <- DUPLICATE KEY!
```

**Impact**:
- Breaks natural invariant that (article_index, chunk_index) should be unique
- Creates ambiguity when looking up chunks by article + index
- Prevents proper ordering within articles

### 2. Long-Tail Giant Chunks ⚠️
**Problem**: 739 chunks (1.92%) exceeded 400 chars, with max 11,781 chars (structured lists/tables).

**Impact**:
- Giant chunks dominate training gradients (lists, tables, structured data)
- Not suitable for narrative flow learning
- Skews vector space toward catalog-style content

## Fixes Implemented

### Fix 1: Flatten chunk_index to 0..N-1 per Article ✅

**SQL Update**:
```sql
WITH new_indices AS (
  SELECT
    cpe_id,
    ROW_NUMBER() OVER (
      PARTITION BY chunk_position->>'article_index'
      ORDER BY sequential_id
    ) - 1 AS new_chunk_index
  FROM cpe_entry
  WHERE dataset_source='wikipedia_500k'
)
UPDATE cpe_entry
SET chunk_position = jsonb_set(
  chunk_position,
  '{chunk_index}',
  to_jsonb(new_indices.new_chunk_index)
)
FROM new_indices
WHERE cpe_entry.cpe_id = new_indices.cpe_id;
```

**Result**:
- 38,505 rows updated
- 0 duplicate keys remaining
- (article_index, chunk_index) now uniquely identifies each chunk

**After (Article 3596)**:
```
sequential_id | article_index | chunk_index
--------------+---------------+-------------
2498          | 3596          | 0
2499          | 3596          | 1
...
2692          | 3596          | 194
16597         | 3596          | 195  <- Sequential, no duplicates!
16598         | 3596          | 196
```

### Fix 2: Create Train/Catalog Lane Views ✅

**lnsp_train_lane** (8-500 chars):
- **Purpose**: Clean prose chunks for LVM training
- **Count**: 37,712 chunks (97.94%)
- **Range**: 8-500 characters
- **Avg**: 190 characters

**lnsp_catalog_lane** (everything else):
- **Purpose**: Structured/metadata content for RAG/search
- **Count**: 793 chunks (2.06%)
- **Types**:
  - Giants (>500 chars): 739 chunks (lists, tables)
  - Microscopic (<8 chars): 54 chunks (section headers)

**Usage**:
```sql
-- Train LVM on clean prose
SELECT * FROM lnsp_train_lane ORDER BY sequential_id;

-- Use catalog for RAG retrieval
SELECT * FROM lnsp_catalog_lane WHERE catalog_type = 'giant';
```

### Fix 3: CI Gate for Quality Checks ✅

**Script**: `scripts/ci_gate_wikipedia_quality.sh`

**Checks**:
1. ✅ Giant chunks (>500 chars) - WARNING (739 found, moved to catalog)
2. ✅ Microscopic chunks (<8 chars) - WARNING (54 found, moved to catalog)
3. ✅ Duplicate keys - **PASS** (0 duplicates after flatten)
4. ⚠️ Duplicate text within articles - WARNING (1200 found, acceptable)
5. ✅ Train lane size - **PASS** (97.94% in optimal range)

**Run before ingestion**:
```bash
./scripts/ci_gate_wikipedia_quality.sh wikipedia_500k
```

## Data Quality Report

### Current State (After Fixes)

| Metric | Value | Status |
|--------|-------|--------|
| Total chunks | 38,505 | ✅ |
| Train lane (8-500 chars) | 37,712 (97.94%) | ✅ |
| Catalog lane | 793 (2.06%) | ✅ |
| Duplicate keys | 0 | ✅ |
| Avg train chunk size | 190 chars | ✅ |
| Articles processed | 996 | ✅ |

### Distribution by Lane

```
Train Lane (8-500 chars):     ████████████████████████████████████████ 97.94%
Catalog Lane (giants):        █ 1.92%
Catalog Lane (microscopic):   ▏ 0.14%
```

### Key Semantics (Fixed)

**Before**:
- ❌ (article_index, chunk_index) NOT unique
- ❌ 8,684 duplicate key pairs
- ❌ Ambiguous chunk lookup

**After**:
- ✅ (article_index, chunk_index) IS unique
- ✅ 0 duplicate keys
- ✅ Unambiguous chunk ordering within articles

## Usage Examples

### Training Data (LVM)
```sql
-- Get sequential training chunks for article 3596
SELECT
  sequential_id,
  article_index,
  chunk_index,
  char_count,
  concept_text
FROM lnsp_train_lane
WHERE article_index = 3596
ORDER BY chunk_index;
```

### Catalog Data (RAG/Search)
```sql
-- Get giant chunks for comprehensive RAG retrieval
SELECT
  article_title,
  char_count,
  catalog_type,
  concept_text
FROM lnsp_catalog_lane
WHERE catalog_type = 'giant'
ORDER BY char_count DESC
LIMIT 10;
```

### Quality Verification
```bash
# Run CI gate before ingestion
./scripts/ci_gate_wikipedia_quality.sh wikipedia_500k

# Expected output:
# ✅ PASS: No duplicate keys
# ✅ PASS: Train lane 97.94%
# ⚠️  WARNING: 739 giants (expected, moved to catalog)
# CI GATE: PASSED
```

## Production Readiness Status

| Issue | Status | Notes |
|-------|--------|-------|
| Duplicate keys | ✅ **FIXED** | Flattened to 0..N-1 per article |
| Giant chunks | ✅ **SEPARATED** | Moved to catalog lane |
| Microscopic chunks | ✅ **SEPARATED** | Moved to catalog lane |
| Train/catalog lanes | ✅ **CREATED** | Views for each lane |
| CI gates | ✅ **IMPLEMENTED** | Quality checks automated |
| Key semantics | ✅ **FIXED** | Unique (article, chunk) pairs |

**Overall**: ✅ **PRODUCTION READY**

## Next Steps

1. **Continue ingestion**: Process next 3,000 articles with improved pipeline
2. **Monitor CI gate**: Run quality checks after each batch
3. **Train LVM**: Use `lnsp_train_lane` for sequential vector prediction
4. **RAG integration**: Use `lnsp_catalog_lane` for comprehensive retrieval

## Files Modified

- `/tools/ingest_wikipedia_pipeline.py` - Added article metadata, flattened chunk indices
- `/app/api/ingest_chunks.py` - Store complete chunk_position metadata
- `/scripts/ci_gate_wikipedia_quality.sh` - CI gate for quality checks (NEW)
- `cpe_entry` table - Added sequential_id, flattened chunk_position
- Views: `lnsp_train_lane`, `lnsp_catalog_lane` (NEW)
