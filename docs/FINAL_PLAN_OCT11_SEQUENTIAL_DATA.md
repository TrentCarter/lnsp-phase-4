# FINAL PLAN: Sequential Training Data for LVM
**Date**: October 11, 2025
**Status**: ALL DECISIONS LOCKED IN - READY FOR EXECUTION
**Owner**: Trent Carter

---

## 🎯 Executive Summary

**Problem**: LVM training data uses ontological hierarchies (taxonomic classifications), not sequential narratives needed for autoregressive models.

**Solution**: Replace with sequential document data (Wikipedia, textbooks) and add `document_id + sequence_index` schema for fast ordered retrieval.

**Impact**: Enables proper LVM training with 100K+ coherent sequences.

---

## ✅ LOCKED IN DECISIONS (Oct 11, 2025)

### Decision 1: `document_id` Format
**APPROVED**: Option A - `"wikipedia_12345"` (source + article ID)

**Implementation**:
```python
document_id = f"{source}_{article_id}"
# Examples:
# "wikipedia_12345"
# "openstax_biology_chapter1"
# "arxiv_2301.00123"
```

---

### Decision 2: `sequence_index` Reset Strategy
**APPROVED**: Option A - Reset to 0 per episode

**Rationale**: Cleaner episode boundaries, easier to skip low-coherence transitions

**Implementation**:
```python
# Episode 1:
chunks = [
    {"episode_id": "ep1", "sequence_index": 0},
    {"episode_id": "ep1", "sequence_index": 1},
    {"episode_id": "ep1", "sequence_index": 2}
]

# Episode 2:
chunks = [
    {"episode_id": "ep2", "sequence_index": 0},  # Resets to 0
    {"episode_id": "ep2", "sequence_index": 1}
]
```

---

### Decision 3: Parent/Child Population
**APPROVED**: Option B - Only within same episode

**Rationale**: Respects coherence boundaries, prevents linking across low-coherence transitions

**Implementation**:
```python
def populate_parent_child(chunks):
    for i, chunk in enumerate(chunks):
        if i > 0 and chunk.episode_id == chunks[i-1].episode_id:
            chunk.parent_cpe_id = chunks[i-1].cpe_id
            chunks[i-1].child_cpe_id = chunk.cpe_id
        else:
            # Episode boundary - no parent link
            chunk.parent_cpe_id = None
```

---

### Decision 4: Migration Strategy
**APPROVED**: Re-ingest existing data (NOT backfill)

**Rationale**:
- Existing data (495 chunks) represents only hours of effort
- Re-ingestion ensures all fields populated correctly
- Clean slate avoids partial/missing metadata

**Action**:
1. Delete existing non-test data
2. Re-run ingestion pipeline with new schema
3. Verify all fields (document_id, sequence_index, episode_id, parent/child) populated

---

### Decision 5: Coherence Threshold
**APPROVED**: Wait for test results

**Process**:
1. Apply migration
2. Re-ingest watercycle-mini data
3. Run coherence test
4. Set threshold based on empirical results
5. Adjust chunking parameters if needed

---

## 🗄️ FINAL SCHEMA (Approved)

```sql
-- Apply this migration AFTER re-ingesting data
ALTER TABLE cpe_entry
ADD COLUMN document_id TEXT NOT NULL,           -- "wikipedia_12345"
ADD COLUMN sequence_index INTEGER NOT NULL,     -- 0, 1, 2... (resets per episode)
ADD COLUMN episode_id TEXT,                     -- "ep1", "ep2", ...
ADD COLUMN parent_cpe_id UUID,                  -- Previous chunk (within episode)
ADD COLUMN child_cpe_id UUID,                   -- Next chunk (within episode)
ADD COLUMN last_accessed TIMESTAMP;

-- Critical performance index
CREATE INDEX idx_document_sequence ON cpe_entry(document_id, sequence_index);
CREATE INDEX idx_episode ON cpe_entry(episode_id, sequence_index) WHERE episode_id IS NOT NULL;
CREATE INDEX idx_parent_cpe ON cpe_entry(parent_cpe_id) WHERE parent_cpe_id IS NOT NULL;
CREATE INDEX idx_child_cpe ON cpe_entry(child_cpe_id) WHERE child_cpe_id IS NOT NULL;
```

**Query Performance**:
- Ordered retrieval: O(log N) with composite index
- Target: <10ms for 1000 chunks

---

## 🏗️ FINAL PIPELINE (Approved)

### Architecture: Option A (Simplified)

```
Wikipedia Article (50KB)
    ↓
[Episode Chunker]
  - Simple mode (paragraph-based)
  - Assigns episode_id
  - Coherence threshold: 0.6
    ↓
Episodes (5x ~10KB coherent spans)
    ↓
[Semantic Chunker :8001]
  - Breakpoint: 75
  - Per episode
  - Assigns document_id + sequence_index
    ↓
Fine Chunks (50x ~1KB concepts)
    ↓
[TMD Extraction :8002]
  - Extract Domain/Task/Modifier codes
    ↓
[GTR-T5 Embeddings :8765]
  - Batch embeddings (768D)
    ↓
[Ingest :8004]
  - Populate parent/child (within episode)
  - Write to PostgreSQL + FAISS
    ↓
PostgreSQL (cpe_entry + cpe_vectors) + FAISS
```

---

## 📋 EXECUTION PLAN (Step-by-Step)

### Phase 0: Cleanup (IMMEDIATE)
```bash
# 1. Backup current data (optional)
pg_dump lnsp > backups/lnsp_before_migration_oct11.sql

# 2. Delete non-test data
psql lnsp -c "DELETE FROM cpe_vectors WHERE cpe_id IN (
    SELECT cpe_id FROM cpe_entry
    WHERE dataset_source NOT LIKE 'test%'
    AND dataset_source NOT LIKE 'smoke%'
);"

psql lnsp -c "DELETE FROM cpe_entry
WHERE dataset_source NOT LIKE 'test%'
AND dataset_source NOT LIKE 'smoke%';"

# 3. Verify deletion
psql lnsp -c "SELECT dataset_source, COUNT(*) FROM cpe_entry GROUP BY dataset_source;"
```

---

### Phase 1: Schema Migration (Oct 11)
```bash
# Apply migration
psql lnsp < migrations/003_add_sequence_metadata.sql

# Verify schema
psql lnsp -c "\d cpe_entry" | grep -E "(document_id|sequence_index|episode_id|parent_cpe_id|child_cpe_id)"

# Verify indexes
psql lnsp -c "\di" | grep -E "(idx_document_sequence|idx_episode|idx_parent_cpe|idx_child_cpe)"
```

---

### Phase 2: Re-Ingest Watercycle Data (Oct 11)
```bash
# Re-run ingestion with new schema
# (Update ingestion code to populate new fields first)

# Expected result:
# - 495 chunks
# - All have document_id, sequence_index, episode_id
# - Parent/child populated within episodes
```

---

### Phase 3: Validation (Oct 11)
```bash
# Fix validation script to use document_id + sequence_index
# (Edit tools/test_sequential_coherence.py)

# Run coherence test
LNSP_DB_USER=lnsp ./tools/test_sequential_coherence.py \
  --dataset "watercycle-mini|semantic-75" \
  --test-count 50 \
  --walk-count 5

# Expected output:
# - Mean coherence: ??? (empirical result)
# - Set threshold based on results
```

---

### Phase 4: Pilot Test (Oct 12)
```bash
# Ingest 10 Wikipedia articles
./tools/fetch_serialize_episodes.py \
  --config data/wikipedia_pilot.yaml \
  --outdir artifacts/episodes/ \
  --tau-local 0.6 \
  --target-len 512

# Verify:
# - document_id format correct
# - sequence_index ordering correct
# - episode_id assigned
# - parent/child within episodes
# - Coherence >80%
```

---

### Phase 5: Full Ingestion (Oct 13-15)
```bash
# Ingest 3000 Wikipedia articles
./tools/fetch_serialize_episodes.py \
  --config data/wikipedia_full.yaml \
  --outdir artifacts/episodes/ \
  --tau-local 0.6

# Monitor performance:
# - Target: 3 articles/sec
# - Total time: ~17 minutes
# - Total chunks: 100K+

# Validate coherence
LNSP_DB_USER=lnsp ./tools/test_sequential_coherence.py \
  --dataset "wikipedia-%" \
  --test-count 100 \
  --order random
```

---

### Phase 6: Export Training Data (Oct 16-17)
```bash
# Create export script
# tools/export_training_sequences.py

# Run export
./tools/export_training_sequences.py \
  --dataset "wikipedia-%" \
  --output artifacts/lvm/wikipedia_training_sequences.npz \
  --min-chain-length 3

# Expected output:
# - 100K+ training pairs
# - NPZ file >50MB
# - Format: {X: context_vectors, y: target_vectors}
```

---

## 🔧 CODE CHANGES REQUIRED

### 1. Update Ingestion API (app/api/ingest_chunks.py)

**Add to ChunkInput model**:
```python
class ChunkInput(BaseModel):
    text: str
    document_id: str          # NEW: "wikipedia_12345"
    sequence_index: int       # NEW: 0, 1, 2...
    episode_id: Optional[str] # NEW: "ep1", "ep2", ...
    # ... existing fields
```

**Add parent/child population**:
```python
async def ingest_chunks_endpoint(request: ChunksRequest):
    # ... existing code

    # Populate parent/child within episodes
    for i, chunk in enumerate(request.chunks):
        if i > 0 and chunk.episode_id == request.chunks[i-1].episode_id:
            chunk_data.parent_cpe_id = previous_cpe_id
            # Update previous chunk's child_cpe_id
            await update_child_cpe_id(previous_cpe_id, chunk_data.cpe_id)

        previous_cpe_id = chunk_data.cpe_id
```

---

### 2. Update Episode Chunker (tools/fetch_serialize_episodes.py)

**Already has coherence detection** - just ensure it assigns:
```python
episode_id = f"ep{episode_index}"
document_id = f"wikipedia_{article_id}"
```

---

### 3. Fix Validation Script (tools/test_sequential_coherence.py)

**Replace**:
```python
# OLD:
cursor.execute("SELECT MIN(id), MAX(id) FROM cpe_entry")

# NEW:
cursor.execute("""
    SELECT document_id, MIN(sequence_index), MAX(sequence_index)
    FROM cpe_entry
    WHERE dataset_source = %s
    GROUP BY document_id
""", (dataset_source,))
```

**Query consecutive chunks**:
```python
# OLD:
WHERE ce.id >= %s ORDER BY ce.id LIMIT %s

# NEW:
WHERE ce.document_id = %s
ORDER BY ce.sequence_index
LIMIT %s
```

---

### 4. Create Training Data Exporter (tools/export_training_sequences.py)

```python
#!/usr/bin/env python3
"""Export training sequences from sequential document data."""

import psycopg2
import numpy as np
from typing import List

def export_training_sequences(dataset_pattern: str, output_path: str):
    conn = psycopg2.connect(dbname="lnsp", user="lnsp")

    # Get all documents
    docs = conn.execute("""
        SELECT DISTINCT document_id
        FROM cpe_entry
        WHERE dataset_source LIKE %s
    """, (dataset_pattern,)).fetchall()

    X_sequences = []
    y_targets = []

    for (document_id,) in docs:
        # Get ordered chunks
        chunks = conn.execute("""
            SELECT cv.concept_vec, ce.sequence_index, ce.episode_id
            FROM cpe_entry ce
            JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
            WHERE ce.document_id = %s
            ORDER BY ce.episode_id, ce.sequence_index
        """, (document_id,)).fetchall()

        # Generate training pairs
        for i in range(len(chunks) - 1):
            vec_i, idx_i, ep_i = chunks[i]
            vec_j, idx_j, ep_j = chunks[i+1]

            # Skip episode boundaries
            if ep_i == ep_j:
                X_sequences.append(vec_i)
                y_targets.append(vec_j)

    # Save NPZ
    np.savez(
        output_path,
        X=np.array(X_sequences, dtype=np.float32),
        y=np.array(y_targets, dtype=np.float32),
        metadata={
            'num_documents': len(docs),
            'num_sequences': len(X_sequences)
        }
    )

    print(f"✓ Exported {len(X_sequences)} sequences to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    export_training_sequences(args.dataset, args.output)
```

---

## 📊 SUCCESS CRITERIA

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Sequential coherence** | >80% | Run validation script on all documents |
| **Mean cosine similarity** | >0.6 | Per-document coherence test |
| **Total chunks** | 100K+ | `SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'wikipedia-%'` |
| **Unique documents** | 3000+ | `SELECT COUNT(DISTINCT document_id) FROM cpe_entry` |
| **Query performance** | <10ms | `EXPLAIN ANALYZE` on ordered retrieval query |
| **Parent/child populated** | 100% | `SELECT COUNT(*) FROM cpe_entry WHERE parent_cpe_id IS NOT NULL` |
| **Training data size** | >50MB | `ls -lh artifacts/lvm/wikipedia_training_sequences.npz` |

---

## 📄 DOCUMENTATION LOCATIONS

| Document | Location | Status |
|----------|----------|--------|
| **Final Plan** | `docs/FINAL_PLAN_OCT11_SEQUENTIAL_DATA.md` | ✅ This doc |
| **PRD** | `docs/PRDs/PRD_Sequential_Training_Data.md` | ✅ Complete |
| **Sprint Plan** | `sprints/sprint_10112025_S1.md` | ✅ Updated |
| **Migration Script** | `migrations/003_add_sequence_metadata.sql` | ✅ Ready |
| **Session Summary** | `docs/SESSION_SUMMARY_OCT11_SEQUENTIAL_DATA.md` | ✅ Complete |
| **Archive README** | `artifacts/archive/ontological_DEPRECATED_20251011/README.md` | ✅ Complete |
| **CLAUDE.md** | `CLAUDE.md` | ✅ Updated (ontology warnings) |

---

## 🚀 NEXT ACTIONS (Priority Order)

### IMMEDIATE (Before /clear)
✅ All decisions locked in
✅ All documentation created
✅ Ready to execute

### AFTER /clear
1. **Delete non-test data** (cleanup existing chunks)
2. **Apply migration** (`psql lnsp < migrations/003_add_sequence_metadata.sql`)
3. **Update ingestion code** (populate document_id, sequence_index, episode_id, parent/child)
4. **Re-ingest watercycle data** (test new schema)
5. **Fix validation script** (use document_id + sequence_index)
6. **Run coherence test** (establish baseline)
7. **Pilot test** (10 Wikipedia articles)
8. **Full ingestion** (3000 articles)
9. **Export training data** (100K+ sequences)

---

## 💡 KEY INSIGHTS

### Insight 1: Re-Ingestion > Backfill
For small datasets (<1000 chunks), re-ingestion is cleaner than backfilling partial metadata.

### Insight 2: document_id + sequence_index = Fast
O(log N) performance with composite index vs O(N) with parent-child traversal.

### Insight 3: Episode Boundaries Matter
Resetting sequence_index per episode makes it easier to skip low-coherence transitions during training.

### Insight 4: Parent/Child Still Useful
Keep for future graph training, but don't use for sequence retrieval (slow!).

---

## ⚠️ CRITICAL REMINDERS

1. **NEVER use ontological data for LVM training** (archived, not deleted)
2. **Coherence is testable pre-training** (don't wait for inference!)
3. **Simplicity > Complexity** (Option A pipeline, not over-engineered)
4. **Test on pilot before full scale** (10 articles → 3000 articles)
5. **Query performance matters** (indexed queries, not recursive CTEs)

---

**ALL DECISIONS LOCKED IN. DOCUMENTATION COMPLETE. READY FOR EXECUTION.**

**Owner**: Trent Carter
**Date**: October 11, 2025
**Status**: ✅ APPROVED - PROCEED TO PHASE 0 (CLEANUP)
