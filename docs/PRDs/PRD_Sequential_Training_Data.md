# PRD: Sequential Training Data for LVM

**Date**: October 11, 2025
**Status**: Approved
**Owner**: Trent Carter
**Technical Lead**: Claude

---

## 🎯 Problem Statement

### Current State (BROKEN)
LVM training data consists of **ontological hierarchies** (WordNet, SWO, GO) which are:
- ❌ **Taxonomic classifications** ("dog → mammal → animal")
- ❌ **NOT sequential narratives** (no temporal/causal flow)
- ❌ **Unsuitable for autoregressive models** (predict next vector from context)

### Desired State (FIXED)
LVM training data should consist of **sequential document chunks** which are:
- ✅ **Narrative/expository text** ("Process begins → Step 1 → Step 2 → Conclusion")
- ✅ **Temporal/causal relationships** (coherent progression)
- ✅ **Suitable for autoregressive training** (next-vector prediction)

---

## 📋 Requirements

### Functional Requirements

#### FR1: Sequential Ordering
**Requirement**: Every chunk must have a deterministic position in its source document.

**Implementation**:
- `document_id`: Groups chunks from the same source
- `sequence_index`: Integer order within document (0, 1, 2, ...)
- `episode_id`: (Optional) Groups coherent paragraphs within document

**Acceptance Criteria**:
- ✅ Can retrieve chunks in document order with single SQL query
- ✅ Query performance <10ms for 1000 chunks (O(log N) with index)

---

#### FR2: Sequential Coherence
**Requirement**: Consecutive chunks must maintain semantic similarity >0.6 (cosine).

**Validation**:
```python
for i in range(len(chunks) - 1):
    similarity = cosine(chunks[i].vector, chunks[i+1].vector)
    assert similarity > 0.6, "Chunks are incoherent"
```

**Acceptance Criteria**:
- ✅ >80% of documents pass coherence test (mean similarity >0.6)
- ✅ Coherence tested BEFORE training (pre-validation)

---

#### FR3: Data Volume
**Requirement**: 100K+ sequential chunks from diverse sources.

**Sources** (Priority order):
1. Wikipedia articles (1000+ articles)
2. OpenStax textbooks (if needed)
3. Scientific papers (arXiv, if needed)

**Acceptance Criteria**:
- ✅ 100K+ chunks ingested
- ✅ From 1000+ unique documents
- ✅ Dataset source: `wikipedia-*`, `openstax-*` (NOT `ontology-*`)

---

#### FR4: Fast Training Data Extraction
**Requirement**: Extract ordered training sequences in <1 second per document.

**Query Performance**:
```sql
-- Must complete in <10ms:
SELECT cv.concept_vec, ce.sequence_index
FROM cpe_entry ce
JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
WHERE ce.document_id = 'wikipedia_12345'
ORDER BY ce.sequence_index;
```

**Acceptance Criteria**:
- ✅ Single query per document (no recursive CTEs)
- ✅ O(log N) performance with composite index
- ✅ Extract 100K sequences in <2 minutes total

---

### Non-Functional Requirements

#### NFR1: Schema Design
**Requirement**: Minimal overhead, maximum query performance.

**Schema**:
```sql
ALTER TABLE cpe_entry
ADD COLUMN document_id TEXT NOT NULL,
ADD COLUMN sequence_index INTEGER NOT NULL,
ADD COLUMN episode_id TEXT,
ADD COLUMN parent_cpe_id UUID,      -- Optional: for graph training
ADD COLUMN child_cpe_id UUID,       -- Optional: for graph training
ADD COLUMN last_accessed TIMESTAMP;

CREATE INDEX idx_document_sequence ON cpe_entry(document_id, sequence_index);
CREATE INDEX idx_episode ON cpe_entry(episode_id, sequence_index);
```

**Acceptance Criteria**:
- ✅ Index overhead <5% of table size
- ✅ Query performance <10ms for ordered retrieval
- ✅ Parent/child populated if low overhead (<10ms per chunk)

---

#### NFR2: Backward Compatibility
**Requirement**: Existing data (watercycle-mini) must be migrated without loss.

**Migration Strategy**:
```sql
-- Backfill from existing chunk_position JSON
UPDATE cpe_entry
SET document_id = dataset_source || '_' || COALESCE(batch_id, cpe_id::text),
    sequence_index = COALESCE((chunk_position->>'index')::integer, 0)
WHERE chunk_position IS NOT NULL;

-- Default values for legacy data
UPDATE cpe_entry
SET document_id = 'legacy_' || dataset_source,
    sequence_index = 0
WHERE document_id IS NULL;
```

**Acceptance Criteria**:
- ✅ Zero data loss during migration
- ✅ watercycle-mini chunks (495) retain order
- ✅ Legacy data marked as `legacy_*` (skipped for training)

---

## 🏗️ Technical Design

### Schema Changes

```sql
-- Migration: 003_add_sequence_metadata.sql

-- Add sequential ordering columns
ALTER TABLE cpe_entry
ADD COLUMN document_id TEXT NOT NULL DEFAULT 'unknown',
ADD COLUMN sequence_index INTEGER NOT NULL DEFAULT 0,
ADD COLUMN episode_id TEXT,
ADD COLUMN parent_cpe_id UUID,
ADD COLUMN child_cpe_id UUID,
ADD COLUMN last_accessed TIMESTAMP;

-- Performance indexes
CREATE INDEX idx_document_sequence ON cpe_entry(document_id, sequence_index);
CREATE INDEX idx_episode ON cpe_entry(episode_id, sequence_index);
CREATE INDEX idx_parent_child ON cpe_entry(parent_cpe_id, child_cpe_id);

-- Foreign key constraints (optional - adds overhead)
-- ALTER TABLE cpe_entry ADD CONSTRAINT fk_parent FOREIGN KEY (parent_cpe_id) REFERENCES cpe_entry(cpe_id);
-- ALTER TABLE cpe_entry ADD CONSTRAINT fk_child FOREIGN KEY (child_cpe_id) REFERENCES cpe_entry(cpe_id);
```

---

### Data Flow (2-Stage Pipeline)

```
Wikipedia Article (50KB)
    ↓
[Episode Chunker] (Simple mode, 1-10 paragraphs)
    ↓ Assigns episode_id
Episodes (5x ~10KB spans)
    ↓
[Semantic Chunker] (Breakpoint=75, per episode)
    ↓ Assigns document_id + sequence_index
Fine Chunks (50x ~1KB concepts)
    ↓
[Ingest API] (Populates parent/child if same episode)
    ↓
PostgreSQL (cpe_entry + cpe_vectors)
```

---

### Training Data Extraction

```python
#!/usr/bin/env python3
"""
Extract training sequences from sequential document data.
"""

def export_training_sequences(dataset_source_pattern: str, output_path: str):
    # Get all documents
    documents = db.query("""
        SELECT DISTINCT document_id
        FROM cpe_entry
        WHERE dataset_source LIKE %s
        ORDER BY document_id
    """, dataset_source_pattern)

    X_sequences = []
    y_targets = []

    for doc in documents:
        # Get ordered chunks (fast: O(log N) with index)
        chunks = db.query("""
            SELECT cv.concept_vec, ce.sequence_index, ce.episode_id
            FROM cpe_entry ce
            JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
            WHERE ce.document_id = %s
            ORDER BY ce.sequence_index
        """, doc.document_id)

        # Generate training pairs: predict next vector
        for i in range(len(chunks) - 1):
            # Skip episode boundaries (coherence drops)
            if chunks[i].episode_id == chunks[i+1].episode_id:
                X_sequences.append(chunks[i].concept_vec)
                y_targets.append(chunks[i+1].concept_vec)

    # Save to NPZ
    np.savez(
        output_path,
        X=np.array(X_sequences, dtype=np.float32),
        y=np.array(y_targets, dtype=np.float32),
        metadata={
            'source_pattern': dataset_source_pattern,
            'num_documents': len(documents),
            'num_sequences': len(X_sequences)
        }
    )

    print(f"✓ Exported {len(X_sequences)} training sequences to {output_path}")
```

---

## ✅ Success Criteria

### Quantitative Metrics

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Sequential coherence** | >80% | `tools/test_sequential_coherence.py` |
| **Mean cosine similarity** | >0.6 | Per-document coherence test |
| **Total chunks** | 100K+ | `SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE 'wikipedia-%'` |
| **Unique documents** | 1000+ | `SELECT COUNT(DISTINCT document_id) FROM cpe_entry` |
| **Query performance** | <10ms | EXPLAIN ANALYZE on ordered retrieval |
| **Training extraction** | <2 min | Benchmark export_training_sequences.py |

### Qualitative Criteria

- ✅ **No ontological data** in training set (archived to `artifacts/archive/ontological_DEPRECATED_20251011/`)
- ✅ **All sequences from documents** (Wikipedia, textbooks, papers)
- ✅ **Episode boundaries respected** (parent/child NULL across episodes)
- ✅ **Fast SQL queries** (indexed, no recursive CTEs)

---

## 🚀 Implementation Plan

### Phase 1: Schema Migration (Oct 11) ✅ IN PROGRESS
- [x] Create migration script (`migrations/003_add_sequence_metadata.sql`)
- [ ] Apply to database
- [ ] Backfill existing data (watercycle-mini)
- [ ] Verify indexes created

### Phase 2: Validation (Oct 11-12)
- [ ] Fix `tools/test_sequential_coherence.py` (use document_id + sequence_index)
- [ ] Test on watercycle-mini (establish baseline)
- [ ] Confirm coherence threshold (0.6 or adjust)

### Phase 3: Pilot Ingestion (Oct 12-13)
- [ ] Ingest 10 Wikipedia articles (test pipeline)
- [ ] Validate document_id assignment
- [ ] Validate sequence_index ordering
- [ ] Test coherence (should be >80%)

### Phase 4: Full Ingestion (Oct 13-15)
- [ ] Ingest 3000 Wikipedia articles
- [ ] Validate coherence >80%
- [ ] Fix any issues
- [ ] Expand to 10K+ articles if time permits

### Phase 5: Training Data Export (Oct 16-17)
- [ ] Create `tools/export_training_sequences.py`
- [ ] Export to `artifacts/lvm/wikipedia_training_sequences.npz`
- [ ] Validate NPZ format (X, y, metadata)
- [ ] Benchmark extraction speed

---

## 📊 Open Questions

### Q1: document_id Format
**Options**:
- `"wikipedia_12345"` (source + ID)
- `"https://en.wikipedia.org/wiki/Article"` (full URL)
- `"wiki-en-Article_Name"` (source-lang-title)

**Decision**: **Pending** (recommend `wikipedia_12345`)

---

### Q2: sequence_index Reset Strategy
**Options**:
- Reset to 0 per episode
- Continuous across episodes

**Decision**: **Pending** (recommend reset per episode)

---

### Q3: Parent/Child Population
**Options**:
- Always (within document)
- Only within episode
- Manual population later

**Decision**: **Pending** (recommend within-episode only)

---

### Q4: Coherence Threshold
**Current**: 0.6 (tentative)

**Decision**: Wait for watercycle-mini test results, adjust if needed

---

## 🔗 Related Documents

- **Sprint Plan**: `sprints/sprint_10112025_S1.md`
- **Migration Script**: `migrations/003_add_sequence_metadata.sql`
- **Archive Explanation**: `artifacts/archive/ontological_DEPRECATED_20251011/README.md`
- **CLAUDE.md**: Updated with ontology warnings (Oct 11, 2025)
- **Database Locations**: `docs/DATABASE_LOCATIONS.md`

---

## 📝 Notes

### Ontologies Still Useful for GraphRAG
**Important**: This PRD deprecates ontologies for **LVM training** only.

**Ontologies remain valuable for**:
- ✅ Neo4j knowledge graphs
- ✅ GraphRAG retrieval (neighbor expansion)
- ✅ Relation extraction
- ✅ Concept classification

**Do NOT delete ontology data** - just don't use for autoregressive training!

---

**Approved by**: Trent Carter (Oct 11, 2025)
**Status**: Implementation in progress
