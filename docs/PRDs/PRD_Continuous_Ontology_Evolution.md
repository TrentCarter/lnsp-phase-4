# PRD: Continuous Ontology Evolution with Deduplication & Hierarchical Placement

**Status**: Draft
**Priority**: P0 (Critical Architecture)
**Owner**: System Architecture
**Created**: 2025-10-05
**Last Updated**: 2025-10-05

---

## Executive Summary

LNSP is not a static knowledge graph—it's a **living, evolving ontology** that continuously grows through multiple ingestion pathways. New concepts arrive from structured datasets, unstructured queries, and MAMBA vector outputs decoded via Vec2Text. Each new concept must be:

1. **Deduplicated** against existing knowledge (vector similarity)
2. **Placed hierarchically** within the ontology (parent/child/sibling relationships)
3. **Synchronized** across all storage layers (PostgreSQL, Neo4j, FAISS)

This PRD defines the architecture for maintaining ontological coherence as the knowledge graph scales from 4K to 173K+ concepts and beyond.

---

## Problem Statement

### Current State
- **Static ingestion**: Bulk loading of pre-structured ontology datasets (GO, SWO, DBpedia)
- **No deduplication**: Ingestion scripts assume unique data
- **No hierarchical placement**: Concepts added in flat batches
- **Single entry point**: Only handles structured JSONL files

### Gaps
1. **No duplicate detection**: Same concept could exist multiple times with different wording
2. **No ontological reasoning**: Can't determine WHERE a new concept belongs in the hierarchy
3. **No multi-source pipeline**: Can't handle:
   - Unstructured text queries → vecRAG → concept extraction
   - MAMBA outputs → Vec2Text → concept extraction
   - Real-time single-concept additions
4. **No incremental updates**: Must clear + re-ingest entire dataset

### Impact
- **Data integrity risk**: Duplicate concepts pollute the knowledge graph
- **Retrieval quality degradation**: Redundant entries reduce precision
- **Ontological chaos**: Flat additions break hierarchical reasoning
- **MAMBA integration blocked**: Can't safely add decoded vector outputs

---

## Goals & Non-Goals

### Goals
1. **Vector-based deduplication**: Identify duplicates using cosine similarity (threshold: 0.95+)
2. **Ontological placement**: Automatically find parent concepts and insert new concepts correctly
3. **Multi-source ingestion**: Support 4 entry pathways:
   - Bulk ontology imports (existing)
   - Unstructured dataset imports (new)
   - Single query additions (new)
   - MAMBA output integration (new)
4. **Atomic operations**: New concept addition is all-or-nothing (PostgreSQL + Neo4j + FAISS)
5. **Idempotent ingestion**: Re-running ingestion doesn't create duplicates

### Non-Goals
- **Automatic concept merging**: If duplicates found, reject (don't auto-merge)
- **LLM-based hierarchy inference**: Use vector similarity + graph structure only
- **Real-time FAISS updates**: Batch FAISS rebuilds acceptable (hourly/daily)
- **Conflict resolution UI**: CLI-only for v1

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  INGESTION SOURCES                          │
├─────────────────┬────────────────┬─────────────┬────────────┤
│ 1. Bulk Ontology│ 2. Unstructured│ 3. vecRAG   │ 4. MAMBA   │
│    (GO, SWO)    │    Text Files  │    Queries  │   Outputs  │
└────────┬────────┴────────┬───────┴──────┬──────┴──────┬─────┘
         │                 │              │             │
         └─────────────────┼──────────────┼─────────────┘
                          │              │
                 ┌────────▼──────────────▼──────────┐
                 │  UNIFIED INGESTION PIPELINE      │
                 │  - Parse/extract concepts        │
                 │  - Generate 768D GTR-T5 vectors  │
                 │  - Extract TMD (16D)             │
                 └────────┬─────────────────────────┘
                          │
                 ┌────────▼──────────────────────────┐
                 │  DEDUPLICATION ENGINE             │
                 │  - Query FAISS for top-10 matches │
                 │  - Cosine similarity threshold    │
                 │  - Return: NEW | DUPLICATE | UUIDs│
                 └────────┬─────────────────────────┘
                          │
                ┌─────────▼────────────────────────┐
                │  ONTOLOGICAL PLACEMENT ENGINE    │
                │  - Find parent concepts (vecRAG) │
                │  - Identify siblings (graph walk)│
                │  - Determine insertion point     │
                └─────────┬───────────────────────┘
                          │
              ┌───────────▼──────────────────────────┐
              │  ATOMIC COMMIT COORDINATOR           │
              │  - PostgreSQL: INSERT concept + vector│
              │  - Neo4j: CREATE node + RELATES_TO   │
              │  - FAISS: Mark dirty (batch rebuild) │
              │  - Rollback on ANY failure           │
              └──────────────────────────────────────┘
```

---

## Detailed Design

### 1. Deduplication Engine

**Input**:
- `concept_text` (str): New concept description
- `concept_vector` (768D float32): GTR-T5 embedding

**Algorithm**:
```python
def check_duplicate(concept_text, concept_vector):
    """
    Returns: (is_duplicate: bool, match_cpe_id: str | None, similarity: float)
    """
    # Query FAISS for top-10 nearest neighbors
    indices, distances = faiss_db.search(concept_vector, k=10)

    for idx, dist in zip(indices, distances):
        similarity = 1 - (dist / 2)  # Convert L2 to cosine

        if similarity >= DUPLICATE_THRESHOLD:  # 0.95
            existing_cpe_id = faiss_db.get_id(idx)
            existing_text = pg_db.get_concept_text(existing_cpe_id)

            return (True, existing_cpe_id, similarity)

    return (False, None, max_similarity)
```

**Configuration**:
- `DUPLICATE_THRESHOLD`: 0.95 (cosine similarity)
- `CANDIDATE_POOL`: 10 (top-k FAISS results)
- `FALLBACK_TEXT_CHECK`: Levenshtein distance for near-duplicates

**Edge Cases**:
- **Paraphrases**: "oxidoreductase activity" vs "oxidoreductase function" → Likely duplicate (0.98 similarity)
- **Specific vs General**: "histone H3K27me2 demethylase" vs "demethylase" → NOT duplicates (0.75 similarity)
- **Typos**: "moleular function" vs "molecular function" → Catch via text similarity fallback

---

### 2. Ontological Placement Engine

**Goal**: Find the correct parent concept(s) for a new concept in the hierarchy.

**Input**:
- `concept_text`: New concept
- `concept_vector`: 768D embedding

**Algorithm**:
```python
def find_placement(concept_text, concept_vector):
    """
    Returns: (parent_cpe_ids: List[str], confidence: float)
    """
    # Step 1: vecRAG search for semantically similar concepts
    top_k_results = faiss_db.search(concept_vector, k=20)

    candidates = []
    for idx in top_k_results:
        cpe_id = faiss_db.get_id(idx)

        # Step 2: For each candidate, check Neo4j graph structure
        # Is this concept MORE GENERAL (parent) or SAME LEVEL (sibling)?
        generality_score = compute_generality(concept_text, cpe_id)

        if generality_score > 0.7:  # Likely a parent
            candidates.append((cpe_id, generality_score))

    # Step 3: Rank candidates by:
    # - Vector similarity (is it semantically related?)
    # - Generality score (is it more abstract?)
    # - Graph position (how many children does it have?)

    parents = rank_candidates(candidates)
    return parents[:3], confidence  # Top 3 parents
```

**Generality Scoring**:
```python
def compute_generality(new_concept, existing_cpe_id):
    """
    Heuristics to determine if existing concept is MORE GENERAL than new one.
    """
    existing_text = pg_db.get_concept_text(existing_cpe_id)

    # Heuristic 1: Substring containment
    if new_concept.startswith(existing_text):
        return 0.9  # "histone demethylase" contains "demethylase"

    # Heuristic 2: Word count (more specific = more words)
    new_words = len(new_concept.split())
    existing_words = len(existing_text.split())
    if new_words > existing_words:
        return 0.7  # New concept is more specific

    # Heuristic 3: Neo4j graph depth
    # If existing concept has many children, it's likely a parent
    child_count = neo4j_db.count_children(existing_cpe_id)
    if child_count > 5:
        return 0.8

    return 0.5  # Unsure
```

**Fallback Strategy**:
- If no clear parent found (confidence < 0.6): Create as **root-level concept**
- User can manually assign parent later via CLI

---

### 3. Multi-Source Ingestion Pipeline

#### Source 1: Bulk Ontology Import (Existing)
```bash
# Current workflow - already implemented
./scripts/ingest_ontologies.sh
```
**Changes**: Add deduplication check before each insert

#### Source 2: Unstructured Text Files
```bash
# New workflow
./scripts/ingest_unstructured.sh --input data/research_papers.txt
```
**Pipeline**:
1. Parse text → Extract candidate concepts (NER or LLM)
2. Generate vectors (GTR-T5)
3. Deduplicate
4. Find placement
5. Atomic commit

#### Source 3: Single Query Addition
```bash
# New CLI command
lnsp add-concept "histone H3K27me3 demethylase activity"
```
**Pipeline**:
1. Generate vector
2. Deduplicate
3. Find placement
4. Atomic commit
5. Return: `Concept added with ID: abc123` or `Duplicate found: xyz789`

#### Source 4: MAMBA Output Integration
```python
# Workflow
mamba_vector = mamba_model.generate(prompt)  # 768D output
decoded_text = vec2text.decode(mamba_vector)  # "oxidoreductase activity"

# Add to RAG
result = lnsp.add_concept(
    text=decoded_text,
    vector=mamba_vector,
    source="mamba_output",
    metadata={"prompt": prompt}
)
```

---

### 4. Atomic Commit Coordinator

**Transaction Flow**:
```python
def atomic_add_concept(concept_text, concept_vector, parent_cpe_ids):
    """
    All-or-nothing insertion across 3 storage layers.
    """
    pg_transaction = None
    neo4j_transaction = None

    try:
        # Step 1: PostgreSQL
        pg_transaction = pg_db.begin()
        cpe_id = pg_db.insert_concept(concept_text, concept_vector)
        pg_transaction.commit()

        # Step 2: Neo4j
        neo4j_transaction = neo4j_db.begin()
        neo4j_db.create_node(cpe_id, concept_text, concept_vector)
        for parent_id in parent_cpe_ids:
            neo4j_db.create_edge(parent_id, cpe_id, "RELATES_TO")
        neo4j_transaction.commit()

        # Step 3: FAISS (mark dirty, rebuild later)
        faiss_db.mark_dirty()

        return cpe_id

    except Exception as e:
        # Rollback everything
        if pg_transaction:
            pg_transaction.rollback()
        if neo4j_transaction:
            neo4j_transaction.rollback()

        raise OntologyInsertionError(f"Failed to add concept: {e}")
```

**FAISS Batch Rebuild**:
```bash
# Cron job: Every hour or on-demand
./scripts/rebuild_faiss_index.sh
```
- Reads all vectors from PostgreSQL
- Rebuilds IVF index
- Atomically swaps old → new index

---

## Data Model Changes

### PostgreSQL Schema Updates

**New Table: `concept_relationships`**
```sql
CREATE TABLE concept_relationships (
    id SERIAL PRIMARY KEY,
    parent_cpe_id UUID NOT NULL REFERENCES cpe_entry(cpe_id),
    child_cpe_id UUID NOT NULL REFERENCES cpe_entry(cpe_id),
    relationship_type VARCHAR(50) DEFAULT 'RELATES_TO',
    confidence FLOAT,  -- How confident are we in this relationship?
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(parent_cpe_id, child_cpe_id)
);
```

**New Column in `cpe_entry`**:
```sql
ALTER TABLE cpe_entry ADD COLUMN ingestion_source VARCHAR(50);
-- Values: 'ontology-go', 'ontology-swo', 'unstructured', 'query', 'mamba'

ALTER TABLE cpe_entry ADD COLUMN is_root BOOLEAN DEFAULT FALSE;
-- TRUE if this concept has no parent (top-level)
```

### Neo4j Schema Updates

**Node Properties**:
```cypher
// Existing
(:Concept {
    cpe_id: UUID,
    name: String,
    vector: [Float]  // 768D
})

// Add
(:Concept {
    ingestion_source: String,
    is_root: Boolean,
    created_at: Timestamp
})
```

**Edge Types**:
```cypher
// Existing
()-[:RELATES_TO]->()

// Add metadata
()-[:RELATES_TO {
    confidence: Float,
    inferred: Boolean  // TRUE if auto-placed, FALSE if from ontology
}]->()
```

---

## API Design

### CLI Commands

```bash
# Add single concept
lnsp add-concept "histone demethylase activity" \
    --source mamba \
    --parent "demethylase activity"

# Check for duplicates (dry-run)
lnsp check-duplicate "molecular function"

# Find placement (without inserting)
lnsp find-placement "histone demethylase"

# Rebuild FAISS index
lnsp rebuild-index --confirm

# Show concept hierarchy
lnsp show-hierarchy "oxidoreductase activity" --depth 3
```

### Python API

```python
from src.ontology_evolution import OntologyManager

om = OntologyManager()

# Add concept
result = om.add_concept(
    text="histone H3K27me3 demethylase",
    vector=gtr_t5_vector,
    source="mamba_output",
    metadata={"mamba_prompt": "..."}
)

if result.is_duplicate:
    print(f"Duplicate of {result.existing_cpe_id} (similarity: {result.similarity:.3f})")
else:
    print(f"Added: {result.cpe_id}")
    print(f"Parents: {result.parent_cpe_ids}")
    print(f"Confidence: {result.placement_confidence:.2f}")
```

---

## Implementation Plan

### Phase 1: Deduplication Engine (Week 1)
- [ ] Implement `check_duplicate()` function
- [ ] Add FAISS k-NN search with configurable threshold
- [ ] Add text similarity fallback (Levenshtein)
- [ ] Unit tests: 100 duplicate pairs, 100 unique pairs

### Phase 2: Ontological Placement (Week 2)
- [ ] Implement `find_placement()` function
- [ ] Generality scoring heuristics
- [ ] Neo4j graph traversal for hierarchy analysis
- [ ] Integration tests: 50 GO concepts with known parents

### Phase 3: Atomic Commit Coordinator (Week 3)
- [ ] PostgreSQL transaction wrapper
- [ ] Neo4j transaction wrapper
- [ ] Rollback logic on failure
- [ ] FAISS dirty-flag + batch rebuild

### Phase 4: Multi-Source Pipelines (Week 4)
- [ ] Unstructured text ingestion
- [ ] Single-query CLI command
- [ ] MAMBA output integration
- [ ] End-to-end tests for all 4 sources

### Phase 5: Production Hardening (Week 5)
- [ ] Logging + metrics (Prometheus)
- [ ] Error recovery (dead-letter queue for failures)
- [ ] Performance optimization (batch inserts)
- [ ] Documentation + runbook

---

## Metrics & Monitoring

### Success Metrics
- **Deduplication accuracy**: >95% precision, >90% recall
- **Placement accuracy**: >80% agreement with human-labeled test set
- **Throughput**: 100 concepts/sec for bulk imports
- **Latency**: <500ms for single-concept addition

### Dashboards
```
┌─────────────────────────────────────────┐
│ Ontology Evolution Metrics              │
├─────────────────────────────────────────┤
│ Total Concepts:        173,529          │
│ Additions Today:       1,247            │
│ Duplicates Blocked:    89               │
│ FAISS Index Age:       2h 15m           │
│ Failed Commits:        3 (retry queue)  │
└─────────────────────────────────────────┘
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False duplicate detection | Lost valid concepts | Adjustable threshold + manual override |
| Wrong parent placement | Broken hierarchy | Confidence scores + manual review queue |
| FAISS index drift | Stale search results | Hourly rebuilds + dirty-flag system |
| Transaction deadlocks | Ingestion failures | Retry logic + exponential backoff |
| Vector distribution shift | Dedup breaks over time | Periodic re-calibration of thresholds |

---

## Open Questions

1. **How to handle concept updates?** (e.g., "molecular function" definition changes)
   - **Proposal**: Versioning system (`concept_version` table)

2. **Multi-parent concepts?** (e.g., "histone demethylase" is both "histone enzyme" AND "demethylase")
   - **Proposal**: Allow multiple `parent_cpe_ids` in placement engine

3. **Concept deletion?** (e.g., deprecated ontology terms)
   - **Proposal**: Soft delete (mark `is_deleted=TRUE`, keep in DB for history)

4. **How to prioritize placement when multiple parents are equally valid?**
   - **Proposal**: Use graph centrality (prefer more-connected parents)

---

## References

- **Existing Code**:
  - `src/ingest_ontology_simple.py` (current bulk ingestion)
  - `src/db_faiss.py` (FAISS operations)
  - `src/db_neo4j.py` (Neo4j operations)

- **Related PRDs**:
  - `PRD_GraphRAG_LightRAG_Architecture.md` (graph reasoning)
  - `PRD_KnownGood_vecRAG_Data_Ingestion.md` (baseline ingestion)

- **Academic Papers**:
  - "Ontology Evolution" (Stojanovic et al., 2002)
  - "Hierarchical Concept Discovery" (Navigli et al., 2011)

---

## Appendix A: Example Workflows

### Workflow 1: MAMBA Output → RAG
```python
# User query to MAMBA
user_query = "What enzyme removes methyl groups from histones?"

# MAMBA generates answer as 768D vector
mamba_output_vector = mamba.generate(user_query)

# Vec2Text decodes to natural language
decoded_text = vec2text.decode(mamba_output_vector, backend="jxe", steps=5)
# Output: "histone demethylase activity"

# Check if this concept exists
om = OntologyManager()
result = om.add_concept(
    text=decoded_text,
    vector=mamba_output_vector,
    source="mamba_output"
)

if result.is_duplicate:
    # Concept already exists - return to user
    existing_concept = pg_db.get_concept(result.existing_cpe_id)
    return existing_concept
else:
    # New concept - show user and add to RAG
    print(f"New knowledge discovered: {decoded_text}")
    print(f"Parents: {result.parent_names}")
    return result
```

### Workflow 2: Research Paper Ingestion
```bash
# Extract concepts from scientific paper
./scripts/extract_concepts.py --input paper.pdf --output concepts.jsonl

# Ingest with deduplication
./scripts/ingest_unstructured.sh --input concepts.jsonl --deduplicate
```

---

**End of PRD**
