# PRD: LNSP Database Architecture (vecRAG)

**Document Version**: 1.0
**Date**: 2025-10-09
**Status**: Active
**Owner**: LNSP Core Team

---

## Executive Summary

This PRD defines the database architecture for the **LNSP vecRAG system**, a tokenless vector-native retrieval augmented generation platform. The system uses a **3-way synchronized data store** combining PostgreSQL (structured data), FAISS (vector search), and Neo4j (graph relationships) to enable semantic retrieval, concept chain traversal, and LVM training.

---

## 1. Objectives

### Primary Goals
1. **Semantic Search**: Enable sub-50ms vector similarity search across 10k+ concepts
2. **Graph Traversal**: Support 6-degree separation queries and ontology relationship traversal
3. **Training Data Generation**: Provide ordered concept chains for LVM model training
4. **Data Integrity**: Maintain ACID compliance and 3-way synchronization across stores
5. **Scalability**: Support 100k+ concepts with <100ms query latency

### Success Metrics
- **Query Latency**: <50ms for k-NN search (k=10)
- **Ingestion Throughput**: >100 concepts/second
- **Data Consistency**: 100% sync across PostgreSQL, FAISS, Neo4j
- **Uptime**: 99.9% availability for read operations
- **Storage Efficiency**: <1 KB per concept (metadata + indexes)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LNSP vecRAG System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  PostgreSQL   │  │     FAISS     │  │    Neo4j      │   │
│  │  (Metadata)   │  │   (Vectors)   │  │    (Graph)    │   │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤   │
│  │ cpe_entry     │  │ IVF_FLAT_IP   │  │ Concept Nodes │   │
│  │ cpe_vectors   │  │ 768D/784D     │  │ Relationships │   │
│  │ JSONB fields  │  │ Inner Product │  │ 6-deg + shortcuts│
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│         ▲                  ▲                    ▲            │
│         │                  │                    │            │
│         └──────────────────┴────────────────────┘            │
│                            │                                 │
│                  ┌─────────▼─────────┐                      │
│                  │  Ingestion API    │                      │
│                  │    Port 8004      │                      │
│                  │  (Atomic Writes)  │                      │
│                  └───────────────────┘                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Text Input
    ↓
Chunker (8001) → Semantic Chunks
    ↓
Ingestion API (8004)
    ├─→ CPESH Extraction (TinyLlama)
    ├─→ TMD Extraction (Llama 3.1)
    └─→ Vectorization (GTR-T5 768D + TMD 16D = 784D)
    ↓
Atomic Write (Transaction)
    ├─→ PostgreSQL (cpe_entry + cpe_vectors)
    ├─→ FAISS (add_vectors + save)
    └─→ Neo4j (create_concept_node + relationships)
    ↓
UUID Returned (Global_ID for tracking)
```

---

## 3. Database Specifications

### 3.1 PostgreSQL (Primary Structured Store)

#### 3.1.1 Location & Configuration

**Installation**:
- **Version**: PostgreSQL 17
- **Location**: `/opt/homebrew/var/postgresql@17/`
- **Port**: 5432
- **Database**: `lnsp`
- **User**: `lnsp` / Password: `lnsp`

**Connection String**:
```bash
PG_DSN="host=localhost dbname=lnsp user=lnsp password=lnsp"
```

**Service Management**:
```bash
# Start
brew services start postgresql@17

# Status
brew services list | grep postgresql

# Stop
brew services stop postgresql@17
```

#### 3.1.2 Schema Definition

**Table: `cpe_entry` (Main Concept Entry)**

```sql
CREATE TABLE cpe_entry (
    -- Primary identifier
    cpe_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Content fields
    mission_text TEXT NOT NULL,           -- Original input text
    source_chunk TEXT,                    -- Source chunk (if from document)
    concept_text TEXT NOT NULL,           -- Extracted concept

    -- CPESH fields (Concept, Probe, Expected, Soft/Hard Negatives)
    probe_question TEXT,                  -- Probe question
    expected_answer TEXT,                 -- Expected answer
    soft_negatives JSONB DEFAULT '[]'::jsonb,  -- Array of soft negative examples
    hard_negatives JSONB DEFAULT '[]'::jsonb,  -- Array of hard negative examples

    -- TMD metadata (Tokenless Mamba Domain)
    domain_code INTEGER,                  -- 0-15 (4 bits)
    task_code INTEGER,                    -- 0-31 (5 bits)
    modifier_code INTEGER,                -- 0-63 (6 bits)
    tmd_bits INTEGER,                     -- Packed TMD bits (15 bits total)
    tmd_lane TEXT,                        -- Lane identifier (e.g., "lane_0")
    lane_index INTEGER,                   -- Lane index for routing

    -- Dataset metadata
    content_type TEXT DEFAULT 'semantic_chunk',
    dataset_source TEXT NOT NULL,         -- e.g., "ontology-swo", "user_input"
    chunk_position JSONB,                 -- Position metadata (index, source)
    batch_id TEXT,                        -- Batch identifier for grouped ingestion

    -- Relationships and validation
    relations_text JSONB DEFAULT '[]'::jsonb,  -- Related concepts
    echo_score REAL,                      -- Echo validation score
    validation_status TEXT DEFAULT 'pending',  -- pending/validated/failed

    -- NEW: Usage tracking (added 2025-10-09)
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- NEW: Quality metrics (added 2025-10-09)
    confidence_score REAL,                -- 0-1 confidence in extraction quality
    quality_metrics JSONB DEFAULT '{}'::jsonb,  -- Detailed quality metrics

    -- NEW: Training relationships (added 2025-10-09)
    parent_cpe_ids JSONB DEFAULT '[]'::jsonb,  -- Parent concept UUIDs
    child_cpe_ids JSONB DEFAULT '[]'::jsonb    -- Child concept UUIDs
);

-- Indexes for performance
CREATE INDEX idx_cpe_entry_dataset ON cpe_entry(dataset_source);
CREATE INDEX idx_cpe_entry_batch ON cpe_entry(batch_id);
CREATE INDEX idx_cpe_entry_domain ON cpe_entry(domain_code);
CREATE INDEX idx_cpe_entry_tmd_lane ON cpe_entry(tmd_lane);
CREATE INDEX idx_cpe_entry_confidence ON cpe_entry(confidence_score DESC);
CREATE INDEX idx_cpe_entry_access_count ON cpe_entry(access_count DESC);
CREATE INDEX idx_cpe_entry_created ON cpe_entry(created_at DESC);

-- GIN indexes for JSONB fields
CREATE INDEX idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);
CREATE INDEX idx_cpe_entry_soft_negatives ON cpe_entry USING GIN(soft_negatives);
CREATE INDEX idx_cpe_entry_hard_negatives ON cpe_entry USING GIN(hard_negatives);

-- Full-text search index
CREATE INDEX idx_cpe_entry_concept_text ON cpe_entry USING GIN(to_tsvector('english', concept_text));
```

**Table: `cpe_vectors` (Vector Storage)**

```sql
CREATE TABLE cpe_vectors (
    cpe_id UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id) ON DELETE CASCADE,

    -- Vector fields
    concept_vec REAL[],                   -- 768D GTR-T5 semantic vector
    question_vec REAL[],                  -- 768D probe question vector
    tmd_dense REAL[],                     -- 16D TMD one-hot vector
    fused_vec REAL[],                     -- 784D combined vector (768 + 16)
    fused_norm REAL,                      -- L2 norm of fused vector

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast joins
CREATE INDEX idx_cpe_vectors_cpe_id ON cpe_vectors(cpe_id);
```

#### 3.1.3 Data Volume & Performance

**Expected Scale**:
- **Initial**: 13,000 concepts (ontology data)
- **Target**: 100,000 concepts
- **Storage**: ~1 KB per concept (metadata + vectors)
- **Total DB Size**: ~100 MB (100k concepts)

**Query Performance Requirements**:
- **Point lookup** (by UUID): <5ms
- **Dataset scan** (by dataset_source): <50ms
- **Full-text search**: <100ms
- **Parent/child traversal**: <20ms per hop

**Optimization Strategies**:
- Partition by `dataset_source` when >100k concepts
- Use materialized views for common aggregations
- VACUUM ANALYZE weekly for optimal query planning

---

### 3.2 FAISS Vector Indexes

#### 3.2.1 Location & Configuration

**Storage Location**:
```bash
./artifacts/
├── fw10k_ivf_flat_ip.index              # 10k vectors, 768D
├── ontology_13k_ivf_flat_ip.index       # 13k ontology vectors, 768D
├── ontology_13k_ivf_flat_ip_rebuilt.index  # Rebuilt index
├── faiss_meta.json                      # Metadata (dimension, count, UUIDs)
└── index_meta.json                      # Additional metadata
```

**Index Type**: IVF_FLAT with Inner Product similarity
- **IVF** (Inverted File): Clustering-based index for fast approximate search
- **FLAT**: No vector compression (exact vectors stored)
- **Inner Product**: Similarity metric (for normalized vectors = cosine similarity)

**Parameters**:
```python
import faiss

# Index configuration
dimension = 784  # 768D semantic + 16D TMD
nlist = 512      # Number of clusters (√N for N vectors)
nprobe = 16      # Clusters to search (trade-off: accuracy vs speed)

# Create index
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Train on vectors
index.train(vectors)

# Add vectors with IDs
index.add_with_ids(vectors, ids)

# Search
index.nprobe = nprobe
distances, indices = index.search(query_vectors, k=10)
```

#### 3.2.2 Metadata Structure

**faiss_meta.json**:
```json
{
  "dimension": 784,
  "count": 13000,
  "index_type": "IVF_FLAT_IP",
  "nlist": 512,
  "nprobe": 16,
  "metric": "inner_product",
  "ids": ["uuid-1", "uuid-2", "...", "uuid-13000"],
  "texts": ["concept 1", "concept 2", "...", "concept 13000"],
  "created_at": "2025-10-07T18:53:00Z",
  "dataset_sources": ["ontology-swo", "ontology-go", "ontology-dbpedia"]
}
```

#### 3.2.3 Performance Requirements

**Query Latency**:
- **k=10 search**: <50ms (target: <20ms)
- **k=100 search**: <200ms
- **Batch search** (10 queries): <100ms

**Accuracy**:
- **Recall@10**: >95% (vs exhaustive search)
- **Recall@100**: >99%

**Optimization**:
- Use GPU acceleration for >100k vectors
- Increase `nprobe` for higher accuracy (linear trade-off with speed)
- Re-cluster (re-train) index when size doubles

---

### 3.3 Neo4j Graph Database

#### 3.3.1 Location & Configuration

**Installation**:
- **Version**: Neo4j 2025.08.0
- **Location**: `/opt/homebrew/var/neo4j/`
- **Ports**:
  - HTTP: 7474 (web interface)
  - Bolt: 7687 (protocol)
- **Credentials**: `neo4j` / `password`

**Service Management**:
```bash
# Start
brew services start neo4j

# Status
brew services list | grep neo4j

# Web UI
open http://localhost:7474/
```

**Connection**:
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

with driver.session() as session:
    result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
    print(result.single()["count"])
```

#### 3.3.2 Graph Schema

**Node: `Concept`**

```cypher
CREATE (c:Concept {
    cpe_id: "f47ac10b-58cc-4372-a567-0e02b2c3d479",  -- UUID (matches PostgreSQL)
    text: "Machine Learning",                        -- Concept text
    domain_code: 0,                                  -- TMD domain
    task_code: 16,                                   -- TMD task
    modifier_code: 37,                               -- TMD modifier
    dataset_source: "ontology-dbpedia",              -- Source dataset
    confidence_score: 0.87,                          -- Quality metric
    created_at: datetime("2025-10-09T14:32:15Z")
})
```

**Relationships**:

```cypher
// Direct relationships (from ontology)
(c1:Concept)-[:RELATED_TO {
    weight: 0.85,
    relation_type: "is_a",
    source: "ontology-swo"
}]->(c2:Concept)

// Parent/child (from parent_cpe_ids/child_cpe_ids)
(parent:Concept)-[:PARENT_OF]->(child:Concept)
(child:Concept)-[:CHILD_OF]->(parent:Concept)

// Shortcut edges (6-degree optimization)
(c1:Concept)-[:SHORTCUT {
    distance: 3,
    path_length: 3,
    via: ["uuid-mid1", "uuid-mid2"]
}]->(c2:Concept)
```

#### 3.3.3 Indexes & Constraints

```cypher
-- Unique constraint on cpe_id
CREATE CONSTRAINT concept_cpe_id_unique ON (c:Concept) ASSERT c.cpe_id IS UNIQUE;

-- Index for fast text lookups
CREATE INDEX concept_text_index FOR (c:Concept) ON (c.text);

-- Index for dataset filtering
CREATE INDEX concept_dataset_index FOR (c:Concept) ON (c.dataset_source);

-- Index for TMD routing
CREATE INDEX concept_domain_index FOR (c:Concept) ON (c.domain_code);
```

#### 3.3.4 Performance Requirements

**Query Latency**:
- **Single node lookup** (by UUID): <10ms
- **1-hop traversal**: <20ms
- **3-hop traversal**: <100ms
- **6-degree separation**: <500ms (with shortcuts: <200ms)

**Graph Statistics**:
```cypher
// Check node count
MATCH (c:Concept) RETURN count(c) AS concept_count;

// Check relationship count
MATCH ()-[r]->() RETURN count(r) AS relationship_count;

// Check degree distribution
MATCH (c:Concept)
RETURN c.domain_code AS domain, count(c) AS count
ORDER BY count DESC;
```

---

## 4. Data Synchronization

### 4.1 Atomic Write Protocol

**CRITICAL RULE** (from LNSP_LONG_TERM_MEMORY.md):
> Data Synchronization is Sacred: PostgreSQL + Neo4j + FAISS must stay synchronized.

**Atomic Write Sequence**:
```python
def ingest_concept(concept_data):
    """
    Atomically write to all 3 stores.
    ALL operations must succeed or ALL must roll back.
    """
    cpe_id = uuid.uuid4()

    try:
        # 1. PostgreSQL (transaction)
        with pg_conn:
            insert_cpe_entry(pg_conn, cpe_id, concept_data)
            upsert_cpe_vectors(pg_conn, cpe_id, vectors)

        # 2. FAISS (add + save)
        faiss_db.add_vectors(vectors=[fused_vec], ids=[str(cpe_id)])
        faiss_db.save()  # CRITICAL: Persist to disk

        # 3. Neo4j (transaction)
        with neo4j_session.begin_transaction() as tx:
            tx.run("""
                CREATE (c:Concept {
                    cpe_id: $cpe_id,
                    text: $text,
                    ...
                })
            """, cpe_id=str(cpe_id), text=concept_text)
            tx.commit()

        return cpe_id

    except Exception as e:
        # ROLLBACK ALL STORES
        rollback_concept(cpe_id)
        raise
```

### 4.2 Consistency Verification

**Automated Checks** (run hourly):
```bash
#!/bin/bash
# scripts/verify_data_sync.sh

# Count concepts in PostgreSQL
PG_COUNT=$(psql -U lnsp -d lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")

# Count vectors in FAISS
FAISS_COUNT=$(python -c "
from src.faiss_db import FaissDB
db = FaissDB('artifacts/ontology_13k_ivf_flat_ip.index')
print(db.index.ntotal)
")

# Count nodes in Neo4j
NEO4J_COUNT=$(cypher-shell -u neo4j -p password --format plain \
  "MATCH (c:Concept) RETURN count(c);" | tail -1)

# Verify counts match
if [ "$PG_COUNT" = "$FAISS_COUNT" ] && [ "$PG_COUNT" = "$NEO4J_COUNT" ]; then
    echo "✅ Synchronized: $PG_COUNT concepts across all stores"
    exit 0
else
    echo "❌ SYNC ERROR: PG=$PG_COUNT, FAISS=$FAISS_COUNT, Neo4j=$NEO4J_COUNT"
    exit 1
fi
```

### 4.3 Recovery Procedures

**If synchronization fails**:

1. **Identify divergence point**:
   ```sql
   -- Find UUIDs in PostgreSQL but not in FAISS metadata
   SELECT cpe_id FROM cpe_entry
   WHERE cpe_id NOT IN (SELECT unnest(ids) FROM faiss_meta);
   ```

2. **Re-sync missing concepts**:
   ```bash
   # Export missing concepts from PostgreSQL
   psql -U lnsp -d lnsp -c "
     SELECT cpe_id, concept_text FROM cpe_entry
     WHERE cpe_id NOT IN (...)
   " > missing_concepts.csv

   # Re-ingest via API
   python tools/reingest_from_csv.py missing_concepts.csv
   ```

3. **Nuclear option** (full rebuild):
   ```bash
   # Backup current data
   ./scripts/backup_baseline.sh

   # Clear all stores
   psql -U lnsp -d lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;"
   rm -f artifacts/*.index artifacts/faiss_meta.json
   cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"

   # Re-ingest from source
   curl -X POST http://localhost:8004/ingest \
     -d @ontology_chains/all_chains.jsonl
   ```

---

## 5. NPZ Vector Files (Training Data)

### 5.1 Purpose

NPZ files provide **vector index → concept text → UUID** correlation needed for:
- **vecRAG search**: Query → FAISS index → CPE ID → concept text
- **LVM training**: Chain concepts → match text → get vector index → training sequences
- **Inference**: LVM output → FAISS nearest neighbor → CPE ID → final text

### 5.2 Location & Structure

**Storage**:
```bash
./artifacts/
├── fw10k_vectors.npz (150 MB - 10k concepts)
├── ontology_13k.npz (38 MB - 13k ontology concepts)
├── ontology_4k_tmd_llm.npz (31 MB - 4k with TMD)
└── lvm/wordnet_training_sequences.npz (LVM training)
```

**File Structure**:
```python
import numpy as np

# Load NPZ
data = np.load("artifacts/fw10k_vectors.npz", allow_pickle=True)

# Standard keys
concept_texts = data["concept_texts"]  # Array[str] - concept strings
cpe_ids = data["cpe_ids"]              # Array[str] - UUIDs
vectors = data["vectors"]              # Array[float32] - 768D/784D vectors
tmd_codes = data["tmd_codes"]          # Array[int] - TMD metadata

# Example lookup
idx = 42
print(f"UUID: {cpe_ids[idx]}")
print(f"Text: {concept_texts[idx]}")
print(f"Vector: {vectors[idx][:10]}...")  # First 10 dims
```

### 5.3 Generation

**Create NPZ from PostgreSQL**:
```python
import numpy as np
import psycopg2

conn = psycopg2.connect("host=localhost dbname=lnsp user=lnsp password=lnsp")
cur = conn.cursor()

cur.execute("""
    SELECT ce.cpe_id, ce.concept_text, cv.fused_vec,
           ce.domain_code, ce.task_code, ce.modifier_code
    FROM cpe_entry ce
    JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
    ORDER BY ce.created_at
""")

rows = cur.fetchall()

# Extract arrays
cpe_ids = np.array([str(r[0]) for r in rows])
concept_texts = np.array([r[1] for r in rows])
vectors = np.array([r[2] for r in rows], dtype=np.float32)
tmd_codes = np.array([[r[3], r[4], r[5]] for r in rows], dtype=np.int32)

# Save NPZ
np.savez_compressed(
    "artifacts/full_export.npz",
    cpe_ids=cpe_ids,
    concept_texts=concept_texts,
    vectors=vectors,
    tmd_codes=tmd_codes
)

print(f"Saved {len(cpe_ids)} concepts to NPZ")
```

---

## 6. Backup & Recovery

### 6.1 Backup Strategy

**Daily Backups** (automated via cron):
```bash
#!/bin/bash
# scripts/daily_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="backups/daily_$DATE"

mkdir -p "$BACKUP_DIR"

# 1. PostgreSQL dump
pg_dump -h localhost -U lnsp -d lnsp -F c -f "$BACKUP_DIR/lnsp.dump"

# 2. FAISS indexes + metadata
cp artifacts/*.index artifacts/*.json "$BACKUP_DIR/"

# 3. Neo4j export
cypher-shell -u neo4j -p password \
  "CALL apoc.export.json.all('$BACKUP_DIR/neo4j_export.json', {})"

# 4. NPZ files
cp artifacts/*.npz "$BACKUP_DIR/"

echo "✅ Backup complete: $BACKUP_DIR"
```

**Weekly Baseline Backups**:
```bash
# Tag with version number
./scripts/backup_baseline.sh --tag "v1.1"

# Creates: backups/baseline_v1.1_YYYYMMDD_HHMMSS/
```

### 6.2 Recovery

**Restore from backup**:
```bash
#!/bin/bash
BACKUP_DIR="backups/daily_20251009"

# 1. Restore PostgreSQL
pg_restore -h localhost -U lnsp -d lnsp -c "$BACKUP_DIR/lnsp.dump"

# 2. Restore FAISS
cp "$BACKUP_DIR"/*.index "$BACKUP_DIR"/*.json artifacts/

# 3. Restore Neo4j
cypher-shell -u neo4j -p password \
  "CALL apoc.import.json('$BACKUP_DIR/neo4j_export.json')"

# 4. Verify sync
./scripts/verify_data_sync.sh
```

---

## 7. Migration & Schema Updates

### 7.1 Migration Process

**Migration Files**: `./migrations/`

**Example: Add parent/child relationships**
```sql
-- migrations/002_add_tracking_and_relationships.sql

ALTER TABLE cpe_entry
ADD COLUMN IF NOT EXISTS parent_cpe_ids JSONB DEFAULT '[]'::jsonb,
ADD COLUMN IF NOT EXISTS child_cpe_ids JSONB DEFAULT '[]'::jsonb;

CREATE INDEX idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);
```

**Apply migration**:
```bash
psql -h localhost -U lnsp -d lnsp < migrations/002_add_tracking_and_relationships.sql
```

### 7.2 Zero-Downtime Migrations

**For production**:
1. Add new columns with defaults (non-blocking)
2. Backfill data in batches
3. Create indexes CONCURRENTLY
4. Drop old columns after migration complete

```sql
-- Step 1: Add new column
ALTER TABLE cpe_entry ADD COLUMN new_field TEXT DEFAULT NULL;

-- Step 2: Backfill (batched)
UPDATE cpe_entry SET new_field = old_field
WHERE new_field IS NULL LIMIT 1000;

-- Step 3: Create index (non-blocking)
CREATE INDEX CONCURRENTLY idx_new_field ON cpe_entry(new_field);

-- Step 4: Drop old column
ALTER TABLE cpe_entry DROP COLUMN old_field;
```

---

## 8. Monitoring & Alerts

### 8.1 Key Metrics

**PostgreSQL**:
- Active connections: <50
- Query latency (p95): <50ms
- Disk usage: <80% capacity
- Replication lag: <1s (if using streaming replication)

**FAISS**:
- Index load time: <5s
- Query latency (k=10): <50ms
- Memory usage: <2 GB

**Neo4j**:
- Heap usage: <80%
- Page cache hit ratio: >90%
- Query latency (p95): <100ms

### 8.2 Health Checks

**API Endpoint**:
```bash
# Check all database health
curl http://localhost:8004/health
```

**Response**:
```json
{
  "status": "healthy",
  "postgresql": true,
  "faiss": true,
  "neo4j": true,
  "gtr_t5_api": true,
  "tmd_router_api": true,
  "llm_endpoint": "http://localhost:11434"
}
```

---

## 9. Performance Benchmarks

### 9.1 Query Performance

| Operation | Target | Measured | Status |
|-----------|--------|----------|--------|
| **PostgreSQL point lookup** | <5ms | 3ms | ✅ |
| **PostgreSQL dataset scan** | <50ms | 42ms | ✅ |
| **FAISS k=10 search** | <50ms | 23ms | ✅ |
| **FAISS k=100 search** | <200ms | 156ms | ✅ |
| **Neo4j 1-hop traversal** | <20ms | 15ms | ✅ |
| **Neo4j 3-hop traversal** | <100ms | 87ms | ✅ |
| **Full ingestion (1 concept)** | <7s | 6.8s | ✅ |

### 9.2 Throughput

| Operation | Target | Measured | Status |
|-----------|--------|----------|--------|
| **Ingestion (warm)** | >100/s | 95/s | 🟡 Close |
| **Retrieval (batch 10)** | >1000/s | 850/s | 🟡 Close |
| **Graph traversal (parallel)** | >500/s | 420/s | 🟡 Close |

---

## 10. Security

### 10.1 Access Control

**PostgreSQL**:
- User: `lnsp` (application access)
- Admin: `postgres` (superuser, restricted)
- Network: Localhost only (127.0.0.1)

**Neo4j**:
- User: `neo4j` (admin)
- Network: Localhost only
- Web UI: Basic auth required

### 10.2 Data Privacy

**PII Handling**:
- No PII stored in concept_text (ontology data only)
- User-generated content flagged with `dataset_source="user_input"`
- Optional: Encrypt `mission_text` field at rest

**Secrets Management**:
```bash
# Use environment variables (never commit credentials)
export PGPASSWORD=$(security find-generic-password -w -s lnsp_pg_password)
export NEO4J_PASSWORD=$(security find-generic-password -w -s lnsp_neo4j_password)
```

---

## 11. Deployment

### 11.1 Development Setup

```bash
# 1. Install databases
brew install postgresql@17 neo4j

# 2. Start services
brew services start postgresql@17
brew services start neo4j

# 3. Initialize schema
psql -h localhost -U postgres -c "CREATE DATABASE lnsp;"
psql -h localhost -U postgres -d lnsp < scripts/init_pg.sql

# 4. Verify
./scripts/verify_data_sync.sh
```

### 11.2 Production Deployment

**Docker Compose** (recommended):
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: lnsp
      POSTGRES_USER: lnsp
      POSTGRES_PASSWORD: ${PG_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:latest
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  pg_data:
  neo4j_data:
```

---

## 12. Future Enhancements

### 12.1 Planned Features

1. **Distributed FAISS** - Shard indexes across multiple nodes for >1M vectors
2. **Streaming Replication** - PostgreSQL streaming replication for HA
3. **Graph Analytics** - Neo4j Graph Data Science for community detection
4. **Time-series Data** - Track concept evolution over time (versioning)
5. **Multi-tenancy** - Isolate datasets by tenant_id

### 12.2 Research Directions

1. **Hybrid Search** - Combine dense (FAISS) + sparse (BM25) + graph (Neo4j)
2. **Active Learning** - Use confidence_score to prioritize re-labeling
3. **Federated Queries** - Cross-database join optimization
4. **Vector Compression** - Product Quantization for 10x storage reduction

---

## 13. How To Clear All Databases

### 13.1 Complete Database Reset

**⚠️ WARNING**: This operation is IRREVERSIBLE. Always create a backup first.

#### Step 1: Create Backup

```bash
#!/bin/bash
# scripts/clear_all_databases.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/pre_clear_$DATE"

mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup at $BACKUP_DIR..."

# 1. PostgreSQL dump
pg_dump -h localhost -U lnsp -d lnsp -F c -f "$BACKUP_DIR/lnsp.dump"

# 2. FAISS indexes + metadata
cp artifacts/*.index artifacts/*.json artifacts/*.npz "$BACKUP_DIR/" 2>/dev/null || true

# 3. Neo4j node count (for verification)
cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) as count" \
  --format plain 2>/dev/null | tail -1 > "$BACKUP_DIR/neo4j_count.txt"

echo "✅ Backup complete: $BACKUP_DIR ($(du -sh $BACKUP_DIR | cut -f1))"
```

#### Step 2: Clear PostgreSQL

```bash
# Truncate all tables and reset sequences
psql -h localhost -U lnsp -d lnsp <<'EOF'
-- Clear all data
TRUNCATE TABLE cpe_entry CASCADE;
TRUNCATE TABLE cpe_vectors CASCADE;

-- Reset UUID generation (gen_random_uuid() auto-generates, no sequence reset needed)
-- But if you're using a serial ID column, reset it:
-- ALTER SEQUENCE cpe_entry_id_seq RESTART WITH 1;

-- Verify empty
SELECT COUNT(*) as cpe_entry_count FROM cpe_entry;
SELECT COUNT(*) as cpe_vectors_count FROM cpe_vectors;
EOF

echo "✅ PostgreSQL cleared"
```

**Note on UUIDs**: The `cpe_id` field uses `gen_random_uuid()` which generates true random UUIDs (UUID v4). There's no counter to reset - each new UUID is globally unique and independent of previous values.

**If using serial counters** (not recommended for distributed systems):
```sql
-- Only needed if you have auto-incrementing integer IDs
ALTER SEQUENCE your_sequence_name RESTART WITH 1;
```

#### Step 3: Clear FAISS Indexes

```bash
# Remove all FAISS indexes and metadata
rm -f artifacts/*.index
rm -f artifacts/faiss_meta.json
rm -f artifacts/index_meta.json

# Optional: Remove NPZ files (but keep training data)
# rm -f artifacts/*.npz  # Uncomment to also clear NPZ files

echo "✅ FAISS indexes cleared"
```

#### Step 4: Clear Neo4j Graph

```bash
# Delete all nodes and relationships
cypher-shell -u neo4j -p password <<'EOF'
MATCH (n) DETACH DELETE n;
EOF

# Verify empty
cypher-shell -u neo4j -p password --format plain \
  "MATCH (n) RETURN count(n) as count;" 2>/dev/null | tail -1

echo "✅ Neo4j graph cleared"
```

#### Step 5: Verify Empty State

```bash
#!/bin/bash
# scripts/verify_empty_databases.sh

echo "=== Database Clear Verification ==="

# PostgreSQL
PG_COUNT=$(psql -h localhost -U lnsp -d lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")
echo "PostgreSQL: $PG_COUNT concepts (should be 0)"

# FAISS
FAISS_COUNT=$(ls -1 artifacts/*.index 2>/dev/null | wc -l)
echo "FAISS: $FAISS_COUNT index files (should be 0)"

# Neo4j
NEO4J_COUNT=$(cypher-shell -u neo4j -p password --format plain \
  "MATCH (n) RETURN count(n);" 2>/dev/null | tail -1)
echo "Neo4j: $NEO4J_COUNT nodes (should be 0)"

# Check all are zero
if [ "$PG_COUNT" = "0" ] && [ "$FAISS_COUNT" = "0" ] && [ "$NEO4J_COUNT" = "0" ]; then
    echo "✅ All databases cleared successfully"
    exit 0
else
    echo "❌ ERROR: Some databases still have data"
    exit 1
fi
```

### 13.2 One-Command Clear Script

**Complete automated clear**:

```bash
#!/bin/bash
# scripts/nuclear_clear.sh
set -e

echo "🚨 WARNING: This will DELETE ALL DATA from PostgreSQL, FAISS, and Neo4j"
echo "Backup will be created automatically"
echo ""
read -p "Type 'DELETE ALL' to confirm: " CONFIRM

if [ "$CONFIRM" != "DELETE ALL" ]; then
    echo "❌ Aborted"
    exit 1
fi

# 1. Backup
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/pre_clear_$DATE"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup..."
pg_dump -h localhost -U lnsp -d lnsp -F c -f "$BACKUP_DIR/lnsp.dump"
cp artifacts/*.{index,json,npz} "$BACKUP_DIR/" 2>/dev/null || true

# 2. Clear PostgreSQL
echo "🗑️  Clearing PostgreSQL..."
psql -h localhost -U lnsp -d lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;"

# 3. Clear FAISS
echo "🗑️  Clearing FAISS..."
rm -f artifacts/*.index artifacts/faiss_meta.json artifacts/index_meta.json

# 4. Clear Neo4j
echo "🗑️  Clearing Neo4j..."
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;" 2>/dev/null

# 5. Verify
echo "✅ Verification:"
PG_COUNT=$(psql -h localhost -U lnsp -d lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")
NEO4J_COUNT=$(cypher-shell -u neo4j -p password --format plain "MATCH (n) RETURN count(n);" 2>/dev/null | tail -1)
echo "  PostgreSQL: $PG_COUNT (should be 0)"
echo "  Neo4j: $NEO4J_COUNT (should be 0)"
echo "  Backup saved: $BACKUP_DIR"

if [ "$PG_COUNT" = "0" ] && [ "$NEO4J_COUNT" = "0" ]; then
    echo "✅ All databases cleared successfully"
else
    echo "❌ ERROR: Clear incomplete"
    exit 1
fi
```

### 13.3 UUID/Global ID Management

**Key Points**:

1. **UUIDs are NOT sequential** - `gen_random_uuid()` generates globally unique IDs
2. **No counter to reset** - Each UUID is independent and probabilistically unique
3. **Format**: UUIDs are stored as `UUID` type in PostgreSQL (16 bytes)

**If you need to track ingestion order**, use timestamps:
```sql
SELECT cpe_id, created_at
FROM cpe_entry
ORDER BY created_at ASC
LIMIT 10;
```

**If you need sequential IDs for testing**:
```sql
-- Add a serial column (in addition to UUID primary key)
ALTER TABLE cpe_entry ADD COLUMN seq_id SERIAL;
CREATE INDEX idx_cpe_entry_seq_id ON cpe_entry(seq_id);

-- Reset sequence after clearing
ALTER SEQUENCE cpe_entry_seq_id_seq RESTART WITH 1;
```

**Best Practice**: Keep UUIDs as primary keys for:
- Distributed systems (no coordination needed)
- Merging databases (no ID conflicts)
- Security (non-enumerable IDs)

### 13.4 Recovery from Clear

**To restore from backup**:

```bash
#!/bin/bash
BACKUP_DIR="backups/pre_clear_20251009_202507"

# 1. Restore PostgreSQL
pg_restore -h localhost -U lnsp -d lnsp -c "$BACKUP_DIR/lnsp.dump"

# 2. Restore FAISS
cp "$BACKUP_DIR"/*.index "$BACKUP_DIR"/*.json artifacts/

# 3. Neo4j must be re-ingested (no full backup was created)
# Use the ingestion API to rebuild graph from PostgreSQL

# 4. Verify sync
./scripts/verify_data_sync.sh
```

---

## 14. References

### 14.1 Documentation

- [DATABASE_LOCATIONS.md](../DATABASE_LOCATIONS.md) - File locations and service management
- [INGESTION_API_COMPLETE.md](../INGESTION_API_COMPLETE.md) - API specification
- [TRAINING_SEQUENCE_GENERATION.md](../TRAINING_SEQUENCE_GENERATION.md) - Training data workflows
- [LNSP_LONG_TERM_MEMORY.md](../../LNSP_LONG_TERM_MEMORY.md) - Cardinal rules

### 14.2 External Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/17/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)

---

## 15. Approval & Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-09 | LNSP Team | Initial PRD - 3-way sync architecture |
| 1.1 | 2025-10-09 | LNSP Team | Added Section 13: How To Clear All Databases |

**Status**: ✅ **Approved for Production**

---

## Appendix A: Quick Reference

**PostgreSQL**:
```bash
# Connect
psql -h localhost -U lnsp -d lnsp

# Row count
SELECT COUNT(*) FROM cpe_entry;

# Recent concepts
SELECT cpe_id, concept_text, created_at
FROM cpe_entry ORDER BY created_at DESC LIMIT 10;
```

**FAISS**:
```python
from src.faiss_db import FaissDB
db = FaissDB("artifacts/ontology_13k_ivf_flat_ip.index")
print(f"Index size: {db.index.ntotal}")
```

**Neo4j**:
```cypher
// Node count
MATCH (c:Concept) RETURN count(c);

// Recent concepts
MATCH (c:Concept)
RETURN c.cpe_id, c.text, c.created_at
ORDER BY c.created_at DESC LIMIT 10;
```

**Sync Check**:
```bash
./scripts/verify_data_sync.sh
```
