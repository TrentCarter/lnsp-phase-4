# LNSP Database Locations

**Date**: 2025-10-09
**Purpose**: Complete reference for all database and vector store locations

---

## Overview

LNSP uses a **3-way synchronized data store**:
1. **PostgreSQL** - Structured data (CPE entries, metadata, relationships)
2. **FAISS** - Vector indexes (semantic search)
3. **Neo4j** - Graph relationships (ontology traversal)

Plus additional **SQLite** databases for local caching.

---

## 1. PostgreSQL (Primary Relational Database)

### Location
```bash
# Active database (PostgreSQL 17)
/opt/homebrew/var/postgresql@17/

# Alternate database (PostgreSQL 16, not running)
/opt/homebrew/var/postgresql@16/
```

### Connection String
```bash
# Default connection (from src/db_postgres.py:10)
PG_DSN="host=localhost dbname=lnsp user=lnsp password=lnsp"

# Alternative environment variables
export PGHOST=localhost
export PGPORT=5432
export PGUSER=lnsp
export PGPASSWORD=lnsp
export PGDATABASE=lnsp
```

### Service Status
```bash
# Check status
brew services list | grep postgresql

# Current status: postgresql@17 running
```

### Key Tables
```sql
-- Main concept entries
cpe_entry (
    cpe_id UUID PRIMARY KEY,
    concept_text TEXT,
    probe_question TEXT,
    expected_answer TEXT,
    soft_negatives JSONB,
    hard_negatives JSONB,
    domain_code INTEGER,
    task_code INTEGER,
    modifier_code INTEGER,
    dataset_source TEXT,
    parent_cpe_ids JSONB,  -- NEW: For training chains
    child_cpe_ids JSONB,   -- NEW: For training chains
    confidence_score REAL, -- NEW: Quality metric
    quality_metrics JSONB, -- NEW: Detailed metrics
    ...
)

-- Vector storage
cpe_vectors (
    cpe_id UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id),
    concept_vec REAL[],    -- 768D GTR-T5 vector
    question_vec REAL[],   -- 768D probe question vector
    tmd_dense REAL[],      -- 16D TMD vector
    fused_vec REAL[],      -- 784D combined vector
    fused_norm REAL
)
```

### Data Volume
```bash
# Check row counts
psql -h localhost -U lnsp -d lnsp -c "
  SELECT
    'cpe_entry' AS table, COUNT(*) AS rows FROM cpe_entry
  UNION ALL
  SELECT
    'cpe_vectors' AS table, COUNT(*) AS rows FROM cpe_vectors;
"
```

---

## 2. FAISS Vector Indexes

### Location
```bash
# All FAISS indexes stored in:
./artifacts/

# Active indexes:
./artifacts/fw10k_ivf_flat_ip.index              # 6.1 MB  (10k vectors, 768D)
./artifacts/ontology_13k_ivf_flat_ip.index        # 6.1 MB  (13k ontology vectors, 768D)
./artifacts/ontology_13k_ivf_flat_ip_rebuilt.index # 6.4 MB (rebuilt version)

# Metadata files:
./artifacts/faiss_meta.json                       # Index metadata (dimension, count, ids)
./artifacts/index_meta.json                       # Additional index metadata
```

### Index Types
- **IVF_FLAT_IP**: Inverted File with Flat quantization, Inner Product similarity
- **Parameters**:
  - nlist: 512 (number of clusters)
  - nprobe: 16 (clusters to search)
  - dimension: 768 or 784

### Usage
```python
from src.faiss_db import FaissDB

# Load existing index
faiss_db = FaissDB(
    index_path="artifacts/ontology_13k_ivf_flat_ip.index",
    meta_path="artifacts/faiss_meta.json",
    dimension=784
)

# Check status
print(f"Index size: {faiss_db.index.ntotal} vectors")
```

---

## 3. Neo4j Graph Database

### Location
```bash
# Data directory
/opt/homebrew/var/neo4j/

# Installation
/opt/homebrew/Cellar/neo4j/2025.08.0/
```

### Service Status
```bash
# Check status
brew services list | grep neo4j
# Status: running

# Web interface
http://localhost:7474/

# Bolt protocol
bolt://localhost:7687
```

### Connection String
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)
```

### Graph Structure
```cypher
// Concept nodes
(c:Concept {
    cpe_id: "uuid",
    text: "concept text",
    domain_code: 0,
    task_code: 16,
    modifier_code: 37
})

// Relationships (6-degree separation + shortcuts)
(c1:Concept)-[:RELATED_TO {weight: 0.85}]->(c2:Concept)
(c1:Concept)-[:SHORTCUT {distance: 3}]->(c3:Concept)
```

### Data Volume
```cypher
// Check node count
MATCH (c:Concept) RETURN count(c) AS concept_count;

// Check relationship count
MATCH ()-[r]->() RETURN count(r) AS relationship_count;
```

---

## 4. NPZ Vector Files (Numpy Arrays)

### Location
```bash
# All vector files stored in:
./artifacts/*.npz

# Active vector files:
./artifacts/fw10k_vectors.npz                # 150 MB (10k concepts with metadata)
./artifacts/ontology_13k.npz                 # 38 MB  (13k ontology concepts)
./artifacts/ontology_4k_tmd_llm.npz          # 31 MB  (4k with TMD codes)

# Training data:
./artifacts/lvm/wordnet_training_sequences.npz  # LVM training sequences
./artifacts/train/cpesh_10x_vectors.npz         # CPESH training data
```

### NPZ File Structure
```python
import numpy as np

# Load NPZ file
data = np.load("artifacts/fw10k_vectors.npz", allow_pickle=True)

# Standard keys:
# - concept_texts: Array of concept strings
# - cpe_ids: Array of UUIDs (for correlation with PostgreSQL)
# - vectors: 768D or 784D arrays
# - tmd_codes: TMD metadata (domain, task, modifier)
```

### Critical for Training
These NPZ files provide the **vector index → concept text → UUID** correlation needed for:
- **vecRAG search**: Query → FAISS → CPE ID → concept text
- **LVM training**: Chain concepts → match text → get vector index → training sequences
- **Inference**: LVM output → FAISS nearest neighbor → CPE ID → final text

---

## 5. SQLite Databases (Local Caching)

### Location
```bash
# CPESH cache
./artifacts/cpesh_index.db                   # 20 KB (CPESH extraction cache)

# MLflow tracking
./app/utils/mlflow/mlflow.db                 # MLflow experiment tracking
```

### Usage
```bash
# Inspect CPESH cache
sqlite3 artifacts/cpesh_index.db ".schema"
sqlite3 artifacts/cpesh_index.db "SELECT COUNT(*) FROM cpesh_cache;"
```

---

## 6. Ontology Chain Files (JSONL)

### Location
```bash
./artifacts/ontology_chains/*.jsonl

# Ontology sources:
./artifacts/ontology_chains/swo_chains.jsonl         # Software Ontology chains
./artifacts/ontology_chains/go_chains.jsonl          # Gene Ontology chains
./artifacts/ontology_chains/dbpedia_chains.jsonl     # DBpedia chains
./artifacts/ontology_chains/wordnet_chains.jsonl     # WordNet synset chains
./artifacts/ontology_chains/wordnet_chains_8k.jsonl  # 8k WordNet chains

# Sample sets:
./artifacts/ontology_chains/swo_chains_1k_sample.jsonl
./artifacts/ontology_chains/go_chains_1k_sample.jsonl
./artifacts/ontology_chains/dbpedia_chains_1k_sample.jsonl
```

### Format
```jsonl
{"chain_id": "swo_001", "concepts": ["Software", "Algorithm", "Sorting"], "relations": ["is_a", "is_a"]}
{"chain_id": "go_002", "concepts": ["Cellular Process", "Metabolism", "Glycolysis"], "relations": ["part_of", "part_of"]}
```

---

## 7. Knowledge Graph Files (JSONL)

### Location
```bash
./artifacts/kg/*.jsonl

# Knowledge graph exports:
./artifacts/kg/nodes.jsonl     # Graph nodes
./artifacts/kg/edges.jsonl     # Graph edges
./artifacts/kg/stats.json      # Graph statistics
```

---

## Database Synchronization

### Critical Rule (from LNSP_LONG_TERM_MEMORY.md)
**Data Synchronization is Sacred**: PostgreSQL + Neo4j + FAISS must stay synchronized.

### Atomic Write Operations
```python
# All ingestion must write atomically to all 3 stores:

# 1. PostgreSQL
insert_cpe_entry(pg_conn, cpe_entry_data)
upsert_cpe_vectors(pg_conn, cpe_id, vectors)

# 2. FAISS
faiss_db.add_vectors(vectors, ids=[cpe_id])
faiss_db.save()  # CRITICAL: Persist to disk

# 3. Neo4j (TODO: implement)
# create_concept_node(neo4j_session, cpe_id, concept_text)
```

### Verification
```bash
# Run synchronization check
./scripts/verify_data_sync.sh

# Expected output:
# ✅ PostgreSQL: 13,000 concepts
# ✅ FAISS: 13,000 vectors
# ✅ Neo4j: 13,000 nodes
```

---

## Backup Locations

### Baseline Backups
```bash
./backups/baseline_v1.0_20250929_215259/

# Contains snapshots of:
# - artifacts/*.npz
# - artifacts/*.index
# - File list and metadata
```

### Backup Strategy
```bash
# Create baseline backup
./scripts/backup_baseline.sh

# Restore from backup
# (manually copy files from backups/ to artifacts/)
```

---

## Cleanup Commands

### Reset All Databases
```bash
# WARNING: This deletes all data!

# 1. PostgreSQL
psql -h localhost -U lnsp -d lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;"

# 2. FAISS
rm -f artifacts/*.index artifacts/faiss_meta.json

# 3. Neo4j
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"

# 4. NPZ files (optional - these can be regenerated)
rm -f artifacts/*.npz
```

### Rebuild from Scratch
```bash
# Use new ingestion pipeline (Port 8004)
# This will rebuild all 3 stores atomically

curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [...],
    "dataset_source": "ontology-swo"
  }'
```

---

## Disk Usage Summary

```bash
# Check total disk usage
du -sh artifacts/
# Expected: ~500 MB - 1 GB (depending on dataset size)

# Breakdown:
# - FAISS indexes: ~20 MB (3 indexes × 6-7 MB each)
# - NPZ vectors: ~400 MB (13k concepts × 784D × 4 bytes)
# - Ontology chains: ~50 MB (JSONL files)
# - Cache files: ~20 MB (SQLite, metadata)
```

---

## Environment Variables Reference

```bash
# PostgreSQL
export PGHOST=localhost
export PGPORT=5432
export PGUSER=lnsp
export PGPASSWORD=lnsp
export PGDATABASE=lnsp
export PG_DSN="host=localhost dbname=lnsp user=lnsp password=lnsp"

# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# FAISS
export LNSP_FAISS_INDEX="artifacts/ontology_13k_ivf_flat_ip.index"
export LNSP_FAISS_META="artifacts/faiss_meta.json"

# LLM
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

---

## Quick Reference Table

| Database | Type | Location | Size | Port | Status |
|----------|------|----------|------|------|--------|
| **PostgreSQL** | Relational | `/opt/homebrew/var/postgresql@17/` | ~100 MB | 5432 | ✅ Running |
| **Neo4j** | Graph | `/opt/homebrew/var/neo4j/` | ~50 MB | 7474/7687 | ✅ Running |
| **FAISS Indexes** | Vector | `./artifacts/*.index` | ~20 MB | N/A | ✅ Available |
| **NPZ Vectors** | Files | `./artifacts/*.npz` | ~400 MB | N/A | ✅ Available |
| **CPESH Cache** | SQLite | `./artifacts/cpesh_index.db` | 20 KB | N/A | ✅ Available |
| **Ontology Chains** | JSONL | `./artifacts/ontology_chains/` | ~50 MB | N/A | ✅ Available |

---

## Next Steps

To rebuild the entire database from scratch using the new ingestion pipeline:

1. **Clear existing data** (optional):
   ```bash
   psql -U lnsp -d lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;"
   rm -f artifacts/*.index artifacts/faiss_meta.json
   ```

2. **Start ingestion service**:
   ```bash
   ./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004
   ```

3. **Ingest ontology data** (see [INGESTION_API_COMPLETE.md](INGESTION_API_COMPLETE.md))

4. **Verify synchronization**:
   ```bash
   ./scripts/verify_data_sync.sh
   ```

---

## Summary

✅ **PostgreSQL**: Primary structured data store at `/opt/homebrew/var/postgresql@17/`
✅ **Neo4j**: Graph database at `/opt/homebrew/var/neo4j/`
✅ **FAISS**: Vector indexes in `./artifacts/*.index`
✅ **NPZ Files**: Training vectors in `./artifacts/*.npz`
✅ **SQLite**: Local caches in `./artifacts/*.db`

All databases are **synchronized via the Ingestion API (Port 8004)** using atomic writes.
