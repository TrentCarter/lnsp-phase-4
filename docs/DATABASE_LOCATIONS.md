# LNSP Database Locations & Active Data Map

**Last Updated**: October 16, 2025
**Purpose**: Complete reference for all database and vector store locations with ACTIVE status indicators

---

## 🎯 Current Active Status

| Database | Status | Size | Records | Purpose |
|----------|--------|------|---------|---------|
| **PostgreSQL** | ✅ **ACTIVE** | ~1.5 GB | 80,636 concepts | Primary data store |
| **FAISS** | ✅ **ACTIVE** | 238 MB | 500k vectors | Semantic search |
| **Neo4j** | ⚠️ **EMPTY** | 0 nodes | 0 | Not currently used |
| **NPZ Vectors** | ✅ **ACTIVE** | 230 MB | 500k vectors | Training/inference |
| **LVM Models** | ✅ **ACTIVE** | ~200 MB | 4 models | Prediction |

---

## Overview

LNSP uses a **2-way synchronized data store** (currently):
1. ✅ **PostgreSQL** - Structured data (CPE entries, metadata, relationships)
2. ✅ **FAISS** - Vector indexes (semantic search)
3. ⚠️ **Neo4j** - Graph relationships (service running but empty, planned for future)

Plus additional **NPZ vector files** for training/inference and **SQLite** for local caching.

---

## 1. PostgreSQL (Primary Relational Database)

### ✅ ACTIVE - 80,636 Concepts

### Location
```bash
# Active database (PostgreSQL 17)
/opt/homebrew/var/postgresql@17/

# ⚠️ Alternate database (PostgreSQL 16, not running)
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
    dataset_source TEXT,  -- ✅ 'wikipedia' for active data
    parent_cpe_ids JSONB,
    child_cpe_ids JSONB,
    confidence_score REAL,
    quality_metrics JSONB,
    ...
)

-- Vector storage
cpe_vectors (
    cpe_id UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id),
    concept_vec REAL[],    -- 768D GTR-T5 vector
    question_vec REAL[],   -- 768D probe question vector
    tmd_dense REAL[],      -- 16D TMD vector (not currently used)
    fused_vec REAL[],      -- 784D combined vector (not currently used)
    fused_norm REAL
)
```

### Current Data Volume
```bash
# ✅ ACTIVE: 80,636 concepts from Wikipedia
psql -h localhost -U lnsp -d lnsp -c "
  SELECT COUNT(*) as total FROM cpe_entry;
  -- Result: 80,636
"

# Check data source breakdown
psql -h localhost -U lnsp -d lnsp -c "
  SELECT dataset_source, COUNT(*) as count
  FROM cpe_entry
  GROUP BY dataset_source
  ORDER BY count DESC;
"
```

---

## 2. FAISS Vector Indexes

### ✅ ACTIVE - 500k Wikipedia Vectors

### Location
```bash
# All FAISS indexes stored in:
./artifacts/

# ✅ ACTIVE INDEX:
./artifacts/wikipedia_500k_corrected_ivf_flat_ip.index  # 238 MB - Wikipedia 500k vectors

# 🗑️ DEPRECATED (old ontology data):
./artifacts/fw10k_ivf_flat_ip.index              # 6.1 MB  (replaced by Wikipedia data)
./artifacts/ontology_13k_ivf_flat_ip.index        # 6.1 MB  (ontology data, not for LVM training)
./artifacts/ontology_13k_ivf_flat_ip_rebuilt.index # 6.4 MB (deprecated)

# ✅ Metadata files:
./artifacts/faiss_meta.json                       # Index metadata (dimension, count, ids)
```

### Index Configuration
- **Type**: IVF_FLAT_IP (Inverted File with Flat quantization, Inner Product similarity)
- **Dimension**: 768 (GTR-T5 embeddings)
- **Clusters (nlist)**: 512
- **Search clusters (nprobe)**: 16
- **Vector count**: ~500,000

### Usage
```python
from src.faiss_db import FaissDB

# ✅ Load active Wikipedia index
faiss_db = FaissDB(
    index_path="artifacts/wikipedia_500k_corrected_ivf_flat_ip.index",
    meta_path="artifacts/faiss_meta.json",
    dimension=768
)

# Check status
print(f"Index size: {faiss_db.index.ntotal} vectors")  # ~500,000
```

---

## 3. Neo4j Graph Database

### ⚠️ NOT CURRENTLY ACTIVE (0 nodes)

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
# Status: running (but empty database)

# Web interface
http://localhost:7474/

# Bolt protocol
bolt://localhost:7687
```

### Current Status
```cypher
// Check node count
MATCH (c:Concept) RETURN count(c) AS concept_count;
// Result: 0 (database is empty)
```

### Planned Usage
Neo4j is installed and running but not currently populated. Future use for:
- Graph-based ontology traversal
- 6-degree separation + shortcuts
- Relationship inference

---

## 4. NPZ Vector Files (Numpy Arrays)

### ✅ ACTIVE Vector Files

### Location
```bash
# ✅ ACTIVE - Wikipedia 500k corpus
./artifacts/wikipedia_500k_corrected_vectors.npz     # 230 MB - Primary vector store

# ✅ ACTIVE - LVM Training Data
./artifacts/lvm/training_sequences_ctx5.npz          # 449 MB - 80k training sequences

# 🗑️ DEPRECATED (old ontology/test data):
./artifacts/fw10k_vectors.npz                # 150 MB (old test data)
./artifacts/ontology_13k.npz                 # 38 MB  (ontology data, not for LVM)
./artifacts/ontology_4k_tmd_llm.npz          # 31 MB  (deprecated)
```

### NPZ File Structure (Active Files)

#### Wikipedia Vectors (wikipedia_500k_corrected_vectors.npz)
```python
import numpy as np

# Load active Wikipedia vectors
data = np.load("artifacts/wikipedia_500k_corrected_vectors.npz", allow_pickle=True)

# Keys:
# - vectors: [N, 768] float32 array - GTR-T5 embeddings
# - texts: [N] object array - Original chunk text
# - ids: [N] object array - CPE IDs (UUIDs)
# - metadata: dict - Dataset metadata
```

#### LVM Training Sequences (training_sequences_ctx5.npz)
```python
# Load LVM training data
data = np.load("artifacts/lvm/training_sequences_ctx5.npz", allow_pickle=True)

# Keys:
# - context_vectors: [N, 5, 768] - Context windows (5 previous chunks)
# - target_vectors: [N, 768] - Next chunk to predict
# - context_texts: [N, 5] - Original context text (for debugging)
# - target_texts: [N] - Target text
# - sequence_ids: [N] - Unique sequence identifiers
```

### Critical for Training & Inference
These NPZ files provide the **vector ↔ text ↔ UUID** correlation needed for:
- **vecRAG search**: Query → FAISS → CPE ID → concept text
- **LVM training**: Sequential chunks → context/target pairs → training data
- **LVM inference**: Predicted vector → FAISS nearest neighbor → CPE ID → text

---

## 5. LVM Models (Latent Vector Models)

### ✅ ACTIVE - 4 Trained Models (October 16, 2025)

### Location
```bash
./artifacts/lvm/models/

# ✅ ACTIVE MODELS (Oct 16, 2025 - MSE Loss):
./artifacts/lvm/models/amn_20251016_133427/          # AMN: 0.5664 val cosine, 1.5M params
./artifacts/lvm/models/lstm_20251016_133934/         # LSTM: 0.5758 val cosine, 5.1M params
./artifacts/lvm/models/gru_20251016_134451/          # GRU: 0.5754 val cosine, 7.1M params
./artifacts/lvm/models/transformer_20251016_135606/  # Transformer: 0.5820 val cosine, 17.9M params

# Each model directory contains:
# - best_model.pt                 # Model checkpoint
# - training.log                  # Training log
# - config.json                   # Hyperparameters
```

### Model Performance Summary

| Model | Val Cosine | ms/Query | Params | Recommended For |
|-------|-----------|----------|--------|-----------------|
| **Transformer** | **0.5820** | 2.68 | 17.9M | Maximum accuracy |
| **LSTM** | **0.5758** | 0.56 | 5.1M | **Best balance** ⭐ |
| **GRU** | **0.5754** | 2.08 | 7.1M | Good middle ground |
| **AMN** | **0.5664** | **0.49** | **1.5M** | Ultra-low latency |

### Usage
```python
import torch
from app.lvm.models import create_model

# Load LSTM model (recommended for production)
checkpoint = torch.load('artifacts/lvm/models/lstm_20251016_133934/best_model.pt')
model = create_model('lstm', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
context_vectors = torch.randn(1, 5, 768)  # [batch, context_len, dim]
predicted_vector = model(context_vectors)  # [batch, 768]
```

See `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` for full benchmarks.

---

## 6. SQLite Databases (Local Caching)

### ✅ ACTIVE Caches

### Location
```bash
# CPESH cache
./artifacts/cpesh_index.db                   # 20 KB - CPESH extraction cache

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

## 7. Raw Data Sources

### ✅ ACTIVE - Wikipedia Dataset

```bash
# Wikipedia raw data (primary source for LVM training)
./data/datasets/wikipedia/                   # Raw Wikipedia articles

# FactoidWiki (DEPRECATED - not used per CLAUDE.md rules)
./data/factoidwiki_1k.jsonl                  # 🗑️ DO NOT USE (taxonomic, not sequential)
```

### 🗑️ DEPRECATED - Ontology Data (Not for LVM Training)

```bash
# Ontologies (for vecRAG/GraphRAG only, NOT for LVM training)
./artifacts/ontology_chains/swo_chains.jsonl         # Software Ontology
./artifacts/ontology_chains/go_chains.jsonl          # Gene Ontology
./artifacts/ontology_chains/dbpedia_chains.jsonl     # DBpedia
./artifacts/ontology_chains/wordnet_chains.jsonl     # WordNet

# ⚠️ WARNING: Do NOT use ontologies for LVM training!
# Reason: Taxonomic hierarchies, not sequential narrative flow
# See: docs/LVM_TRAINING_CRITICAL_FACTS.md
```

---

## Database Synchronization

### Current Sync Status

✅ **PostgreSQL ↔ FAISS**: Synchronized (80,636 concepts in both)
⚠️ **Neo4j**: Empty (not currently synced)

### Atomic Write Operations
```python
# Current ingestion writes to PostgreSQL + FAISS atomically:

# 1. PostgreSQL
insert_cpe_entry(pg_conn, cpe_entry_data)
upsert_cpe_vectors(pg_conn, cpe_id, vectors)

# 2. FAISS
faiss_db.add_vectors(vectors, ids=[cpe_id])
faiss_db.save()  # CRITICAL: Persist to disk

# 3. Neo4j (TODO: implement when needed)
# create_concept_node(neo4j_session, cpe_id, concept_text)
```

### Verification
```bash
# Check sync status
psql -h localhost -U lnsp -d lnsp -c "SELECT COUNT(*) FROM cpe_entry;"
# Expected: 80,636

# Check FAISS
python3 -c "import faiss; idx = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index'); print(f'FAISS: {idx.ntotal}')"
# Expected: ~500,000 (includes duplicates/variations)
```

---

## Backup Locations

### Baseline Backups
```bash
./backups/baseline_v1.0_20250929_215259/     # Pre-Wikipedia data
./backups/pre_clear_20251009_202507/         # Before Oct 9 cleanup

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

## Disk Usage Summary

```bash
# Check total disk usage
du -sh artifacts/
# Current: ~1.2 GB

# ✅ ACTIVE Breakdown:
# - FAISS index (Wikipedia 500k): 238 MB
# - NPZ vectors (Wikipedia 500k): 230 MB
# - LVM training sequences: 449 MB
# - LVM models (4 models): ~200 MB
# - Cache files: ~20 MB

# 🗑️ DEPRECATED (can be deleted):
# - Old ontology indexes: ~20 MB
# - Old test data: ~200 MB
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

# Neo4j (service running but not populated)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# ✅ ACTIVE FAISS Index
export LNSP_FAISS_INDEX="artifacts/wikipedia_500k_corrected_ivf_flat_ip.index"
export LNSP_FAISS_META="artifacts/faiss_meta.json"

# LLM (for CPESH generation)
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Vec2Text
export VEC2TEXT_FORCE_PROJECT_VENV=1
export VEC2TEXT_DEVICE=cpu
export TOKENIZERS_PARALLELISM=false
```

---

## Quick Reference Table

| Database | Type | Location | Size | Records | Status |
|----------|------|----------|------|---------|--------|
| **PostgreSQL** | Relational | `/opt/homebrew/var/postgresql@17/` | ~1.5 GB | 80,636 | ✅ **ACTIVE** |
| **FAISS (Wikipedia)** | Vector | `artifacts/wikipedia_500k_corrected_ivf_flat_ip.index` | 238 MB | 500k | ✅ **ACTIVE** |
| **NPZ (Wikipedia)** | Files | `artifacts/wikipedia_500k_corrected_vectors.npz` | 230 MB | 500k | ✅ **ACTIVE** |
| **LVM Training** | Files | `artifacts/lvm/training_sequences_ctx5.npz` | 449 MB | 80k seq | ✅ **ACTIVE** |
| **LVM Models** | Checkpoints | `artifacts/lvm/models/` | ~200 MB | 4 models | ✅ **ACTIVE** |
| **Neo4j** | Graph | `/opt/homebrew/var/neo4j/` | 0 MB | 0 | ⚠️ **EMPTY** |
| **CPESH Cache** | SQLite | `artifacts/cpesh_index.db` | 20 KB | N/A | ✅ Available |

---

## Summary

### ✅ ACTIVE Production Data
- **PostgreSQL**: 80,636 Wikipedia concepts
- **FAISS**: 500k Wikipedia vectors (238 MB index)
- **NPZ**: 500k Wikipedia vectors (230 MB file)
- **LVM Training**: 80k sequential training pairs (449 MB)
- **LVM Models**: 4 trained models (AMN, LSTM, GRU, Transformer)

### ⚠️ Not Currently Used
- **Neo4j**: Service running but database empty (0 nodes)
- **Ontology data**: Available but not used for LVM training (taxonomic, not sequential)

### 🗑️ Deprecated (Can Be Removed)
- Old ontology FAISS indexes (`fw10k_*`, `ontology_13k_*`)
- Old test NPZ files (`fw1k_vectors.npz`, etc.)
- FactoidWiki data (not suitable for LVM training)

### 📊 Data Flow
```
Wikipedia Articles → PostgreSQL (80k concepts)
                  ↓
                  → FAISS Index (500k vectors, 238 MB)
                  ↓
                  → NPZ Vectors (500k vectors, 230 MB)
                  ↓
                  → LVM Training Sequences (80k pairs, 449 MB)
                  ↓
                  → Trained LVM Models (4 architectures)
```

---

## Related Documentation

- **LVM Training Data**: See `docs/LVM_DATA_MAP.md` (comprehensive LVM-specific data guide)
- **Data Flow Diagram**: See `docs/DATA_FLOW_DIAGRAM.md` (visual system architecture)
- **Training Rules**: See `CLAUDE.md` and `docs/LVM_TRAINING_CRITICAL_FACTS.md`
- **Performance**: See `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`

---

**Last Updated**: October 16, 2025
**Status**: ✅ All active systems operational
**Next Review**: When adding Neo4j graph data or new training datasets
