# LNSP Baseline v1.0 - vecRAG System

**Date**: 2025-09-29
**Git Tag**: `v1.0-baseline-vecrag`
**Status**: ✅ Production-Ready

---

## Overview

This baseline represents a **complete, working vecRAG system** with:
- ✅ 999 real Wikipedia documents ingested
- ✅ Real LLM extraction (Ollama Llama 3.1:8b)
- ✅ Real embeddings (GTR-T5 768D)
- ✅ CPESH generation (97.3% coverage with soft/hard negatives)
- ✅ TMD encoding (16D task metadata)
- ✅ Neo4j graph (999 Concepts, 1,629 Entities, 2,124 relationships)
- ✅ Faiss index (999 × 784D vectors)
- ✅ Content-based deduplication
- ✅ Phase-2 entity resolution

---

## System State Snapshot

### Database Statistics
```
PostgreSQL (lnsp):
  - cpe_entry: 999 records
  - cpe_vectors: 999 records (768D GTR-T5 embeddings)
  - Unique documents: 999
  - CPESH coverage: 972/999 (97.3%)
  - Dataset: factoid-wiki-large

Neo4j:
  - Concept nodes: 999
  - Entity nodes: 1,629
  - RELATES_TO relationships: 2,124
  - Cross-document entity linking: Active

Faiss:
  - Index: artifacts/fw1k.npz (18.7 MB)
  - Vectors: 999 × 784D fused vectors
  - Metadata: CPE IDs, lane indices, doc IDs, concept texts
```

### Component Versions
```bash
# LLM
Ollama: latest (as of 2025-09-29)
Model: llama3.1:8b

# Embeddings
sentence-transformers: latest
Model: sentence-transformers/gtr-t5-base (768D)

# Databases
PostgreSQL: 14+ (local)
Neo4j: 5.x (local, password: password)

# Python
Python: 3.11+
Key packages: see requirements.txt
```

---

## How to Restore This Baseline

### 1. Restore Code
```bash
# Clone repository
git clone <repo-url> lnsp-phase-4
cd lnsp-phase-4

# Checkout baseline tag
git checkout v1.0-baseline-vecrag

# Verify tag info
git show v1.0-baseline-vecrag
```

### 2. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.vectorizer import EmbeddingBackend; print('✓ Imports OK')"
```

### 3. Start Services
```bash
# Start Ollama
ollama serve &
ollama pull llama3.1:8b

# Verify LLM
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'

# Start PostgreSQL (if not running)
brew services start postgresql@14

# Start Neo4j (if not running)
brew services start neo4j

# Verify services
./scripts/check_services.sh
```

### 4. Restore Database
```bash
# Restore PostgreSQL backup
psql postgres -c "DROP DATABASE IF EXISTS lnsp;"
psql postgres -c "CREATE DATABASE lnsp OWNER lnsp;"
psql lnsp < backups/lnsp_baseline_v1.0_20250929.sql

# Restore Neo4j (if backup exists)
neo4j-admin restore --from=backups/neo4j_baseline_v1.0_20250929

# Verify restoration
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"
cypher-shell -u neo4j -p password "MATCH (n:Concept) RETURN count(n);"
```

### 5. Restore Artifacts
```bash
# Copy Faiss index
cp backups/artifacts/fw1k.npz artifacts/

# Verify Faiss
python -c "
import numpy as np
data = np.load('artifacts/fw1k.npz')
print(f'✓ Faiss vectors: {data[\"fused\"].shape}')
"
```

### 6. Verify System
```bash
# Run baseline verification script
python reports/scripts/verify_baseline_v1.0.py

# Expected output:
# ✅ PostgreSQL: 999 CPE entries
# ✅ Neo4j: 999 Concepts, 1629 Entities
# ✅ Faiss: 999 vectors (784D)
# ✅ CPESH: 972/999 (97.3%)
# ✅ All services running
```

---

## How to Use This Baseline

### Generate Ingestion Report
```bash
python reports/scripts/generate_ingestion_report.py
# Output: reports/output/ingestion_report_<timestamp>.md
```

### Ingest New Data (Preserves Existing)
```bash
# Content-based deduplication prevents duplicates
./.venv/bin/python -m src.ingest_factoid \
  --file-path data/new_corpus.jsonl \
  --num-samples 100 \
  --write-pg \
  --write-neo4j \
  --faiss-out artifacts/new_vectors.npz
```

### Run Phase-2 Entity Resolution
```bash
# On existing data
python run_phase2_on_existing.py

# During ingestion (automatic)
# Phase-2 runs automatically via src/pipeline/p10_entity_resolution.py
```

### Query the System
```bash
# Start retrieval API
./.venv/bin/uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8080

# Test query
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Who was Ada Lovelace?", "top_k": 5}'
```

---

## Key Files & Locations

### Code
```
src/
├── ingest_factoid.py          # Main ingestion pipeline
├── prompt_extractor.py        # LLM extraction (CPESH)
├── vectorizer.py              # GTR-T5 embeddings
├── tmd_encoder.py             # TMD encoding
├── db_postgres.py             # PostgreSQL interface
├── db_neo4j.py                # Neo4j interface
├── db_faiss.py                # Faiss interface
├── pipeline/
│   ├── p9_graph_extraction.py # Phase-1 graph building
│   └── p10_entity_resolution.py # Phase-2 entity resolution
└── api/
    └── retrieve.py            # Retrieval API
```

### Data
```
data/
└── factoidwiki_1k.jsonl       # Source data (first 999 items used)

artifacts/
└── fw1k.npz                   # Faiss index (999 vectors)

backups/
├── lnsp_baseline_v1.0_20250929.sql    # PostgreSQL dump
└── neo4j_baseline_v1.0_20250929/      # Neo4j backup
```

### Documentation
```
docs/
├── baselines/
│   └── BASELINE_v1.0_vecRAG.md        # This file
├── PRDs/
│   └── PRD_KnownGood_vecRAG_Data_Ingestion.md
├── design_documents/
│   ├── deduplication_strategy.md
│   └── prompt_template_lightRAG_TMD_CPE.md
└── howto/
    └── how_to_access_local_AI.md
```

---

## Known Good Procedures

### 1. Clean Slate Ingestion
```bash
# Start fresh (WARNING: destroys existing data)
psql postgres -c "DROP DATABASE IF EXISTS lnsp; CREATE DATABASE lnsp OWNER lnsp;"
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"
rm artifacts/*.npz

# Initialize schema
psql lnsp < scripts/schema.sql

# Ingest from scratch
./.venv/bin/python -m src.ingest_factoid \
  --file-path data/factoidwiki_1k.jsonl \
  --num-samples 999 \
  --write-pg \
  --write-neo4j \
  --faiss-out artifacts/fw1k.npz

# Run Phase-2 entity resolution
python run_phase2_on_existing.py
```

### 2. Incremental Ingestion
```bash
# Add new data (deduplication active)
./.venv/bin/python -m src.ingest_factoid \
  --file-path data/new_data.jsonl \
  --num-samples 100 \
  --write-pg \
  --write-neo4j \
  --faiss-out artifacts/new_vectors.npz

# Phase-2 runs automatically during ingestion
```

### 3. Verify System Health
```bash
# Check all components
python reports/scripts/verify_baseline_v1.0.py

# Generate current report
python reports/scripts/generate_ingestion_report.py
```

---

## Critical Configuration

### Environment Variables
```bash
# LLM
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Database
export PG_DSN="host=localhost dbname=lnsp user=lnsp password=lnsp"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# Embeddings
export LNSP_EMBEDDER_PATH="./models/gtr-t5-base"  # Optional: use cached model
```

### PostgreSQL Schema
- Tables: `cpe_entry`, `cpe_vectors`
- Key constraint: `cpe_entry_pkey` on `cpe_id` (UUID)
- Deduplication: `ON CONFLICT (cpe_id) DO NOTHING`
- Index: `cpe_source_chunk_hash` for content-based lookups

### Neo4j Schema
- Node labels: `Concept`, `Entity`
- Relationship: `RELATES_TO`
- Properties: `cpe_id`, `concept_text`, `entity_name`, `weight`

---

## Performance Metrics

### Ingestion Rate
- **Average**: ~14 items/minute
- **972 items**: ~70 minutes (18:51 → 20:00)
- Bottleneck: LLM extraction (Ollama local)

### Storage
- PostgreSQL: ~50 MB (999 records)
- Neo4j: ~10 MB (2,628 nodes, 2,124 edges)
- Faiss: 18.7 MB (999 × 784D vectors)
- Total: ~80 MB

### Query Performance
- Vector search (Faiss): <10ms (999 vectors)
- Graph traversal (Neo4j): <50ms (2,628 nodes)
- PostgreSQL lookup: <5ms (indexed)

---

## What Makes This Baseline Special

### 1. No Stub Data
- ✅ Real LLM extraction (not placeholder functions)
- ✅ Real embeddings (not random vectors)
- ✅ Real CPESH generation (not empty arrays)
- ✅ Real Wikipedia content (not test samples)

### 2. Complete Pipeline
- ✅ Ingestion → Extraction → Encoding → Storage
- ✅ Graph building (Phase-1 + Phase-2)
- ✅ Entity resolution across documents
- ✅ Content-based deduplication

### 3. Production Quality
- ✅ 97.3% CPESH coverage
- ✅ Cross-document entity linking
- ✅ TMD-aware retrieval ready
- ✅ Multiple retrieval backends (Faiss, Neo4j, PostgreSQL)

### 4. Reproducible
- ✅ Deterministic UUID generation
- ✅ Documented environment setup
- ✅ Database backup procedures
- ✅ Verification scripts

---

## Future Enhancements (Post-Baseline)

### Planned
1. **GraphRAG Integration**: Full LightRAG hybrid retrieval
2. **Vec2Text Inversion**: Reconstruct text from embeddings
3. **TMD-Aware Routing**: Lane-specific retrieval
4. **Evaluation Framework**: BEIR/MTEB benchmarks
5. **API Hardening**: Rate limiting, auth, caching

### Experimental
1. **Fuzzy Deduplication**: Near-duplicate detection
2. **Multi-Modal**: Image + text ingestion
3. **Streaming Ingestion**: Real-time updates
4. **Federated Search**: Multi-corpus retrieval

---

## Troubleshooting

### Services Not Running
```bash
# Check status
./scripts/check_services.sh

# Restart services
brew services restart postgresql@14
brew services restart neo4j
ollama serve &
```

### Database Connection Errors
```bash
# PostgreSQL
psql lnsp -c "SELECT 1;"  # Should return 1

# Neo4j
cypher-shell -u neo4j -p password "RETURN 1;"  # Should return 1
```

### Ollama Not Responding
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart
pkill ollama
ollama serve &
ollama pull llama3.1:8b
```

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

---

## Contacts & References

### Documentation
- Main README: `/README.md`
- Claude instructions: `/CLAUDE.md`
- Design docs: `/docs/design_documents/`
- PRDs: `/docs/PRDs/`

### Key Commits
- Baseline commit: (to be tagged `v1.0-baseline-vecrag`)
- CPESH implementation: 30be447
- Known-good vecRAG: 000b07b

### Related Sprints
- Sprint S1 (2025-09-29): 999-item ingestion
- Sprint S2 (2025-09-29): Phase-2 entity resolution
- Earlier sprints: See `/sprints/`

---

**Last Updated**: 2025-09-29
**Next Review**: After first production deployment
**Baseline Maintainer**: Project Lead