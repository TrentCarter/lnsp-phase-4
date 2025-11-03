# Wikipedia 790K Dataset - Complete Reference Guide

## 📊 Dataset Overview

The LNSP Phase 4 project utilizes a massive Wikipedia dataset containing **790,391 chunks** of English Wikipedia content. This dataset serves as the foundation for LVM training, vector embeddings, and evaluation testing.

## 📁 Dataset Locations & Formats

### 1. Raw Text Data (JSONL)
**Location:** `data/datasets/wikipedia/wikipedia_500k.jsonl`
- **Size:** 2.1 GB
- **Format:** JSONL (JSON Lines)
- **Records:** 500,000 Wikipedia articles
- **Structure:**
  ```json
  {
    "id": "unique_article_id",
    "title": "Article Title",
    "text": "Full article text content...",
    "url": "wikipedia_url",
    "length": 25076
  }
  ```
- **Use Cases:**
  - LVM evaluation testing
  - Text-to-vector encoding
  - Raw text processing pipelines
  - Dashboard test data source

### 2. PostgreSQL Database
**Location:** PostgreSQL `lnsp` database
- **Table:** Various tables containing Wikipedia chunks
- **Records:** 790,391 entries
- **Connection:** Local PostgreSQL instance
- **Use Cases:**
  - Structured queries for chunk retrieval
  - Metadata storage and indexing
  - Relational data operations
  - API backend data source

### 3. Vector Embeddings (NPZ)
**Location:** `artifacts/wikipedia_500k_corrected_vectors.npz`
- **Size:** 2.1 GB
- **Format:** NumPy compressed array
- **Vectors:** 771,115 768-dimensional vectors
- **Encoding:** GTR-T5 embeddings
- **Use Cases:**
  - FAISS index building
  - Similarity search
  - Vector reconstruction testing
  - LVM training input

### 4. Training Sequences
**Location:** `artifacts/wikipedia_584k_fresh.npz`
- **Format:** NPZ with P6 sequences
- **Purpose:** LVM model training
- **Structure:** Sequential ontological data
- **Use Cases:**
  - P6b model training
  - Sequence prediction tasks
  - Ontological relationship learning

### 5. Additional Chunk Files
**Location:** `artifacts/fw10k_chunks.jsonl`
- **Records:** 10,000 sample chunks
- **Purpose:** Quick testing and development

**Location:** `test_data/100_test_chunks.jsonl`
- **Records:** 100 curated test chunks
- **Purpose:** Evaluation benchmarks

## 🔄 Data Flow & Processing Pipeline

```
Wikipedia Articles (500k JSONL)
        ↓
    Text Chunks
        ↓
    GTR-T5 Encoder (Port 7001)
        ↓
    768D Vectors (NPZ)
        ↓
    [LVM Processing] or [DIRECT Pipeline]
        ↓
    Vec2Text Decoder (Port 7002)
        ↓
    Reconstructed Text
```

## 🎯 Usage in Different Components

### LVM Evaluation Dashboard (Port 8999)
- **Source:** `data/datasets/wikipedia/wikipedia_500k.jsonl`
- **Loading:** First 100-1000 chunks for testing
- **Purpose:** Evaluate model performance on real Wikipedia text
- **Models:** DIRECT pipeline + trained LVM models

### FAISS Vector Index
- **Source:** `artifacts/wikipedia_500k_corrected_vectors.npz`
- **Index:** 771,115 vectors for similarity search
- **Dimension:** 768D embeddings
- **Usage:** Dense retrieval, nearest neighbor search

### Training Pipeline
- **Source:** `artifacts/wikipedia_584k_fresh.npz`
- **Format:** P6 sequential data
- **Models:** P6b, TwoTower, AMN variants
- **Purpose:** Next-concept prediction training

### Vec2Text Services
- **Encoder (7001):** Text → 768D vectors
- **Decoder (7002):** 768D vectors → Text
- **Dataset:** Uses embeddings from Wikipedia chunks

## 📈 Statistics & Metrics

| Component | Count | Size | Format |
|-----------|-------|------|--------|
| Raw Articles | 500,000 | 2.1 GB | JSONL |
| DB Entries | 790,391 | - | PostgreSQL |
| Vectors | 771,115 | 2.1 GB | NPZ |
| Training Seqs | 584,000+ | - | NPZ |
| Test Chunks | 100-10,000 | Variable | JSONL |

## 🔍 Accessing the Data

### Python Examples

#### Load JSONL Chunks
```python
import json

chunks = []
with open('data/datasets/wikipedia/wikipedia_500k.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000:  # Load first 1000
            break
        chunk = json.loads(line)
        chunks.append(chunk)
```

#### Load Vector Embeddings
```python
import numpy as np

data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz')
vectors = data['embeddings']  # Shape: (771115, 768)
metadata = data['metadata'] if 'metadata' in data else None
```

#### Query PostgreSQL
```sql
SELECT id, chunk_text, embedding_id 
FROM wikipedia_chunks 
LIMIT 100;
```

## 🚀 Quick Start Commands

### Test DIRECT Pipeline
```bash
curl -X POST http://localhost:8999/evaluate \
  -H "Content-Type: application/json" \
  -d '{"models":["DIRECT"],"test_mode":"both","num_test_cases":10}'
```

### Check Dataset Size
```bash
# Count JSONL lines
wc -l data/datasets/wikipedia/wikipedia_500k.jsonl

# Check NPZ shape
python -c "import numpy as np; d=np.load('artifacts/wikipedia_500k_corrected_vectors.npz'); print(f'Vectors: {d['embeddings'].shape}')"

# PostgreSQL count
psql -d lnsp -c "SELECT COUNT(*) FROM wikipedia_chunks;"
```

## 📝 Important Notes

1. **Size Discrepancy:** The 500k JSONL expands to 790k chunks after processing due to article segmentation
2. **Vector Count:** 771k vectors (slightly less than 790k) due to encoding failures or filtering
3. **Memory Usage:** Loading full datasets requires significant RAM (8+ GB recommended)
4. **Performance:** Use batched loading for large-scale operations

## 🔧 Maintenance & Updates

- **Update Vectors:** Run `tools/regenerate_vectors.py` after text changes
- **Rebuild Index:** Use `scripts/rebuild_faiss_index.sh` for vector index updates
- **Sync Database:** Execute `tools/sync_postgres_chunks.py` for DB consistency
- **Validate Data:** Run `tests/test_dataset_integrity.py` for sanity checks

---

*Last Updated: November 2, 2024*
*Dataset Version: v2.0 (790K chunks)*
*Previous Version: v1.0 (incorrectly documented as 80K)*
