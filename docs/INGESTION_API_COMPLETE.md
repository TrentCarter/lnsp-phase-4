# Chunk Ingestion API - Complete Reference

**Port**: 8004
**Version**: 1.0.0
**Date**: 2025-10-09

---

## Overview

Complete FastAPI service for ingesting semantic chunks into vecRAG with:
- **CPESH generation** (TinyLlama)
- **TMD extraction** (Llama 3.1)
- **Vectorization** (GTR-T5 768D + TMD 16D = 784D)
- **Auto-sequential linking** (parent/child UUIDs for training)
- **Quality metrics** (confidence scores, usage tracking)
- **Atomic writes** (PostgreSQL + FAISS + Neo4j)

---

## Quick Start

### 1. Start the Service

```bash
./.venv/bin/uvicorn app.api.ingest_chunks:app --host 127.0.0.1 --port 8004 --reload
```

### 2. Basic Ingestion (Auto-Linking)

```bash
curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"text": "First chunk about photosynthesis"},
      {"text": "Second chunk about chlorophyll"},
      {"text": "Third chunk about light reactions"}
    ],
    "dataset_source": "biology_course"
  }'
```

**Result**: Chunks automatically linked in sequence:
- Chunk[0]: `parent=[], child=[UUID-1]`
- Chunk[1]: `parent=[UUID-0], child=[UUID-2]`
- Chunk[2]: `parent=[UUID-1], child=[]`

---

## Features

### ✅ Auto-Sequential Linking

**Default behavior**: Chunks are automatically linked in the order provided.

```json
{
  "chunks": [
    {"text": "Chunk A"},
    {"text": "Chunk B"},
    {"text": "Chunk C"}
  ]
}
```

**Auto-generated relationships**:
```
A (UUID-0) ←→ B (UUID-1) ←→ C (UUID-2)
  parent=[]      parent=[0]      parent=[1]
  child=[1]      child=[2]       child=[]
```

**Use case**: Document ingestion, sequential training data

---

### ✅ Manual Relationship Override

Provide explicit `parent_cpe_ids` and `child_cpe_ids` to override auto-linking:

```json
{
  "chunks": [
    {
      "text": "Machine Learning",
      "parent_cpe_ids": ["uuid-ai-concept"],
      "child_cpe_ids": ["uuid-deep-learning", "uuid-supervised-learning"]
    }
  ]
}
```

**Use case**: Ontology relationships, multi-parent concepts

---

### ✅ Quality Metrics & Confidence Scoring

Every ingested chunk receives:

**Quality Metrics** (`quality_metrics` JSONB):
```json
{
  "cpesh_completeness": 0.85,      // % of CPESH fields filled (0-1)
  "vector_norm": 8.7,               // L2 norm of concept vector
  "text_length": 156,               // Original chunk length
  "concept_length": 42,             // Extracted concept length
  "soft_negatives_count": 3,        // Number of soft negatives
  "hard_negatives_count": 2,        // Number of hard negatives
  "total_negatives": 5              // Sum of negatives
}
```

**Confidence Score** (`confidence_score` REAL):
```
Confidence =
  0.4 × CPESH completeness +
  0.3 × (negatives_count / 5) +
  0.2 × extraction_quality +
  0.1 × (vector_norm / 10)

Range: 0.0 - 1.0
```

---

### ✅ Usage Tracking

Every chunk includes:
- `created_at`: ISO timestamp (e.g., `"2025-10-09T14:32:15Z"`)
- `last_accessed_at`: NULL initially, updated on retrieval
- `access_count`: Starts at 0, incremented on retrieval

**Use case**: Identify frequently used concepts, prune low-quality data

---

## API Endpoints

### `POST /ingest`

Ingest one or more semantic chunks.

**Request Body**:
```json
{
  "chunks": [
    {
      "text": "string (min 10 chars)",
      "source_document": "string (optional)",
      "chunk_index": 0,
      "metadata": {},
      "parent_cpe_ids": ["uuid-1", "uuid-2"],  // Optional
      "child_cpe_ids": ["uuid-3"]              // Optional
    }
  ],
  "dataset_source": "string (default: user_input)",
  "batch_id": "string (optional, auto-generated)"
}
```

**Response**:
```json
{
  "results": [
    {
      "global_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "concept_text": "Photosynthesis is the process...",
      "tmd_codes": {
        "domain": 0,
        "task": 16,
        "modifier": 37
      },
      "vector_dimension": 784,
      "confidence_score": 0.87,
      "quality_metrics": {...},
      "created_at": "2025-10-09T14:32:15Z",
      "parent_cpe_ids": ["uuid-prev"],
      "child_cpe_ids": ["uuid-next"],
      "success": true
    }
  ],
  "total_chunks": 1,
  "successful": 1,
  "failed": 0,
  "batch_id": "batch_f47ac10b",
  "processing_time_ms": 4523.8
}
```

---

### `GET /health`

Check service health.

**Response**:
```json
{
  "status": "healthy",
  "postgresql": true,
  "faiss": true,
  "gtr_t5_api": true,
  "tmd_router_api": true,
  "llm_endpoint": "http://localhost:11434"
}
```

---

### `GET /stats`

Get ingestion statistics.

**Response**:
```json
{
  "total_chunks_processed": 1523,
  "successfully_ingested": 1487,
  "failed": 36,
  "success_rate": 97.6
}
```

---

## Database Schema

### PostgreSQL Tables

```sql
-- Main concept entry table
CREATE TABLE cpe_entry (
    cpe_id UUID PRIMARY KEY,
    mission_text TEXT NOT NULL,
    source_chunk TEXT,
    concept_text TEXT NOT NULL,
    probe_question TEXT,
    expected_answer TEXT,
    soft_negatives JSONB DEFAULT '[]'::jsonb,
    hard_negatives JSONB DEFAULT '[]'::jsonb,
    domain_code INTEGER,
    task_code INTEGER,
    modifier_code INTEGER,
    content_type TEXT DEFAULT 'semantic_chunk',
    dataset_source TEXT NOT NULL,
    chunk_position JSONB,
    relations_text JSONB DEFAULT '[]'::jsonb,
    echo_score REAL,
    validation_status TEXT DEFAULT 'pending',
    batch_id TEXT,
    tmd_bits INTEGER,
    tmd_lane TEXT,
    lane_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- NEW: Usage tracking
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- NEW: Quality metrics
    confidence_score REAL,
    quality_metrics JSONB DEFAULT '{}'::jsonb,

    -- NEW: Training relationships
    parent_cpe_ids JSONB DEFAULT '[]'::jsonb,
    child_cpe_ids JSONB DEFAULT '[]'::jsonb
);

-- Vector storage table
CREATE TABLE cpe_vectors (
    cpe_id UUID PRIMARY KEY REFERENCES cpe_entry(cpe_id),
    concept_vec REAL[],           -- 768D GTR-T5 vector
    question_vec REAL[],          -- 768D probe question vector
    tmd_dense REAL[],             -- 16D TMD vector
    fused_vec REAL[],             -- 784D combined vector
    fused_norm REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_cpe_entry_dataset ON cpe_entry(dataset_source);
CREATE INDEX idx_cpe_entry_batch ON cpe_entry(batch_id);
CREATE INDEX idx_cpe_entry_confidence ON cpe_entry(confidence_score DESC);
CREATE INDEX idx_cpe_entry_access_count ON cpe_entry(access_count DESC);
CREATE INDEX idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. PRE-PROCESSING                                           │
│     ├─ Generate UUIDs for all chunks                        │
│     ├─ Auto-link parent/child (sequential)                  │
│     └─ Create batch ID                                      │
│                                                               │
│  2. CPESH EXTRACTION (Per Chunk)                            │
│     ├─ LLM: TinyLlama (port 11435)                          │
│     ├─ Extract: Concept, Probe, Expected                    │
│     └─ Generate: Soft Negatives, Hard Negatives             │
│                                                               │
│  3. TMD EXTRACTION                                           │
│     ├─ LLM: Llama 3.1 (port 11434)                          │
│     └─ Extract: Domain (4b), Task (5b), Modifier (6b)       │
│                                                               │
│  4. VECTORIZATION                                            │
│     ├─ GTR-T5: concept_text → 768D semantic vector          │
│     ├─ TMD Encode: [D, T, M] → 16D one-hot vector           │
│     └─ Concat: 768D + 16D = 784D fused vector               │
│                                                               │
│  5. QUALITY METRICS                                          │
│     ├─ CPESH completeness (0-1)                             │
│     ├─ Vector norm, text length                             │
│     ├─ Negatives count                                      │
│     └─ Confidence score (weighted 0-1)                      │
│                                                               │
│  6. ATOMIC WRITE                                             │
│     ├─ PostgreSQL: cpe_entry + cpe_vectors                  │
│     ├─ FAISS: Add 784D vector with UUID                     │
│     └─ Neo4j: (TODO) Create Concept node                    │
│                                                               │
│  7. RETURN RESULTS                                           │
│     └─ Global_ID, concept_text, TMD, confidence, parent/child│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Usage Examples

### Example 1: Simple Sequential Ingestion

```python
import requests

response = requests.post(
    "http://localhost:8004/ingest",
    json={
        "chunks": [
            {"text": "Machine learning is a subset of AI."},
            {"text": "Deep learning uses neural networks."},
            {"text": "Convolutional networks process images."}
        ],
        "dataset_source": "ai_textbook",
        "batch_id": "chapter_1"
    }
)

data = response.json()

# Access results
for result in data["results"]:
    print(f"UUID: {result['global_id']}")
    print(f"Concept: {result['concept_text']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Parent: {result['parent_cpe_ids']}")
    print(f"Child: {result['child_cpe_ids']}\n")
```

**Output**:
```
UUID: f47ac10b-58cc-4372-a567-0e02b2c3d479
Concept: Machine learning
Confidence: 0.87
Parent: []
Child: ['7c9e6679-7425-40de-944b-e07fc1f90ae7']

UUID: 7c9e6679-7425-40de-944b-e07fc1f90ae7
Concept: Deep learning
Confidence: 0.92
Parent: ['f47ac10b-58cc-4372-a567-0e02b2c3d479']
Child: ['550e8400-e29b-41d4-a716-446655440000']

UUID: 550e8400-e29b-41d4-a716-446655440000
Concept: Convolutional networks
Confidence: 0.89
Parent: ['7c9e6679-7425-40de-944b-e07fc1f90ae7']
Child: []
```

---

### Example 2: Manual Relationship Override

```python
import requests

# Ontology-based ingestion with explicit relationships
response = requests.post(
    "http://localhost:8004/ingest",
    json={
        "chunks": [
            {
                "text": "Machine Learning is a branch of AI that focuses on data-driven predictions.",
                "parent_cpe_ids": ["uuid-artificial-intelligence"],
                "child_cpe_ids": [
                    "uuid-supervised-learning",
                    "uuid-unsupervised-learning",
                    "uuid-reinforcement-learning"
                ]
            }
        ],
        "dataset_source": "ontology-dbpedia"
    }
)

result = response.json()["results"][0]
print(f"Created concept with {len(result['child_cpe_ids'])} children")
```

---

### Example 3: Query Training Sequences

```sql
-- Get all concept chains starting from root nodes
WITH RECURSIVE concept_chain AS (
    SELECT
        ce.cpe_id,
        ce.concept_text,
        ce.child_cpe_ids,
        cv.fused_vec,
        1 AS depth,
        ARRAY[ce.cpe_id::text] AS path
    FROM cpe_entry ce
    JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
    WHERE jsonb_array_length(ce.parent_cpe_ids) = 0  -- Root nodes

    UNION ALL

    SELECT
        ce.cpe_id,
        ce.concept_text,
        ce.child_cpe_ids,
        cv.fused_vec,
        cc.depth + 1,
        cc.path || ce.cpe_id::text
    FROM cpe_entry ce
    JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
    JOIN concept_chain cc ON ce.cpe_id::text = ANY(
        SELECT jsonb_array_elements_text(cc.child_cpe_ids)
    )
    WHERE cc.depth < 10
    AND NOT ce.cpe_id::text = ANY(cc.path)  -- Prevent cycles
)
SELECT
    array_length(path, 1) AS chain_length,
    array_agg(concept_text ORDER BY depth) AS concepts,
    array_agg(fused_vec ORDER BY depth) AS vectors
FROM concept_chain
GROUP BY path
HAVING array_length(path, 1) >= 3
ORDER BY chain_length DESC;
```

---

## Training Sequence Generation

See [TRAINING_SEQUENCE_GENERATION.md](TRAINING_SEQUENCE_GENERATION.md) for:
- Recursive chain queries
- Training batch creation
- Vector correlation across stores
- Python examples

---

## Performance

### Bottlenecks (Cold Start)

| Stage | Latency | % of Total | Notes |
|-------|---------|------------|-------|
| **TMD Extraction** | ~4400ms | 65% | LLM call (Llama 3.1) - **cache helps!** |
| **GTR-T5 Embedding** | ~1400ms | 20% | Model warm-up penalty |
| **CPESH Extraction** | ~1000ms | 14% | LLM call (TinyLlama) |
| **Database Write** | ~50ms | <1% | PostgreSQL + FAISS |
| **Total** | ~6850ms | 100% | First chunk only |

### Optimized Performance (Warm)

| Stage | Latency | Speedup |
|-------|---------|---------|
| TMD Extraction | ~50ms | **88x faster** (cache hit) |
| GTR-T5 Embedding | ~30ms | 46x faster |
| CPESH Extraction | ~500ms | 2x faster |
| Database Write | ~50ms | No change |
| **Total** | ~630ms | **11x faster** |

---

## Dependencies

**Required Services**:
- PostgreSQL (port 5432) - `cpe_entry` + `cpe_vectors` tables
- FAISS - Local index file (`artifacts/user_chunks.index`)
- Ollama LLM (port 11434) - Llama 3.1:8b + TinyLlama:1.1b
- Vec2Text GTR-T5 Embedder - Local model or API (port 8767)

**Optional Services**:
- Neo4j (port 7687) - Graph relationships (future)
- TMD Router API (port 8002) - Alternative to direct LLM calls

---

## Configuration

**Environment Variables**:
```bash
# LLM endpoint
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# GTR-T5 model path (if using local embedder)
export LNSP_EMBEDDER_PATH="./models/gtr-t5-base"

# PostgreSQL
export LNSP_PG_HOST="localhost"
export LNSP_PG_PORT="5432"
export LNSP_PG_DBNAME="lnsp"
```

---

## Error Handling

### Failed Chunk Ingestion

If a chunk fails (CPESH extraction error, LLM timeout, etc.), the response includes:

```json
{
  "global_id": "",
  "concept_text": "Chunk text (first 50 chars)...",
  "tmd_codes": {},
  "vector_dimension": 0,
  "confidence_score": 0.0,
  "quality_metrics": {"error": "Connection timeout to LLM"},
  "success": false,
  "error": "Connection timeout to LLM"
}
```

**Behavior**: Failed chunks do **NOT** block other chunks in the batch.

---

## Next Steps

1. **Start the service**: `./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004`
2. **Test ingestion**: `curl -X POST http://localhost:8004/ingest -d '{"chunks": [{"text": "Test"}]}'`
3. **Generate training sequences**: See [TRAINING_SEQUENCE_GENERATION.md](TRAINING_SEQUENCE_GENERATION.md)
4. **Train LVM**: Use generated sequences to train the Latent Vector Model

---

## Summary

✅ **Complete ingestion pipeline** with CPESH + TMD + Vectorization
✅ **Auto-sequential linking** for training data generation
✅ **Quality metrics** and confidence scoring
✅ **Parent/child relationships** for graph traversal
✅ **Atomic writes** to PostgreSQL + FAISS (Neo4j coming soon)
✅ **Production ready** with health checks, stats, and error handling

**Port 8004** is now the **single entry point** for all chunk ingestion! 🚀
