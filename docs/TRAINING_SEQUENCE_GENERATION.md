# Training Sequence Generation with Parent/Child Relationships

**Date**: 2025-10-09
**Purpose**: Enable LVM training by creating concept chains (sequences) from graph relationships

---

## Overview

The Chunk Ingestion API now supports **parent/child UUID tracking** to enable:
1. **Graph traversal** - Walk concept chains to create training sequences
2. **Vector correlation** - Link concepts across PostgreSQL, Neo4j, and FAISS
3. **Training data generation** - Create ordered sequences for LVM model training

---

## Database Schema

### PostgreSQL Schema

```sql
-- Parent/child relationship fields (added in migration 002)
ALTER TABLE cpe_entry
ADD COLUMN parent_cpe_ids JSONB DEFAULT '[]'::jsonb,  -- Array of parent UUIDs
ADD COLUMN child_cpe_ids JSONB DEFAULT '[]'::jsonb;   -- Array of child UUIDs

-- GIN indexes for efficient relationship lookups
CREATE INDEX idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);
```

### Running the Migration

```bash
# Apply the schema changes
psql lnsp < migrations/002_add_tracking_and_relationships.sql
```

---

## API Usage

### 1. Ingesting Chunks with Relationships

**Example: Sequential document chunks**

```bash
curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "source_document": "biology.pdf",
        "chunk_index": 0,
        "parent_cpe_ids": [],
        "child_cpe_ids": ["uuid-chunk-1"]
      },
      {
        "text": "Chlorophyll is the pigment responsible for capturing light energy during photosynthesis.",
        "source_document": "biology.pdf",
        "chunk_index": 1,
        "parent_cpe_ids": ["uuid-chunk-0"],
        "child_cpe_ids": ["uuid-chunk-2"]
      },
      {
        "text": "The light-dependent reactions occur in the thylakoid membranes of chloroplasts.",
        "source_document": "biology.pdf",
        "chunk_index": 2,
        "parent_cpe_ids": ["uuid-chunk-1"],
        "child_cpe_ids": []
      }
    ],
    "dataset_source": "biology_course",
    "batch_id": "bio_chapter1"
  }'
```

**Example: Ontology relationships (from Neo4j)**

```bash
# Concept: "Machine Learning" with parent "Artificial Intelligence" and child "Deep Learning"
curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "text": "Machine learning is a subset of artificial intelligence...",
        "parent_cpe_ids": ["f47ac10b-58cc-4372-a567-0e02b2c3d479"],  # AI concept UUID
        "child_cpe_ids": [
          "7c9e6679-7425-40de-944b-e07fc1f90ae7",  # Deep Learning UUID
          "550e8400-e29b-41d4-a716-446655440000"   # Neural Networks UUID
        ]
      }
    ],
    "dataset_source": "ontology-dbpedia"
  }'
```

---

## Training Sequence Generation

### 2. Querying for Training Chains

**Query 1: Get all children of a concept (breadth-first)**

```sql
-- Find all direct children of a concept
SELECT
    ce.cpe_id,
    ce.concept_text,
    ce.parent_cpe_ids,
    ce.child_cpe_ids,
    cv.concept_vec,
    cv.tmd_dense,
    cv.fused_vec
FROM cpe_entry ce
JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
WHERE ce.parent_cpe_ids @> '["f47ac10b-58cc-4372-a567-0e02b2c3d479"]'::jsonb;
```

**Query 2: Walk concept chains (recursive)**

```sql
-- Recursive CTE to walk concept chains up to depth 5
WITH RECURSIVE concept_chain AS (
    -- Base case: start with a root concept
    SELECT
        ce.cpe_id,
        ce.concept_text,
        ce.parent_cpe_ids,
        ce.child_cpe_ids,
        1 AS depth,
        ARRAY[ce.cpe_id::text] AS path
    FROM cpe_entry ce
    WHERE ce.cpe_id = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'  -- Root concept

    UNION ALL

    -- Recursive case: follow child relationships
    SELECT
        ce.cpe_id,
        ce.concept_text,
        ce.parent_cpe_ids,
        ce.child_cpe_ids,
        cc.depth + 1,
        cc.path || ce.cpe_id::text
    FROM cpe_entry ce
    JOIN concept_chain cc ON ce.cpe_id::text = ANY(
        SELECT jsonb_array_elements_text(cc.child_cpe_ids)
    )
    WHERE cc.depth < 5  -- Max depth
    AND NOT ce.cpe_id::text = ANY(cc.path)  -- Prevent cycles
)
SELECT * FROM concept_chain ORDER BY depth, cpe_id;
```

**Query 3: Get concept chains with vectors**

```sql
-- Get complete training sequences (concept chains with vectors)
WITH RECURSIVE concept_chain AS (
    SELECT
        ce.cpe_id,
        ce.concept_text,
        ce.child_cpe_ids,
        cv.fused_vec,  -- 784D vector
        1 AS depth,
        ARRAY[ce.cpe_id::text] AS path
    FROM cpe_entry ce
    JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
    WHERE ce.dataset_source LIKE 'ontology-%'
    AND jsonb_array_length(ce.parent_cpe_ids) = 0  -- Start with root nodes

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
    WHERE cc.depth < 10  -- Longer chains for training
    AND NOT ce.cpe_id::text = ANY(cc.path)
)
SELECT
    depth,
    array_length(path, 1) AS chain_length,
    path,
    array_agg(concept_text ORDER BY depth) AS concept_sequence,
    array_agg(fused_vec ORDER BY depth) AS vector_sequence
FROM concept_chain
GROUP BY depth, path
HAVING array_length(path, 1) >= 3  -- Only chains of 3+ concepts
ORDER BY chain_length DESC, depth;
```

---

## Python Training Data Generator

### 3. Generate Training Sequences

```python
#!/usr/bin/env python3
"""
Generate LVM training sequences from concept chains
"""
import psycopg2
import numpy as np
from typing import List, Dict, Tuple

def get_training_sequences(pg_conn, min_chain_length: int = 3, max_chains: int = 1000) -> List[Dict]:
    """
    Query PostgreSQL for concept chains and return training sequences.

    Returns:
        List of dicts with:
        - chain_id: str
        - concept_texts: List[str]
        - vector_sequence: np.ndarray (shape: [seq_len, 784])
        - chain_length: int
    """

    query = """
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
        WHERE ce.dataset_source LIKE 'ontology-%%'
        AND jsonb_array_length(ce.parent_cpe_ids) = 0

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
        AND NOT ce.cpe_id::text = ANY(cc.path)
    )
    SELECT
        array_to_string(path, '-') AS chain_id,
        array_agg(concept_text ORDER BY depth) AS concept_texts,
        array_agg(fused_vec ORDER BY depth) AS vector_sequence,
        array_length(path, 1) AS chain_length
    FROM concept_chain
    GROUP BY path
    HAVING array_length(path, 1) >= %s
    ORDER BY chain_length DESC
    LIMIT %s;
    """

    cursor = pg_conn.cursor()
    cursor.execute(query, (min_chain_length, max_chains))

    sequences = []
    for row in cursor.fetchall():
        chain_id, concept_texts, vector_sequence, chain_length = row

        # Convert PostgreSQL array to numpy
        vectors = np.array(vector_sequence, dtype=np.float32)

        sequences.append({
            "chain_id": chain_id,
            "concept_texts": concept_texts,
            "vector_sequence": vectors,  # Shape: [chain_length, 784]
            "chain_length": chain_length
        })

    cursor.close()
    return sequences


def prepare_lvm_training_batch(sequence: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a concept chain into LVM training batch.

    Input:
        vector_sequence: [v1, v2, v3, v4]  (shape: [4, 784])

    Output:
        X: [v1, v2, v3]  (shape: [3, 784])  # Input sequences
        y: [v2, v3, v4]  (shape: [3, 784])  # Target vectors

    Returns:
        (X_batch, y_batch)
    """
    vectors = sequence["vector_sequence"]  # Shape: [seq_len, 784]

    # Sliding window: predict next vector
    X = vectors[:-1]  # All except last
    y = vectors[1:]   # All except first

    return X, y


# Example usage
if __name__ == "__main__":
    import psycopg2

    conn = psycopg2.connect(
        dbname="lnsp",
        user="postgres",
        password="password",
        host="localhost",
        port=5432
    )

    # Get training sequences
    sequences = get_training_sequences(conn, min_chain_length=3, max_chains=100)

    print(f"Found {len(sequences)} training sequences")

    # Show example
    if sequences:
        seq = sequences[0]
        print(f"\\nExample chain (length {seq['chain_length']}):")
        for i, text in enumerate(seq["concept_texts"]):
            print(f"  {i}: {text}")

        X, y = prepare_lvm_training_batch(seq)
        print(f"\\nTraining batch shapes:")
        print(f"  X (input): {X.shape}")
        print(f"  y (target): {y.shape}")

    conn.close()
```

---

## Use Cases

### Document-Based Training Sequences

**Use case**: Fine-tune LVM on document structure

1. Ingest document chunks in order with sequential parent/child links
2. Query for document-specific chains
3. Train LVM to predict next chunk vector given previous chunks

**Example**:
```
Chapter 1, Para 1 → Chapter 1, Para 2 → Chapter 1, Para 3
```

### Ontology-Based Training Sequences

**Use case**: Train LVM on ontology relationships

1. Extract relationships from Neo4j (parent/child concepts)
2. Ingest concepts with Neo4j-derived parent/child UUIDs
3. Query for ontology chains (e.g., `Software → Algorithm → Sorting → QuickSort`)
4. Train LVM to predict child concept vector given parent

**Example**:
```
Artificial Intelligence → Machine Learning → Deep Learning → CNN
```

### Hybrid Training Sequences

**Use case**: Mix document and ontology relationships

1. Link document chunks to ontology concepts
2. Create hybrid chains: `Document Chunk → Ontology Concept → Related Concept`
3. Train LVM to generalize across both structure types

---

## Best Practices

### 1. Avoid Cycles
Always check for cycles when building parent/child relationships:
```sql
-- Good: Path prevents cycles
WHERE NOT ce.cpe_id::text = ANY(cc.path)
```

### 2. Limit Chain Depth
Use depth limits to prevent infinite recursion:
```sql
WHERE cc.depth < 10
```

### 3. Validate UUIDs
Ensure parent/child UUIDs exist in the database before ingestion:
```python
def validate_relationships(parent_ids: List[str], child_ids: List[str]) -> bool:
    all_ids = parent_ids + child_ids
    query = "SELECT cpe_id FROM cpe_entry WHERE cpe_id = ANY(%s)"
    cursor.execute(query, (all_ids,))
    existing = {str(row[0]) for row in cursor.fetchall()}
    return set(all_ids) == existing
```

### 4. Index for Performance
Ensure GIN indexes are created:
```sql
CREATE INDEX idx_cpe_entry_parent_ids ON cpe_entry USING GIN(parent_cpe_ids);
CREATE INDEX idx_cpe_entry_child_ids ON cpe_entry USING GIN(child_cpe_ids);
```

---

## Testing

### Test 1: Simple Chain Ingestion

```bash
# Create a 3-concept chain
UUID1=$(uuidgen | tr '[:upper:]' '[:lower:]')
UUID2=$(uuidgen | tr '[:upper:]' '[:lower:]')
UUID3=$(uuidgen | tr '[:upper:]' '[:lower:]')

curl -X POST http://localhost:8004/ingest \
  -H "Content-Type: application/json" \
  -d "{
    \"chunks\": [
      {
        \"text\": \"Concept A is the root\",
        \"child_cpe_ids\": [\"$UUID2\"]
      },
      {
        \"text\": \"Concept B is the middle\",
        \"parent_cpe_ids\": [\"$UUID1\"],
        \"child_cpe_ids\": [\"$UUID3\"]
      },
      {
        \"text\": \"Concept C is the leaf\",
        \"parent_cpe_ids\": [\"$UUID2\"]
      }
    ],
    \"dataset_source\": \"test_chain\"
  }"
```

### Test 2: Query Chain

```sql
-- Verify chain was created
SELECT
    cpe_id,
    concept_text,
    parent_cpe_ids,
    child_cpe_ids
FROM cpe_entry
WHERE dataset_source = 'test_chain'
ORDER BY chunk_position->'index';
```

---

## Future Enhancements

### 1. Neo4j Integration
Auto-populate parent/child relationships from Neo4j graph:
```python
def sync_relationships_from_neo4j(concept_text: str) -> Tuple[List[str], List[str]]:
    """Query Neo4j for parent/child concepts and return UUIDs"""
    # Query: MATCH (c:Concept {text: $concept_text})-[:PARENT_OF]->(child:Concept)
    # Return child.cpe_id
    pass
```

### 2. Relationship Strength
Add weights to parent/child relationships:
```json
{
  "parent_cpe_ids": [
    {"uuid": "...", "strength": 0.9},
    {"uuid": "...", "strength": 0.7}
  ]
}
```

### 3. Training Data Export
Create CLI tool to export training sequences:
```bash
./tools/export_training_sequences.py \
  --min-length 3 \
  --max-chains 10000 \
  --output training_sequences.npz
```

---

## Summary

✅ **Parent/child UUID tracking** enables:
1. Graph traversal for training sequence generation
2. Vector correlation across PostgreSQL, Neo4j, FAISS
3. Flexible training data creation (documents, ontologies, or hybrid)

✅ **Database migration** adds:
- `parent_cpe_ids` JSONB array
- `child_cpe_ids` JSONB array
- GIN indexes for efficient lookups

✅ **API support** for:
- Ingesting chunks with relationships
- Recursive chain queries
- Training batch generation
