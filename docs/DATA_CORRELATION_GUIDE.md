# Data Correlation Guide

**Critical Rule**: ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION

This document explains Rule 4 from CLAUDE.md: Why unique IDs are critical and how to implement them correctly.

---

## The Problem: Data Without Correlation

**Scenario**: You have 339,615 Wikipedia concepts stored across three systems:
- PostgreSQL (concept text, CPESH negatives, metadata)
- Neo4j (graph relationships)
- FAISS (768D vectors)

**Question**: When you retrieve vector #12345 from FAISS, how do you:
1. Find the corresponding concept text in PostgreSQL?
2. Traverse graph relationships in Neo4j?
3. Get CPESH negatives for reranking?

**Answer**: You can't... unless you have a **unique ID** that links all three stores!

---

## The Solution: CPE ID (Concept Primary Embedding ID)

### What is a CPE ID?

A **CPE ID** is a unique identifier (UUID or sequential ID) that links the same concept across all data stores.

**Example**:
```
CPE ID: cpe_550e8400-e29b-41d4-a716-446655440000

PostgreSQL:
  cpe_entry.id = cpe_550e8400-e29b-41d4-a716-446655440000
  cpe_entry.concept_text = "Artificial Intelligence"
  cpe_entry.cpesh_negatives = ["Machine Learning", "Deep Learning", ...]

Neo4j:
  (:Concept {id: "cpe_550e8400-e29b-41d4-a716-446655440000", name: "Artificial Intelligence"})
  -[:RELATED_TO]-> (:Concept {name: "Machine Learning"})

FAISS NPZ:
  cpe_ids[42] = "cpe_550e8400-e29b-41d4-a716-446655440000"
  vectors[42] = [0.123, -0.456, ..., 0.789]  # 768D vector
  concept_texts[42] = "Artificial Intelligence"
```

---

## NPZ File Requirements

### Mandatory Arrays

Every NPZ file MUST include these three arrays:

```python
import numpy as np

# Save NPZ with correlation data
np.savez(
    "artifacts/my_vectors.npz",
    cpe_ids=cpe_ids,              # Array of CPE ID strings (length N)
    concept_texts=concept_texts,  # Array of concept text strings (length N)
    vectors=vectors                # Array of 768D vectors (shape: N x 768)
)

# Load NPZ and verify correlation
data = np.load("artifacts/my_vectors.npz")
assert len(data["cpe_ids"]) == len(data["vectors"])
assert len(data["concept_texts"]) == len(data["vectors"])
```

### Example: Correct NPZ Creation

```python
from uuid import uuid4
import numpy as np

# Generate data with unique IDs
cpe_ids = []
concept_texts = []
vectors = []

for concept in concepts:
    # 1. Generate unique CPE ID
    cpe_id = f"cpe_{uuid4()}"

    # 2. Store in PostgreSQL
    cursor.execute("""
        INSERT INTO cpe_entry (id, concept_text, cpesh_negatives)
        VALUES (%s, %s, %s)
    """, (cpe_id, concept.text, concept.cpesh_negatives))

    # 3. Store in Neo4j
    neo4j_session.run("""
        CREATE (c:Concept {id: $id, name: $name})
    """, id=cpe_id, name=concept.text)

    # 4. Add to NPZ arrays
    cpe_ids.append(cpe_id)
    concept_texts.append(concept.text)
    vectors.append(concept.vector)

# 5. Save NPZ with correlation data
np.savez(
    "artifacts/my_vectors.npz",
    cpe_ids=np.array(cpe_ids),
    concept_texts=np.array(concept_texts),
    vectors=np.array(vectors)
)
```

---

## PostgreSQL Schema

### Required Table Structure

```sql
-- Main concept entry table
CREATE TABLE cpe_entry (
    id TEXT PRIMARY KEY,                    -- CPE ID (UUID)
    concept_text TEXT NOT NULL,             -- Original concept text
    cpesh_negatives JSONB,                  -- Hard negatives for reranking
    source TEXT,                            -- Source dataset (e.g., "wikipedia")
    article_id TEXT,                        -- Source article ID
    chunk_index INTEGER,                    -- Chunk position in article
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector storage table (optional, can use NPZ instead)
CREATE TABLE cpe_vectors (
    cpe_id TEXT PRIMARY KEY REFERENCES cpe_entry(id),
    concept_vec JSONB NOT NULL,             -- 768D vector as JSON array
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookup
CREATE INDEX idx_cpe_entry_source ON cpe_entry(source);
CREATE INDEX idx_cpe_entry_article ON cpe_entry(article_id);
```

### Example Queries

```sql
-- Find concept by CPE ID
SELECT concept_text, cpesh_negatives
FROM cpe_entry
WHERE id = 'cpe_550e8400-e29b-41d4-a716-446655440000';

-- Find all concepts from an article
SELECT id, concept_text
FROM cpe_entry
WHERE article_id = 'wiki_12345'
ORDER BY chunk_index;

-- Get CPESH negatives for reranking
SELECT cpesh_negatives
FROM cpe_entry
WHERE id = ANY($1);  -- Array of CPE IDs from FAISS retrieval
```

---

## Neo4j Schema

### Required Node Structure

```cypher
// Create concept node with CPE ID
CREATE (c:Concept {
    id: "cpe_550e8400-e29b-41d4-a716-446655440000",
    name: "Artificial Intelligence",
    source: "wikipedia",
    article_id: "wiki_12345"
})

// Create relationships
MATCH (c1:Concept {id: "cpe_550e8400-e29b-41d4-a716-446655440000"})
MATCH (c2:Concept {id: "cpe_661f9511-f3ac-52e5-b827-557766551111"})
CREATE (c1)-[:RELATED_TO]->(c2)
```

### Example Queries

```cypher
// Find concept by CPE ID
MATCH (c:Concept {id: "cpe_550e8400-e29b-41d4-a716-446655440000"})
RETURN c.name, c.source

// Find neighbors
MATCH (c:Concept {id: "cpe_550e8400-e29b-41d4-a716-446655440000"})-[:RELATED_TO]-(neighbor)
RETURN neighbor.id, neighbor.name

// 6-degree shortcuts (0.5-3% of edges)
MATCH (c1:Concept {id: "cpe_550e8400-e29b-41d4-a716-446655440000"})
MATCH (c2:Concept {id: "cpe_772g0622-g4bd-63f6-c938-668877662222"})
WHERE shortestPath((c1)-[*..6]-(c2)) IS NOT NULL
CREATE (c1)-[:SHORTCUT {degrees: 6}]->(c2)
```

---

## FAISS Retrieval with Correlation

### Retrieval Flow

```python
import numpy as np
import faiss

# 1. Load NPZ with correlation data
data = np.load("artifacts/my_vectors.npz")
cpe_ids = data["cpe_ids"]
concept_texts = data["concept_texts"]
vectors = data["vectors"]

# 2. Build FAISS index
index = faiss.IndexFlatIP(768)  # Inner product (cosine similarity)
index.add(vectors)

# 3. Retrieve nearest neighbors
query_vector = encode_text("What is AI?")
scores, indices = index.search(query_vector, k=50)

# 4. Get CPE IDs for retrieved vectors
retrieved_cpe_ids = [cpe_ids[idx] for idx in indices[0]]

# 5. Query PostgreSQL for concept text + CPESH negatives
import psycopg2
conn = psycopg2.connect("dbname=lnsp")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, concept_text, cpesh_negatives
    FROM cpe_entry
    WHERE id = ANY(%s)
""", (retrieved_cpe_ids,))

results = cursor.fetchall()

# 6. Query Neo4j for graph context
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687")

with driver.session() as session:
    for cpe_id in retrieved_cpe_ids:
        neighbors = session.run("""
            MATCH (c:Concept {id: $id})-[:RELATED_TO]-(neighbor)
            RETURN neighbor.name
        """, id=cpe_id)
        print(f"Neighbors of {cpe_id}: {[n['neighbor.name'] for n in neighbors]}")
```

---

## Verification Commands

### Check PostgreSQL

```bash
# Count entries
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"

# Check CPE ID format
psql lnsp -c "SELECT id, concept_text FROM cpe_entry LIMIT 5;"

# Verify no duplicates
psql lnsp -c "SELECT id, COUNT(*) FROM cpe_entry GROUP BY id HAVING COUNT(*) > 1;"
```

### Check Neo4j

```bash
# Count nodes
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN COUNT(c)"

# Check CPE ID format
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN c.id, c.name LIMIT 5"

# Verify no duplicates
cypher-shell -u neo4j -p password "MATCH (c:Concept) WITH c.id AS id, COUNT(*) AS cnt WHERE cnt > 1 RETURN id, cnt"
```

### Check NPZ Files

```bash
# Verify NPZ structure
python -c "
import numpy as np
data = np.load('artifacts/my_vectors.npz')
print('Arrays:', list(data.keys()))
print('CPE IDs:', len(data['cpe_ids']))
print('Vectors:', data['vectors'].shape)
print('Concept Texts:', len(data['concept_texts']))
assert len(data['cpe_ids']) == len(data['vectors'])
assert len(data['concept_texts']) == len(data['vectors'])
print('✓ Correlation valid')
"
```

### Check Cross-Store Consistency

```python
import numpy as np
import psycopg2
from neo4j import GraphDatabase

# Load NPZ
data = np.load("artifacts/my_vectors.npz")
npz_cpe_ids = set(data["cpe_ids"])

# Check PostgreSQL
conn = psycopg2.connect("dbname=lnsp")
cursor = conn.cursor()
cursor.execute("SELECT id FROM cpe_entry")
pg_cpe_ids = set(row[0] for row in cursor.fetchall())

# Check Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687")
with driver.session() as session:
    result = session.run("MATCH (c:Concept) RETURN c.id")
    neo4j_cpe_ids = set(record["c.id"] for record in result)

# Verify consistency
print(f"NPZ CPE IDs: {len(npz_cpe_ids)}")
print(f"PostgreSQL CPE IDs: {len(pg_cpe_ids)}")
print(f"Neo4j CPE IDs: {len(neo4j_cpe_ids)}")

missing_in_pg = npz_cpe_ids - pg_cpe_ids
missing_in_neo4j = npz_cpe_ids - neo4j_cpe_ids

if missing_in_pg:
    print(f"⚠️  {len(missing_in_pg)} CPE IDs in NPZ but not in PostgreSQL!")
if missing_in_neo4j:
    print(f"⚠️  {len(missing_in_neo4j)} CPE IDs in NPZ but not in Neo4j!")

if not missing_in_pg and not missing_in_neo4j:
    print("✓ All stores synchronized!")
```

---

## Common Mistakes

### ❌ WRONG: No CPE IDs

```python
# BAD: Only storing vectors, no correlation
np.savez("vectors.npz", vectors=vectors)

# PROBLEM: How do you find concept text or graph relationships?
# Answer: You can't! Data is orphaned.
```

### ❌ WRONG: Index Position as ID

```python
# BAD: Using FAISS index position as ID
concept_id = 42  # Index position in FAISS

# PROBLEM: If you rebuild FAISS index, positions change!
# Answer: Use UUIDs that don't change.
```

### ❌ WRONG: Different IDs in Different Stores

```python
# BAD: Different ID schemes
postgres_id = "pg_12345"       # PostgreSQL auto-increment
neo4j_id = "neo4j_uuid_abc"    # Neo4j UUID
faiss_index = 42               # FAISS position

# PROBLEM: Cannot correlate across stores!
# Answer: Use ONE CPE ID everywhere.
```

### ✅ CORRECT: Single CPE ID Everywhere

```python
# GOOD: Same CPE ID in all stores
cpe_id = f"cpe_{uuid4()}"

# PostgreSQL
cursor.execute("INSERT INTO cpe_entry (id, ...) VALUES (%s, ...)", (cpe_id, ...))

# Neo4j
session.run("CREATE (c:Concept {id: $id, ...})", id=cpe_id)

# FAISS NPZ
cpe_ids.append(cpe_id)
vectors.append(vector)
```

---

## Migration: Fixing Orphaned Data

If you have data without CPE IDs, here's how to fix it:

```python
import numpy as np
import psycopg2
from uuid import uuid4

# 1. Load orphaned vectors
data = np.load("artifacts/old_vectors.npz")
vectors = data["vectors"]
concept_texts = data.get("concept_texts", [])

# 2. Generate CPE IDs retroactively
cpe_ids = [f"cpe_{uuid4()}" for _ in range(len(vectors))]

# 3. Update PostgreSQL
conn = psycopg2.connect("dbname=lnsp")
cursor = conn.cursor()

for i, cpe_id in enumerate(cpe_ids):
    concept_text = concept_texts[i] if i < len(concept_texts) else f"Unknown_{i}"
    cursor.execute("""
        INSERT INTO cpe_entry (id, concept_text)
        VALUES (%s, %s)
    """, (cpe_id, concept_text))

conn.commit()

# 4. Save corrected NPZ
np.savez(
    "artifacts/corrected_vectors.npz",
    cpe_ids=np.array(cpe_ids),
    concept_texts=np.array(concept_texts),
    vectors=vectors
)

print(f"✓ Migrated {len(cpe_ids)} concepts with CPE IDs")
```

---

## Summary

**Rule 4: ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION**

✅ **DO**:
- Generate CPE IDs (UUIDs) for every concept
- Store CPE ID in PostgreSQL, Neo4j, and FAISS NPZ
- Include `cpe_ids`, `concept_texts`, `vectors` in every NPZ file
- Verify cross-store consistency regularly

❌ **DON'T**:
- Use FAISS index positions as IDs (they change!)
- Use different ID schemes in different stores
- Save NPZ files without `cpe_ids` array
- Skip cross-store verification

**Without unique IDs, retrieval is impossible!**
