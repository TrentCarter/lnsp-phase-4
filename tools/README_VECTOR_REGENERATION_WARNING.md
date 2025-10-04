# ⚠️ CRITICAL WARNING: Vector Regeneration Scripts

## 🚨 DO NOT USE THESE SCRIPTS

The scripts in this directory (`regenerate_all_tmd_vectors.py`, `fix_ontology_tmd_*.py`) are **DANGEROUS** and will **BREAK GraphRAG** by creating data inconsistency between:

- PostgreSQL (cpe_entry, cpe_vectors)
- Neo4j (Concept nodes, relationships)
- FAISS (vector index)

## Why These Scripts Are Broken

These scripts:
1. **Only update PostgreSQL** - they regenerate vectors in `cpe_vectors` table
2. **Do NOT update Neo4j** - graph nodes become out of sync
3. **Do NOT update FAISS** - vector index becomes stale
4. **Break GraphRAG** - graph relationships no longer match vector positions

## What Happens If You Run Them

```
BEFORE (synchronized):
PostgreSQL:  concept[0] = "oxidoreductase activity"
Neo4j:       Concept[0] = "oxidoreductase activity"
FAISS:       vector[0]  = embedding("oxidoreductase activity")

AFTER running regenerate script (BROKEN):
PostgreSQL:  concept[0] = "NEW CONCEPT" (regenerated)
Neo4j:       Concept[0] = "oxidoreductase activity" (STALE!)
FAISS:       vector[0]  = embedding("oxidoreductase activity") (STALE!)

Result: GraphRAG retrieves neighbors for "oxidoreductase" but vectors
        are for "NEW CONCEPT" → 0% accuracy
```

## The ONLY Correct Way to Update Vectors

**RE-INGEST EVERYTHING**:

```bash
# 1. Clear ALL data stores
psql lnsp -c "TRUNCATE cpe_entry, cpe_vectors CASCADE;"
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;"
rm artifacts/*.index artifacts/*.npz

# 2. Run ATOMIC ingestion (writes to ALL stores)
./scripts/ingest_10k.sh --write-pg --write-neo4j

# 3. Build FAISS index from the same data
make build-faiss ARGS="--type ivf_flat --metric ip"
```

## Why Atomic Ingestion Is Required

GraphRAG depends on **three data stores being EXACTLY synchronized**:

| Store | Purpose | Must Match |
|-------|---------|------------|
| PostgreSQL | Concept metadata, CPE entries | ✓ Concept texts, TMD codes |
| Neo4j | Graph relationships | ✓ Concept texts, edges |
| FAISS | Vector search | ✓ Vector positions = concept order |

If **any ONE** gets out of sync, GraphRAG breaks completely.

## How We Got Into This Mess (Oct 2, 2025)

1. **Oct 1**: Ran `ingest_10k.sh` → Created FactoidWiki data in PostgreSQL + Neo4j
2. **Oct 2**: Someone ran `regenerate_all_tmd_vectors.py` → Updated PostgreSQL vectors ONLY
3. **Result**: Neo4j has "Moroccan royal family", vectors have "oxidoreductase activity"
4. **GraphRAG**: 0 neighbors found because concepts don't match

## Action Items

- [ ] **DEPRECATE** these regeneration scripts (move to `_deprecated/`)
- [ ] **ENFORCE** atomic writes in `ingest_factoid.py` (all-or-nothing)
- [ ] **UPDATE** all docs to emphasize synchronized ingestion
- [ ] **ADD** verification step to check PostgreSQL ↔ Neo4j ↔ FAISS sync

---

**Bottom line**: If you need to update vectors, **RE-INGEST**. There are no shortcuts.
