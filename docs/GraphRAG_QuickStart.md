# GraphRAG Quick Start Guide

**TL;DR**: Ensure data is synchronized, then run `./scripts/run_graphrag_benchmark.sh`

⚠️ **CRITICAL**: See [CRITICAL_GraphRAG_Data_Synchronization.md](CRITICAL_GraphRAG_Data_Synchronization.md) first!

---

## What is GraphRAG?

GraphRAG = **vecRAG + Neo4j knowledge graph** for better retrieval

```
Pure vecRAG:       Query → FAISS vector search → Top-K results
                   P@1: 54.4%

GraphRAG:          Query → FAISS → Expand via graph → RRF fusion → Top-K results
                   P@1: 60-70% (expected)
```

**Key Innovation**: Uses concept relationships from Neo4j to improve ranking

---

## Prerequisites (CRITICAL!)

```bash
# 1. Verify data synchronization
./scripts/verify_data_sync.sh

# If out of sync, RE-INGEST:
./scripts/ingest_10k.sh          # Writes to PostgreSQL + Neo4j + FAISS atomically
make build-faiss                  # Builds FAISS index from NPZ

# 2. Verify counts match
psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;"
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN count(c)" --format plain | tail -1
# Numbers MUST be identical!
```

## 30-Second Test (After Sync Verification)

```bash
# 1. Run benchmark
./scripts/run_graphrag_benchmark.sh

# 2. View results (5 minutes later)
cat RAG/results/summary_*.md | tail -20
```

---

## What Gets Tested

| Backend | Description | Expected P@1 |
|---------|-------------|--------------|
| **vec** | Baseline vecRAG (no graph) | 54.4% |
| **bm25** | Traditional keyword search | 49.4% |
| **graphrag_local** | vecRAG + 1-hop neighbors | **~60%** |
| **graphrag_global** | vecRAG + graph walks | **~58%** |
| **graphrag_hybrid** | vecRAG + both modes | **~65-70%** |

---

## How It Works

### 1. Local Context (1-hop)
```cypher
// Find direct neighbors
MATCH (c:Concept {text: "photosynthesis"})-[:RELATES_TO]->(neighbor)
RETURN neighbor.text, r.confidence
```
**Result**: `chlorophyll (0.85)`, `light reactions (0.78)`, ...

### 2. Global Context (graph walks)
```cypher
// Find concepts via shortcuts
MATCH (c:Concept {text: "photosynthesis"})-[:SHORTCUT_6DEG*1..2]-(neighbor)
RETURN neighbor.text, length(path)
```
**Result**: `cellular respiration (2-hop)`, `ATP synthesis (3-hop)`, ...

### 3. Hybrid Fusion (RRF)
```python
# Combine vector rank + graph confidence
score = (1 / (60 + vector_rank)) + (graph_confidence * 0.5)
```

---

## Graph Data Available

**Neo4j contains:**
- ✅ 4,993 Concept nodes (from ontology ingestion)
- ✅ 7,446 Entity nodes (from LightRAG extraction)
- ✅ 10,070 RELATES_TO edges (concept↔entity relationships)
- ✅ 34 SHORTCUT_6DEG edges (6-degree concept paths)

**Example relationship:**
```json
{
  "subj": "material entity",
  "pred": "is a type of",
  "obj": "independent continuant",
  "confidence": 0.588
}
```

---

## Performance Expectations

| Metric | vecRAG | GraphRAG Hybrid | Improvement |
|--------|--------|-----------------|-------------|
| P@1 | 54.4% | **65-70%** | **+20%** |
| P@5 | 77.8% | **82-85%** | **+6%** |
| MRR@10 | 65.8% | **72-75%** | **+10%** |
| Latency | 0.04ms | **3-5ms** | 75-125x slower |

**Trade-off**: +15% accuracy for +100x latency (still faster than BM25!)

---

## Troubleshooting

**Neo4j not running?**
```bash
brew services start neo4j
# Wait 15 seconds
```

**Graph has no data?**
```bash
./scripts/ingest_all_ontologies.sh
```

**Import errors?**
```bash
./.venv/bin/pip install neo4j
```

---

## Next Steps

After running benchmark:

1. **Review results**: `cat RAG/results/summary_*.md`
2. **Compare metrics**: vecRAG vs GraphRAG modes
3. **If P@1 > 60%**: ✅ GraphRAG works! Proceed to LVM training
4. **If P@1 < 60%**: Debug graph quality (check edge confidences)

---

## Full Documentation

- **Implementation details**: `docs/GraphRAG_Implementation.md`
- **Benchmark code**: `RAG/bench.py` + `RAG/graphrag_backend.py`
- **GWOM design** (for LVM training): `docs/PRDs/PRD_GWOM_design_Options.md`

---

**Ready to test?** → `./scripts/run_graphrag_benchmark.sh`
