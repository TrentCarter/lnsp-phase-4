# GraphRAG Diagnosis - October 4, 2025

## Problem Statement
GraphRAG benchmark shows terrible performance (P@1=10%, P@5=40%) despite having a correctly structured Neo4j graph with 107,346 Concept→Concept edges.

## Investigation Timeline

### 1. Neo4j Edge Fix (Completed Successfully)
**Script:** `scripts/fix_neo4j_concept_edges.py`

**Results:**
- Fixed 107,346 edges from Entity→Concept to Concept→Concept
- Deleted 38 orphaned Entity nodes
- Graph structure now correct:
  - Concepts: 4,484
  - Concept→Concept edges: 107,346
  - Concept→Entity edges: 5,846 (remaining legitimate edges)

**Cypher Verification:**
```cypher
MATCH (c1:Concept)-[r]->(c2:Concept) RETURN count(r)
-- Result: 107,346 ✅

MATCH (c:Concept)-[r]->(e:Entity) RETURN count(r)
-- Result: 5,846 ✅
```

### 2. GraphRAG Benchmark After Fix
**Command:**
```bash
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
PYTHONPATH=. \
./.venv/bin/python RAG/bench.py \
  --dataset self --n 10 --topk 10 \
  --backends graphrag_hybrid \
  --out RAG/results/graphrag_test_10.jsonl
```

**Results:**
| Metric | Score |
|--------|-------|
| P@1 | 0.100 (10%) ❌ |
| P@5 | 0.400 (40%) ❌ |
| MRR@10 | 0.195 ❌ |
| nDCG@10 | 0.267 ❌ |
| Mean latency | 774ms |

**Expected:** P@1 ≈ 60-65% after fixing edges
**Actual:** P@1 = 10% (minimal improvement from 7.5%)

### 3. Root Cause Analysis

**Test 1: Manual GraphRAG Expansion**
```python
g = GraphRAGBackend()
test_concept = "material entity"

# 1-hop neighbors (RELATES_TO)
neighbors = g._get_1hop_neighbors(test_concept)
print(f"Found {len(neighbors)} neighbors")  # ✅ Found 10 neighbors

# Graph walks (SHORTCUT_6DEG)
walks = g._get_graph_walks(test_concept)
print(f"Found {len(walks)} walk targets")  # ❌ Found 0 walk targets
```

**Test 2: Direct Neo4j Query**
```cypher
MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r)
-- Result: 0 ❌
```

**ROOT CAUSE FOUND:** No SHORTCUT_6DEG edges exist in Neo4j graph!

## Why GraphRAG Is Failing

### 1. Missing Shortcuts (Critical)
GraphRAG relies on two types of edges:
1. ✅ **RELATES_TO** (direct 1-hop neighbors) - Working correctly (107K edges)
2. ❌ **SHORTCUT_6DEG** (long-range connections) - Missing entirely (0 edges)

The `graphrag_hybrid` mode tries to use both:
```python
# RAG/graphrag_backend.py:177-199
if mode in ("local", "hybrid"):
    neighbors = graphrag_backend._get_1hop_neighbors(concept_text)  # ✅ Works

if mode in ("global", "hybrid"):
    walks = graphrag_backend._get_graph_walks(concept_text)  # ❌ Returns empty
```

### 2. Bad Seed Concepts (Secondary Issue)
GraphRAG expands from top-5 vector results:
```python
# RAG/graphrag_backend.py:177
top_vec_texts = [concept_texts[i] for i in vec_idxs[:5]]
```

If vecRAG only achieves P@1=55%, then 45% of the time the seed concepts are wrong, leading to expansion in wrong direction.

## Solution Required

### Priority 1: Generate Shortcuts
Run the shortcut generation script:
```bash
./scripts/generate_6deg_shortcuts.sh
```

**Expected outcome:**
- Create ~100-300 SHORTCUT_6DEG edges (0.5-3% of total edges)
- Connect concepts within 6-degree separation
- Enable GraphRAG global context

### Priority 2: Test GraphRAG with Shortcuts
After generating shortcuts, re-run benchmark:
```bash
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 FAISS_NUM_THREADS=1
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
PYTHONPATH=. \
./.venv/bin/python RAG/bench.py \
  --dataset self --n 50 --topk 10 \
  --backends graphrag_hybrid \
  --out RAG/results/graphrag_with_shortcuts.jsonl
```

**Expected improvement:**
- P@1: 10% → 40-50% (4-5x improvement)
- P@5: 40% → 65-75% (1.6-1.9x improvement)

### Priority 3: Test Local Mode Only (Baseline)
Test GraphRAG with only RELATES_TO edges (no shortcuts):
```bash
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self --n 50 --topk 10 \
  --backends graphrag_local \
  --out RAG/results/graphrag_local_only.jsonl
```

This will show us the baseline performance with 1-hop expansion only.

## Current System State

### ✅ What's Working
1. Neo4j graph structure (107K edges)
2. Concept→Concept RELATES_TO edges
3. 1-hop neighbor expansion
4. RRF fusion logic

### ❌ What's Broken
1. No SHORTCUT_6DEG edges generated
2. Global context (graph walks) returns empty
3. GraphRAG hybrid mode degraded to local-only mode

### 🔧 What Needs Fixing
1. Generate shortcuts using existing script
2. Verify shortcuts are created correctly
3. Re-benchmark GraphRAG hybrid mode
4. Compare local vs hybrid vs global modes

## Technical Details

### GraphRAG Architecture (from code)
```
RAG/graphrag_backend.py
├── _get_1hop_neighbors()      # ✅ Works (RELATES_TO edges)
├── _get_graph_walks()         # ❌ Broken (no SHORTCUT_6DEG edges)
└── _rrf_fusion()             # ✅ Works (fusion logic correct)
```

### Neo4j Schema
```
Concept {
  text: string
  cpe_id: int
  tmdLane: int
  tmdBits: string
  laneIndex: int
}

RELATES_TO {
  confidence: float [0.0-1.0]
}

SHORTCUT_6DEG {
  # No properties (just connectivity)
}
```

## Next Steps

1. ✅ Diagnosis complete - root cause identified
2. ⏳ Generate shortcuts using `./scripts/generate_6deg_shortcuts.sh`
3. ⏳ Verify shortcuts created (should see ~100-300 edges)
4. ⏳ Re-run GraphRAG benchmark
5. ⏳ Compare local vs hybrid performance
6. ⏳ Document final results

## Lessons Learned

1. **Always verify edge types exist** - Don't assume scripts ran successfully
2. **Test components independently** - Manual testing revealed the issue quickly
3. **Check for missing prerequisites** - Shortcuts were never generated
4. **Performance depends on data quality** - Even correct code fails with missing data

## References

- Neo4j fix script: `scripts/fix_neo4j_concept_edges.py`
- GraphRAG backend: `RAG/graphrag_backend.py`
- Shortcut generation: `scripts/generate_6deg_shortcuts.sh`
- Benchmark script: `RAG/bench.py`
- Test results: `RAG/results/graphrag_test_10.jsonl`
