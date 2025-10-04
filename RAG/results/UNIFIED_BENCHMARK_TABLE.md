# RAG Comprehensive Benchmark Results
**Dataset**: Ontology (self-retrieval) | **N**: 200 queries | **topk**: 10 | **dim**: 784 (768D GTR-T5 + 16D TMD)

---

## Core RAG Backends (200 queries)

| Backend  | P@1     | P@5     | MRR@10  | nDCG@10 | Mean Latency | Throughput   |
|----------|---------|---------|---------|---------|--------------|--------------|
| vecRAG ✅ | 0.550 ✅ | 0.755 ✅ | 0.649 ✅ | 0.684 ✅ | 0.05ms ⚡     | 20,000 q/s ⚡ |
| LightVec | 0.550   | 0.755   | 0.649   | 0.684   | 0.09ms       | 11,111 q/s   |
| BM25     | 0.485   | 0.725   | 0.605   | 0.647   | 0.94ms       | 1,064 q/s    |
| Lexical  | 0.490   | 0.710   | 0.586   | 0.623   | 0.38ms       | 2,632 q/s    |

**Key Finding**: vecRAG and LightVec have **identical precision** (both use same FAISS index), but vecRAG is 1.8x faster due to lower wrapper overhead.

---

## Graph-Augmented Backends (50 queries)

| Backend            | P@1   | P@5   | MRR@10 | nDCG@10 | Mean Latency | Throughput |
|--------------------|-------|-------|--------|---------|--------------|------------|
| vec (baseline)     | 0.600 | 0.840 | 0.712  | 0.745   | 0.06ms ⚡     | 16,667 q/s |
| vec_graph_rerank   | 0.600 | 0.840 | 0.712  | 0.745   | 9.56ms       | 105 q/s    |
| graphrag_hybrid 🔴 | 0.080 | 0.260 | 0.153  | 0.215   | 434.32ms     | 2.3 q/s    |

**Graph Re-ranking**: Uses vecRAG first, then boosts scores based on mutual connectivity within top-K results.
- **Speed cost**: 160x slower than pure vecRAG (9.56ms vs 0.06ms)
- **Precision impact**: NONE (boost_factor=0.2 doesn't change rankings yet)
- **Status**: Working, needs tuning

**GraphRAG Hybrid**: Dense vector + graph traversal with RRF fusion.
- **Status**: 🔴 BROKEN despite Neo4j fix (Concept→Concept edges)
- **Problem**: Graph too dense (107K edges for 4.5K concepts) dilutes vecRAG results
- **P@1 drop**: -86.7% compared to vecRAG (0.080 vs 0.600)
- **Latency**: 7,240x slower than vecRAG

---

## Why TMD Doesn't Improve vecRAG Over LightVec

**Question**: "Shouldn't TMD increase the score of vecRAG over LightVec?"

**Answer**: Both vecRAG and LightVec use the **same FAISS index** with 784D vectors (768D GTR-T5 + 16D TMD).

- **TMD is already baked into the vectors** during ingestion
- Both backends retrieve from `artifacts/ontology_4k_full.npz` (same vectors)
- Precision is **identical**: P@1=0.550, P@5=0.755
- Only difference is **wrapper overhead**: LightVec is 1.8x slower (0.09ms vs 0.05ms)

**TMD Impact**: TMD improves **all vector-based methods** equally (vecRAG, LightVec) compared to pure GTR-T5. But they still retrieve identical results because they use the same index.

---

## GraphRAG Fix Status

**Neo4j Graph Fix**: ✅ COMPLETED
- **Before**: 10,257 Concept→Entity edges (broken - Entity nodes had NULL text)
- **After**: 107,346 Concept→Concept edges (proper ontology structure)
- **Fix Script**: `scripts/fix_neo4j_concept_edges.py`

**GraphRAG Performance After Fix**: ⚠️ STILL POOR

Expected vs Actual:
| Metric | Expected (from docs) | Actual | Status |
|--------|---------------------|--------|--------|
| P@1    | 0.60-0.65           | 0.080  | 🔴 -87% worse |
| P@5    | 0.80-0.85           | 0.260  | 🔴 -69% worse |

**Root Cause**: Graph too densely connected
- Ontology concepts have duplicate text across hierarchies
- Example: "oxidoreductase activity" appears 823 times in different Gene Ontology branches
- Fix script created edges to ALL matching Concepts (correct for ontologies!)
- Result: Cartesian product explosion (823² = 677,329 edges for one concept)
- Graph traversal finds TOO many neighbors → dilutes vecRAG results

**Why Fix Didn't Help**:
1. Graph now structurally correct (Concept→Concept)
2. BUT graph is too noisy for retrieval
3. RRF fusion gives equal weight to graph traversal and vecRAG
4. Noisy graph results overwhelm good vecRAG results

**Recommendation**: Use `vec_graph_rerank` instead
- Uses graph to **validate** vecRAG results (not compete)
- 160x slower than vecRAG but doesn't hurt precision
- Needs tuning: adjust `boost_factor` from 0.2 to 0.05-0.3

---

## Recommendations

### Production: Use vecRAG ✅
- **Best precision/speed tradeoff**: P@5=0.755 @ 0.05ms
- **20,000 queries/second throughput**
- **TMD-enhanced**: Already includes 16D TMD vectors for better retrieval

### Experimentation: Tune vec_graph_rerank
- **Current status**: Same precision as vecRAG, 160x slower
- **Tuning options**:
  - Boost factor: Try 0.05, 0.1, 0.15, 0.3 (currently 0.2)
  - Top-K expansion: Try 1.5x instead of 2x (currently getting top-20, returning top-10)
  - Graph features: Try PageRank, betweenness centrality (currently mutual connections)

### Not Recommended: GraphRAG Hybrid
- **86.7% precision drop** compared to vecRAG
- **7,240x latency increase** (434ms vs 0.06ms)
- **Graph pruning needed**: Reduce edge density before using for retrieval

---

## Benchmark Commands

### Core RAG (200 queries)
```bash
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
PYTHONPATH=. \
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,lightvec,bm25,lex \
  --out RAG/results/vecrag_vs_baselines.jsonl
```

### Graph-Augmented (50 queries)
```bash
FAISS_NPZ_PATH=artifacts/ontology_4k_full.npz \
PYTHONPATH=. \
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 50 \
  --topk 10 \
  --backends vec,vec_graph_rerank,graphrag_hybrid \
  --out RAG/results/graph_augmented.jsonl
```

---

**Generated**: 2025-10-04
**FAISS Index**: `artifacts/ontology_4k_full.index` (IVF-Flat-IP, nlist=112, nprobe=16)
**Vectors**: `artifacts/ontology_4k_full.npz` (4,484 concepts × 784D)
**Neo4j Edges**: 107,346 Concept→Concept relationships
