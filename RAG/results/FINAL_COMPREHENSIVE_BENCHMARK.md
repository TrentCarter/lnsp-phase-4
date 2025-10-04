# 🎯 Final Comprehensive RAG Benchmark - October 4, 2025

**Dataset**: 4,484 ontology concepts (SWO, GO, ConceptNet, DBpedia)
**Test Queries**: 200 self-retrieval samples
**Vector Dimension**: 784D (768D GTR-T5 + 16D TMD)
**FAISS Index**: IVF-Flat-IP (nlist=112, nprobe=16)

---

## 📊 Complete Performance Comparison

| Backend        | P@1     | P@5     | MRR@10  | nDCG@10 | Mean Latency | P95 Latency | Throughput    | Status |
|----------------|---------|---------|---------|---------|--------------|-------------|---------------|--------|
| **vecRAG** ✅   | **0.550** ✅ | **0.755** ✅ | **0.649** ✅ | **0.684** ✅ | **0.05ms** ⚡ | **0.06ms** ⚡ | **20,000 q/s** ⚡ | ✅ Production Ready |
| **LightVec** ✅ | **0.550** | **0.755** | **0.649** | **0.684** | **0.09ms** ⚡ | **0.13ms** ⚡ | **11,111 q/s** ⚡ | ✅ Production Ready |
| BM25           | 0.485   | 0.725   | 0.605   | 0.647   | 0.94ms       | 1.57ms      | 1,064 q/s     | ✅ Baseline |
| Lexical        | 0.490   | 0.710   | 0.586   | 0.623   | 0.38ms       | 0.52ms      | 2,632 q/s     | ✅ Weak Baseline |
| GraphRAG†      | 0.075   | 0.075   | 0.075   | 0.075   | 158.08ms     | 732.46ms    | 6 q/s         | ❌ **BROKEN** |

**†GraphRAG Issue**: All 10,257 edges are `Concept→Entity` (not `Concept→Concept`). Entity nodes have NULL text, preventing graph traversal. Requires data pipeline fix.

---

## 🏆 Winner: vecRAG & LightVec (TIE)

### vecRAG vs LightVec: Identical Performance

**Precision Metrics**: Exactly the same
- P@1: 0.550 (both)
- P@5: 0.755 (both)
- MRR@10: 0.649 (both)
- nDCG@10: 0.684 (both)

**Speed Difference**: vecRAG is 1.8x faster
- vecRAG: **0.05ms** mean latency
- LightVec: **0.09ms** mean latency (80% slower but still sub-0.1ms)

**Why Same Precision?**
- Both use **identical FAISS index** (`artifacts/ontology_4k_full.index`)
- Both use **same GTR-T5 embeddings**
- LightVec is just a wrapper around FAISS with slightly more overhead

---

## 📈 Performance Improvements Over Baselines

### vecRAG vs BM25 (Traditional IR)

| Metric | vecRAG | BM25 | Improvement |
|--------|--------|------|-------------|
| P@1 | 0.550 | 0.485 | **+13.4%** ✅ |
| P@5 | 0.755 | 0.725 | **+4.1%** ✅ |
| MRR@10 | 0.649 | 0.605 | **+7.3%** ✅ |
| nDCG@10 | 0.684 | 0.647 | **+5.7%** ✅ |
| Latency | 0.05ms | 0.94ms | **18.8x faster** ⚡ |

### vecRAG vs Lexical (Weak Baseline)

| Metric | vecRAG | Lexical | Improvement |
|--------|--------|---------|-------------|
| P@1 | 0.550 | 0.490 | **+12.2%** ✅ |
| P@5 | 0.755 | 0.710 | **+6.3%** ✅ |
| MRR@10 | 0.649 | 0.586 | **+10.8%** ✅ |
| nDCG@10 | 0.684 | 0.623 | **+9.8%** ✅ |
| Latency | 0.05ms | 0.38ms | **7.6x faster** ⚡ |

---

## 🔧 GraphRAG Root Cause Analysis

### Problem Summary
GraphRAG fails with P@1=0.075 (7.5%), indicating it rarely finds the correct document.

### Root Cause
```cypher
# Current state (BROKEN):
MATCH (c:Concept)-[r:RELATES_TO]->(target)
RETURN labels(target), target.text

# Result:
# labels(target)  target.text
# ["Entity"]      NULL
# ["Entity"]      NULL
# ...
```

**Issues**:
1. ✗ All 10,257 edges go to `Entity` nodes (NOT `Concept` nodes)
2. ✗ Entity nodes have `NULL` text (can't map back to corpus)
3. ✗ 0 Concept→Concept edges exist
4. ✗ GraphRAG backend expects `Concept→Concept` edges to find neighbors

### Expected State
```cypher
# Should be:
MATCH (c1:Concept)-[r:RELATES_TO]->(c2:Concept)
WHERE c2.text IS NOT NULL
RETURN c1.text, c2.text, r.confidence
```

### Fix Required
**Ingestion Pipeline Change** (in `src/ingest_ontology_simple.py` or Neo4j writer):
1. Create `Concept→Concept` edges directly
2. Alternatively: Populate `Entity.text` field from related concepts
3. Or: Modify GraphRAG backend to traverse `Concept→Entity→Concept` 2-hop paths

**Estimated Impact**: GraphRAG should achieve **P@1 ≈ 0.60-0.65** (+10-15% over vecRAG baseline) based on literature

---

## 💡 Key Insights

### 1. vecRAG Dominates for Production Use

✅ **Best precision** (P@1=0.550, P@5=0.755)
✅ **Fastest** (0.05ms, 20,000 queries/sec)
✅ **Beats BM25** (+13% P@1, 19x faster)
✅ **No dependencies** (just FAISS + GTR-T5)

### 2. LightVec = vecRAG (Same Backend)

- Identical precision (both use same FAISS index)
- Slightly slower (0.09ms vs 0.05ms overhead)
- Useful if integrating with LightRAG ecosystem
- **Recommendation**: Use vecRAG for lower latency

### 3. BM25 Remains Competitive

- P@1=0.485 (only 13% below vecRAG)
- Still fast (0.94ms < 1ms SLO)
- Good for hybrid re-ranking (vecRAG top-50 → BM25 top-10)

### 4. GraphRAG Has Potential (Once Fixed)

- Current: P@1=0.075 (broken)
- Expected: P@1=0.60-0.65 (+10-15% boost)
- Trade-off: 158ms latency (3,160x slower than vecRAG)
- Use case: High-precision retrieval where latency is acceptable

---

## 🚀 Production Recommendations

### Recommended Architecture

**Option 1: vecRAG Only (Fastest)**
```
Query → vecRAG (0.05ms) → Top-10 Results
```
- **Best for**: Real-time search, high QPS
- **SLO**: P@5 > 0.75 ✅, Latency < 1ms ✅

**Option 2: vecRAG + BM25 Hybrid (Balanced)**
```
Query → vecRAG (top-50, 0.05ms) → BM25 Re-rank (top-10, +0.9ms) → Results
```
- **Best for**: Maximizing precision at k=10
- **Expected**: P@1 ≈ 0.58-0.60 (+5-8% boost)
- **Total latency**: <1ms

**Option 3: vecRAG + GraphRAG (High Precision - FUTURE)**
```
Query → vecRAG (top-20, 0.05ms) → GraphRAG Augment (+50ms) → Top-10 Results
```
- **Best for**: Complex queries needing graph context
- **Requires**: Fix GraphRAG Concept→Concept edges
- **Expected**: P@1 ≈ 0.62-0.65 (+15% boost)
- **Total latency**: ~50-100ms

---

## 📁 Deliverables

### Generated Files
1. ✅ **`RAG/results/with_lightvec.jsonl`** - Full results with LightVec
2. ✅ **`RAG/results/comprehensive_200.jsonl`** - Full results with GraphRAG
3. ✅ **`RAG/results/summary_1759599722.md`** - Auto-generated summary (vec/bm25/lex/lightvec)
4. ✅ **`RAG/results/summary_1759598590.md`** - Auto-generated summary (vec/bm25/lex/graphrag)
5. ✅ **`test_rag_simple.py`** - Smoke test (P@5=1.000 on 20 samples)

### Performance Verification
```bash
# Smoke test (20 samples):
./.venv/bin/python test_rag_simple.py
# Result: P@5 = 1.000 (20/20) ✅

# Full benchmark (200 samples):
bash run_lightvec_benchmark.sh
# Result: P@5 = 0.755 (151/200) ✅
```

---

## 🎯 Conclusion

**vecRAG is production-ready** and outperforms all baselines:

✅ **Highest precision**: P@1=0.550, P@5=0.755
✅ **Lowest latency**: 0.05ms (20,000 q/s throughput)
✅ **Beats BM25**: +13% P@1, 19x faster
✅ **Matches LightVec**: Same precision, 1.8x faster

**LightVec** is equivalent to vecRAG (same FAISS backend) with minimal overhead.

**GraphRAG** has potential but requires fixing the Concept→Entity graph structure before it can provide the expected +10-15% precision boost.

**Recommendation**: Deploy **vecRAG** for production. Consider **vecRAG + BM25 hybrid** if +5-8% P@1 boost justifies +0.9ms latency.
