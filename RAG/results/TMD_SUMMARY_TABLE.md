# TMD Re-ranking Summary Results

## Benchmark Comparison (200 queries, top-k=10)

| Backend | P@1 | P@5 | P@10 | MRR | nDCG | Notes |
|---------|-----|-----|------|-----|------|-------|
| **vec (baseline)** | 91.5% | 95.6% | 97.0% | 0.9356 | 0.9652 | Dense vector retrieval only |
| **vec_tmd_rerank** | 94.5% | 97.5% | 98.5% | 0.9557 | 0.9775 | LLM-based query TMD, alpha=0.3 |
| **Improvement** | +3.0% | +2.0% | +1.5% | +0.020 | +0.012 | Absolute percentage points |

## Key Findings

1. **LLM-based TMD re-ranking improves all metrics**
   - P@1: 91.5% → 94.5% (+3.0 percentage points)
   - P@5: 95.6% → 97.5% (+2.0 percentage points)
   - Best improvement at top-1 position (most important for user experience)

2. **Current Configuration**
   - Alpha: 0.3 (TMD weight) / 0.7 (vector weight)
   - TMD extraction: LLM-based for queries, pattern-based for corpus
   - LLM: Ollama Llama 3.1:8b (local)

3. **Next Steps**
   - Tune alpha parameter (test 0.4, 0.5, 0.6)
   - Potentially re-ingest corpus with LLM-based TMD (~1.9 hours)

## Test Details

- **Dataset**: Self-retrieval (ontology concepts)
- **Query count**: 200
- **Top-k**: 10
- **Backend comparison**: Dense vecRAG vs TMD re-ranked vecRAG
- **Date**: 2025-10-04
- **Results files**:
  - Baseline: `RAG/results/comprehensive_200.jsonl`
  - TMD re-rank: `RAG/results/tmd_200_oct4.jsonl`


# Combined:
## Unified RAG Backend Benchmarks

| Backend | Query Set | P@1 | P@5 | P@10 | MRR | nDCG | Latency | Throughput |
|---------|-----------|-----|-----|------|-----|------|---------|------------|
| **vecRAG** ✅ | Core (200q) | 55.0% | 75.5% | — | 0.649 | 0.684 | 0.05ms ⚡ | 20,000 q/s ⚡ |
| **vec (baseline)** | TMD (200q) | 91.5% | 95.6% | 97.0% | 0.936 | 0.965 | — | — |
| **vec_tmd_rerank** (α=0.3) | TMD (200q) | 94.5% | 97.5% | 98.5% | 0.956 | 0.978 | ~1.5s/q | — |
| **vec (baseline)** | Graph (50q) | 60.0% | 84.0% | — | 0.712 | 0.745 | 0.06ms ⚡ | 16,667 q/s |
| **vec_graph_rerank** | Graph (50q) | 60.0% | 84.0% | — | 0.712 | 0.745 | 9.56ms | 105 q/s |
| **LightVec** | Core (200q) | 55.0% | 75.5% | — | 0.649 | 0.684 | 0.09ms | 11,111 q/s |
| **BM25** | Core (200q) | 48.5% | 72.5% | — | 0.605 | 0.647 | 0.94ms | 1,064 q/s |
| **Lexical** | Core (200q) | 49.0% | 71.0% | — | 0.586 | 0.623 | 0.38ms | 2,632 q/s |
| **graphrag_hybrid** 🔴 | Graph (50q) | 8.0% | 26.0% | — | 0.153 | 0.215 | 434ms | 2.3 q/s |

### TMD Alpha Tuning (In Progress)

| Alpha | TMD Weight | Vector Weight | Status | Expected P@5 |
|-------|------------|---------------|--------|--------------|
| 0.2 | 20% | 80% | Queued | TBD |
| 0.3 | 30% | 70% | ✅ Complete | 97.5% |
| 0.4 | 40% | 60% | Queued | TBD |
| 0.5 | 50% | 50% | Queued | TBD |
| 0.6 | 60% | 40% | Queued | TBD |

**Tuning Infrastructure:** `tune_alpha.sh` → `compare_alpha_results.py` (~25min total)  
**Goal:** Optimize P@5 from 97.5% → 98%+

**Notes:**
- vecRAG/LightVec identical (same FAISS+TMD index)
- TMD re-ranking (α=0.3): +3.0pp P@1, +2.0pp P@5
- GraphRAG broken: -86.7% P@1 (10x edge expansion)
- Core/TMD = different query sets