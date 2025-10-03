# RAG Test Setup PRD

**Date:** 10/1/2025
**Status:** ✅ COMPLETE - Production Ready
**Version:** 2.0 (with BM25 + Enhanced Output)

---

## Overview

RAG-only benchmarking harness to evaluate **vecRAG** in isolation and compare against state-of-art RAG baselines. Runs retrieval-only (no LVM, no generation) with proper metrics and latency tracking.

---

## Components

### Main Harness
**RAG/bench.py** — Retrieval-only benchmark runner with 5 backends:

| Backend | Type | Description | Requires |
|---------|------|-------------|----------|
| `vec` | Dense | FAISS vecRAG (768D or 784D fused) | FAISS index + NPZ |
| `bm25` | Lexical | BM25 (STRONG baseline) | rank-bm25 |
| `lex` | Lexical | Token overlap (weak, reference only) | - |
| `lightvec` | Dense | LightRAG vector-only (same FAISS) | lightrag-hku |
| `lightrag_full` | Hybrid | LightRAG + graph (EXPERIMENTAL) | artifacts/kg/ |

### Documentation
- **RAG/README.md** — Usage, options, comprehensive interpretation guide
- **RAG/rag_test_prd.md** — This file (PRD + status)

### Testing
- **RAG/test_simple.py** — Component verification (imports, BM25 smoke test)

---

## Architecture

### Reused Components
- `src/db_faiss.py` (FaissDB) — FAISS search interface
- `src/vectorizer.py` (EmbeddingBackend) — GTR-T5 768D embeddings
- `src/adapters/lightrag/vectorstore_faiss.py` — LightRAG FAISS bridge
- `rank-bm25` (BM25Okapi) — Strong lexical baseline

### Data Flow
1. Load corpus from NPZ (auto-detect 768D or 784D)
2. Build queries:
   - **768D**: Pure concept embeddings
   - **784D**: `[tmd_dense(16) + concept_vec(768)]` → L2-normalized for IP
3. Run backends in parallel (shared FAISS index for fairness)
4. Compute metrics: P@1, P@5, MRR@10, nDCG@10
5. Track latency: mean + P95 per backend
6. Output: per-query JSONL + summary Markdown

---

## Datasets

### 1. Self-Retrieval (Sanity Check)
- **Query:** `concept_text[i]`
- **Gold:** Position `i`
- **Purpose:** Verify index can find exact matches (P@1 should be >0.95)

### 2. CPESH Queries (Real-World)
- **Source:** `artifacts/cpesh_cache.jsonl`
- **Query:** CPESH `probe` (fallback: `expected` or `concept`)
- **Gold:** Mapped `doc_id` position in NPZ
- **Purpose:** Realistic retrieval performance

---

## Metrics & Outputs

### Metrics
- **P@1, P@5** — Precision at top-1 and top-5
- **MRR@10** — Mean Reciprocal Rank (rewards top hits)
- **nDCG@10** — Normalized Discounted Cumulative Gain (graded relevance)
- **Latency** — Mean + P95 in milliseconds

### Enhanced Output Format

**JSONL** (`RAG/results/bench_<timestamp>.jsonl`):
```json
{
  "backend": "vec",
  "query": "What is the capital of France?",
  "gold_pos": 42,
  "gold_doc_id": "doc_france_001",
  "hits": [
    {"doc_id": "doc_france_001", "score": 0.95, "rank": 1},
    {"doc_id": "doc_europe_012", "score": 0.87, "rank": 2},
    ...
  ],
  "gold_rank": 1
}
```

**Markdown Summary** (`RAG/results/summary_<timestamp>.md`):
```
| Backend | P@1 | P@5 | MRR@10 | nDCG@10 | Mean ms | P95 ms |
|---------|-----|-----|--------|---------|---------|--------|
| vec     | 0.95| 0.72| 0.85   | 0.78    | 12.3    | 18.5   |
| bm25    | 0.48| 0.61| 0.67   | 0.59    | 3.2     | 4.8    |
```

---

## Usage

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt  # includes rank-bm25

# Ensure artifacts exist
ls artifacts/faiss_meta.json  # Points to FAISS index
ls artifacts/fw*_vectors.npz  # NPZ with vectors + metadata
```

### Examples

```bash
# RECOMMENDED: vecRAG vs BM25 (strongest comparison)
python RAG/bench.py --dataset self --n 1000 --topk 10 --backends vec,bm25

# Full comparison (all backends)
python RAG/bench.py --dataset self --n 1000 --topk 10 --backends vec,bm25,lex,lightvec

# CPESH queries (real-world performance)
python RAG/bench.py --dataset cpesh --n 500 --topk 10 --backends vec,bm25

# Self-retrieval sanity check (should get P@1 > 0.95)
python RAG/bench.py --dataset self --n 100 --topk 5 --backends vec
```

### Environment Variables
```bash
# Point to specific NPZ
export FAISS_NPZ_PATH=artifacts/fw10k_vectors.npz

# Tune IVF search
export FAISS_NPROBE=16  # Balance speed/accuracy
```

---

## Performance Interpretation Guide

### Expected Baselines (CPESH queries)

| Backend | P@5 Range | Notes |
|---------|-----------|-------|
| Token overlap (lex) | 0.30-0.45 | Weak baseline |
| **BM25** | **0.50-0.65** | **Strong lexical** |
| vecRAG (dense) | 0.60-0.75 | **Target: beat BM25 by 10-15%** |
| LightRAG vector | 0.60-0.75 | Same FAISS backend |
| LightRAG full | 0.70-0.85 | Graph boost, 2-3x slower |

### Latency Targets

| Backend | Mean Latency | Notes |
|---------|--------------|-------|
| 768D FAISS | <10ms | nprobe=16 |
| 784D fused | <15ms | nprobe=16 |
| BM25 | <5ms | No embedding overhead |
| LightRAG full | 30-50ms | Graph traversal |

### Red Flags 🚩

⚠️ **P@1 < 0.90 on self-retrieval**
→ Index corruption or dimension mismatch

⚠️ **vecRAG worse than BM25**
→ Embedding quality or FAISS tuning issue

⚠️ **Latency > 30ms for dense retrieval**
→ IVF parameters need tuning (reduce nprobe or nlist)

⚠️ **P@5 < 0.30 on CPESH**
→ Poor query-corpus alignment, check embedding coverage

---

## Implementation Details

### What Changed (v2.0)

1. **Added BM25 Baseline**
   - Strong lexical comparator via `rank-bm25`
   - Proper tokenization and BM25Okapi scoring
   - Fast (<5ms) reference baseline

2. **Enhanced JSONL Output**
   - Per-hit `doc_id`, `score`, `rank`
   - Gold document tracking
   - Better debugging and error analysis

3. **LightRAG Graph Mode**
   - `lightrag_full` backend for hybrid retrieval
   - Requires `artifacts/kg/` with knowledge graph
   - Experimental (result mapping not yet complete)

4. **Comprehensive Documentation**
   - Expected performance baselines
   - Latency targets by backend
   - Red flag thresholds
   - Interpretation guide

5. **Fixed Import Issues**
   - Proper absolute imports from `src/`
   - NPZ loading handles missing fields safely
   - Component tests verify setup

### Files Modified
- `RAG/bench.py` — Added BM25, enhanced output, fixed imports
- `RAG/README.md` — Complete interpretation guide
- `RAG/rag_test_prd.md` — This PRD (updated)
- `requirements.txt` — Added rank-bm25
- `RAG/test_simple.py` — New component test

---

## Testing

### Smoke Test
```bash
# Verify all imports work
python RAG/test_simple.py
# Expected: ✓ All tests passed!
```

### Small Benchmark
```bash
# Run on 50 queries (fast sanity check)
export FAISS_NPZ_PATH=artifacts/fw1k_vectors.npz
python RAG/bench.py --dataset self --n 50 --topk 5 --backends vec,bm25
```

### Production Run
```bash
# Full evaluation (1000 queries, all backends)
python RAG/bench.py --dataset cpesh --n 1000 --topk 10 \
  --backends vec,bm25,lex,lightvec \
  --out RAG/results/production_$(date +%Y%m%d).jsonl
```

---

## Notes

- Harness **auto-detects NPZ vector shape** (768 vs 784) and builds queries to match
- LightRAG vector-only uses **same FAISS index** for fair comparison
- BM25 is the **primary lexical baseline** (stronger than token overlap)
- Enhanced JSONL includes per-hit details for deep analysis
- `lightrag_full` requires artifacts/kg/ with knowledge graph data

---

## Status ✅ COMPLETE (10/1/2025)

### Implemented
- ✅ BM25 strong lexical baseline
- ✅ Enhanced JSONL output (scores + ranks)
- ✅ LightRAG graph mode support (experimental)
- ✅ Comprehensive interpretation guide
- ✅ Component tests verified
- ✅ Dependencies updated (rank-bm25)
- ✅ Import issues fixed
- ✅ NPZ loading robustness

### Ready For
- ✅ Production RAG benchmarks
- ✅ vecRAG vs BM25 comparison
- ✅ Latency/accuracy tradeoff analysis
- ✅ Multi-backend evaluation

### Future Enhancements (Optional)
- [ ] Complete `lightrag_full` result mapping
- [ ] Add ELSER/ColBERT baselines
- [ ] Cross-dataset validation
- [ ] Confidence calibration metrics

---

**Ready to run production-grade RAG benchmarks!** 🚀
