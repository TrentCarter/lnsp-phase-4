# GraphRAG Benchmark Guide

**Last Updated:** 2025-10-05
**Status:** ✅ GraphRAG Phase 1+2 Fixes Validated

---

## 📊 Quick Reference

### Current Performance (Oct 5, 2025)

| Backend | P@1 | P@5 | MRR@10 | nDCG@10 | Mean Latency | Status |
|---------|-----|-----|--------|---------|--------------|--------|
| **vecRAG + TMD** | 0.510 | **0.910** | 0.698 | **0.760** | 1928.31ms | 💎 **Best Quality** |
| **BM25** | 0.545 | 0.890 | 0.699 | 0.756 | 0.50ms | 🔍 Strong baseline |
| **vecRAG** | 0.515 | 0.890 | 0.691 | 0.747 | 0.05ms | ⚡ **Fastest** |
| **GraphRAG (local)** | 0.515 | 0.890 | 0.691 | 0.747 | 63.42ms | ✅ Matches baseline |
| **GraphRAG (hybrid)** | 0.515 | 0.890 | 0.691 | 0.747 | 65.71ms | ✅ Matches baseline |
| **Lexical** | 0.510 | 0.890 | 0.678 | 0.736 | 0.19ms | ✅ Solid baseline |

**Key Findings:**
- ✅ **GraphRAG Fixed:** Now matches vecRAG baseline (was catastrophically broken at P@1=0.030)
- ✅ **Safety Guarantee:** GraphRAG cannot perform worse than vector baseline
- 🔧 **Graph Not Helping:** Graph traversal adds 1,268x latency without quality improvement
- 💎 **TMD Best Quality:** Highest P@5 (0.910) and nDCG@10 (0.760) - use for max precision
- ⚡ **vecRAG Best Speed:** 38,566x faster than TMD, 1,268x faster than GraphRAG

---

## 🚀 Quick Start

### Running Benchmarks

```bash
# Make script executable (first time only)
chmod +x scripts/benchmark_graphrag.sh

# Run baseline benchmark (vec, BM25, lex, GraphRAG)
./scripts/benchmark_graphrag.sh baseline

# Run comprehensive with TMD reranking
./scripts/benchmark_graphrag.sh with-tmd

# Run GraphRAG validation only
./scripts/benchmark_graphrag.sh graphrag-only

# Tune graph weights (grid search)
./scripts/benchmark_graphrag.sh tune-weights

# Run all benchmarks
./scripts/benchmark_graphrag.sh all
```

### Custom Configuration

```bash
# Custom graph weight
GR_GRAPH_WEIGHT=0.5 ./scripts/benchmark_graphrag.sh graphrag-only

# More seeds for expansion
GR_SEED_TOP=20 ./scripts/benchmark_graphrag.sh graphrag-only

# Custom TMD alpha
TMD_ALPHA=0.4 ./scripts/benchmark_graphrag.sh with-tmd

# Small test run
N_QUERIES=50 ./scripts/benchmark_graphrag.sh baseline
```

---

## 🔧 Environment Variables

### GraphRAG Configuration (Phase 1+2 Fixes)

| Variable | Default | Description |
|----------|---------|-------------|
| `GR_RRF_K` | 60 | RRF k parameter (standard from literature) |
| `GR_GRAPH_WEIGHT` | 1.0 | Graph signal weight (0.0 = pure vector, higher = more graph) |
| `GR_SEED_TOP` | 10 | Number of top vector results used as expansion seeds |
| `GR_SIM_WEIGHT` | 1.0 | Query similarity term weight |

### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `FAISS_NPZ_PATH` | `artifacts/fw10k_vectors.npz` | Path to NPZ file with vectors |
| `FAISS_INDEX_PATH` | `artifacts/fw10k_ivf_flat_ip.index` | Path to FAISS index |

### TMD Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TMD_ALPHA` | 0.3 | TMD reranking alpha (optimal from Oct 4 tuning) |

### Benchmark Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `N_QUERIES` | 200 | Number of queries to benchmark |
| `TOPK` | 10 | Top-K results to retrieve |

---

## 📝 Manual Commands

### 1. Baseline Comparison

```bash
export FAISS_NPZ_PATH=artifacts/fw10k_vectors.npz
export FAISS_INDEX_PATH=artifacts/fw10k_ivf_flat_ip.index
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex,graphrag_local,graphrag_hybrid \
  --npz artifacts/fw10k_vectors.npz \
  --index artifacts/fw10k_ivf_flat_ip.index \
  --out RAG/results/baseline_$(date +%Y%m%d_%H%M).jsonl
```

**Results:** `RAG/results/summary_*.md` (latest file)

---

### 2. GraphRAG Validation (Fixed Implementation)

```bash
export GR_RRF_K=60
export GR_GRAPH_WEIGHT=1.0
export GR_SEED_TOP=10
export GR_SIM_WEIGHT=1.0

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,graphrag_local,graphrag_hybrid \
  --npz artifacts/fw10k_vectors.npz \
  --index artifacts/fw10k_ivf_flat_ip.index \
  --out RAG/results/graphrag_validation_$(date +%Y%m%d_%H%M).jsonl
```

---

### 3. Comprehensive with TMD Reranking

```bash
export TMD_ALPHA=0.3

./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 200 \
  --topk 10 \
  --backends vec,bm25,lex,vec_tmd_rerank,graphrag_local,graphrag_hybrid \
  --npz artifacts/fw10k_vectors.npz \
  --index artifacts/fw10k_ivf_flat_ip.index \
  --out RAG/results/comprehensive_tmd_$(date +%Y%m%d_%H%M).jsonl
```

---

## 🔍 Understanding Results

### Reading Summary Files

```bash
# View latest benchmark summary
cat $(ls -t RAG/results/summary_*.md | head -1)

# Compare last 3 runs
for f in $(ls -t RAG/results/summary_*.md | head -3); do
    echo "=== $f ==="
    cat "$f"
    echo ""
done
```

### Key Metrics

- **P@1:** Precision at rank 1 (% of queries where correct doc is #1)
- **P@5:** Precision at rank 5 (% where correct doc is in top-5)
- **MRR@10:** Mean Reciprocal Rank (average of 1/rank for correct doc)
- **nDCG@10:** Normalized Discounted Cumulative Gain (quality-weighted metric)
- **Mean ms:** Average query latency in milliseconds
- **P95 ms:** 95th percentile latency

### Interpreting GraphRAG Results

**Matching baseline (P@k = vec):**
- ✅ Safety guarantee working correctly
- 🔧 Graph not providing useful signal
- 💡 Try tuning `GR_GRAPH_WEIGHT` or check graph quality

**Better than baseline (P@k > vec):**
- 🎉 Graph providing useful semantic relationships
- ✅ Phase 1+2 fixes working as designed

**Worse than baseline (P@k < vec):**
- ❌ Should never happen with Phase 1+2 fixes
- 🐛 Report as bug if observed

---

## 🎯 Tuning Guide

### Graph Weight Tuning

Test different `GR_GRAPH_WEIGHT` values to find optimal balance:

```bash
# Conservative (less graph influence)
GR_GRAPH_WEIGHT=0.25 ./scripts/benchmark_graphrag.sh graphrag-only

# Balanced
GR_GRAPH_WEIGHT=0.5 ./scripts/benchmark_graphrag.sh graphrag-only

# Default
GR_GRAPH_WEIGHT=1.0 ./scripts/benchmark_graphrag.sh graphrag-only

# Aggressive (more graph influence)
GR_GRAPH_WEIGHT=2.0 ./scripts/benchmark_graphrag.sh graphrag-only

# Very aggressive
GR_GRAPH_WEIGHT=5.0 ./scripts/benchmark_graphrag.sh graphrag-only
```

**Automated grid search:**
```bash
./scripts/benchmark_graphrag.sh tune-weights
```

### Seed Count Tuning

More seeds = broader graph exploration, slower queries:

```bash
# Conservative (fewer seeds, faster)
GR_SEED_TOP=5 ./scripts/benchmark_graphrag.sh graphrag-only

# Default
GR_SEED_TOP=10 ./scripts/benchmark_graphrag.sh graphrag-only

# Aggressive (more seeds, slower)
GR_SEED_TOP=20 ./scripts/benchmark_graphrag.sh graphrag-only
```

### Query Similarity Weight

Adjust importance of query-document similarity:

```bash
# Disable query similarity (pure vector+graph RRF)
GR_SIM_WEIGHT=0.0 ./scripts/benchmark_graphrag.sh graphrag-only

# Conservative
GR_SIM_WEIGHT=0.5 ./scripts/benchmark_graphrag.sh graphrag-only

# Default
GR_SIM_WEIGHT=1.0 ./scripts/benchmark_graphrag.sh graphrag-only

# Aggressive (query similarity dominates)
GR_SIM_WEIGHT=2.0 ./scripts/benchmark_graphrag.sh graphrag-only
```

---

## 🐛 Troubleshooting

### GraphRAG Returns P@1 < vec baseline

**This should never happen with Phase 1+2 fixes!**

1. Check GraphRAG code version:
   ```bash
   grep "Phase 1+2 Fixes" RAG/graphrag_backend.py
   ```
   Should show: `Phase 1+2 Fixes (Oct 5, 2025)`

2. Verify safety guarantee is active:
   ```bash
   grep "CRITICAL: re-rank only" RAG/graphrag_backend.py
   ```

3. If issue persists, file a bug report with:
   - Exact command used
   - Summary markdown file
   - Environment variable values

### Neo4j Connection Errors

```bash
# Check Neo4j is running
cypher-shell -u neo4j -p password "RETURN 1"

# Check concept count
cypher-shell -u neo4j -p password \
  "MATCH (c:Concept) RETURN count(c) as total"

# Check edge count
cypher-shell -u neo4j -p password \
  "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as edges"
```

### FAISS Index Not Found

```bash
# Check files exist
ls -lh artifacts/fw10k_vectors.npz artifacts/fw10k_ivf_flat_ip.index

# If missing, rebuild
make build-faiss
```

---

## 📚 Historical Results

### Oct 5, 2025 23:07 - GraphRAG Fixed (Phase 1+2)

**Command:**
```bash
GR_RRF_K=60 GR_GRAPH_WEIGHT=1.0 GR_SEED_TOP=10 GR_SIM_WEIGHT=1.0 \
./scripts/benchmark_graphrag.sh graphrag-only
```

**Results:**
| Backend | P@1 | P@5 | MRR@10 | nDCG@10 | Mean ms |
|---------|-----|-----|--------|---------|---------|
| vec | 0.515 | 0.890 | 0.691 | 0.747 | 0.05 |
| graphrag_local | **0.515** | 0.890 | 0.691 | 0.747 | 63.42 |
| graphrag_hybrid | **0.515** | 0.890 | 0.691 | 0.747 | 65.71 |

**File:** `RAG/results/summary_1759720092.md`

---

### Oct 5, 2025 22:52 - Comprehensive with TMD

**Command:**
```bash
TMD_ALPHA=0.3 ./scripts/benchmark_graphrag.sh with-tmd
```

**Results:**
| Backend | P@1 | P@5 | MRR@10 | nDCG@10 | Mean ms |
|---------|-----|-----|--------|---------|---------|
| vec | 0.515 | 0.890 | 0.691 | 0.747 | 0.05 |
| bm25 | **0.545** | 0.890 | 0.699 | 0.756 | 0.50 |
| lex | 0.510 | 0.890 | 0.678 | 0.736 | 0.19 |
| vec_tmd_rerank | 0.510 | **0.910** | 0.698 | **0.760** | 1928.31 |
| graphrag_local | 0.030 | 0.030 | 0.030 | 0.030 | 28.67 |
| graphrag_hybrid | 0.030 | 0.030 | 0.030 | 0.030 | 31.29 |

**File:** `RAG/results/summary_1759719555.md`
**Note:** This was BEFORE the Phase 1+2 fixes (broken GraphRAG)

---

## 🔗 Related Documentation

- [GraphRAG Implementation](../RAG/graphrag_backend.py) - Phase 1+2 fixes source code
- [Benchmark Script](../RAG/bench.py) - Main benchmarking harness
- [TMD Reranking Guide](TMD_Configuration_Guide.md) - TMD alpha tuning
- [Long-Term Memory](../LNSP_LONG_TERM_MEMORY.md) - Cardinal rules for LNSP

---

## 📞 Support

**Questions or Issues?**
1. Check [Troubleshooting](#-troubleshooting) section above
2. Review historical results for expected performance
3. File issue with full benchmark output and environment config

**Expected Behavior:**
- GraphRAG should **always** match or exceed vecRAG baseline
- If P@1 < 0.515 for GraphRAG, something is wrong
- Latency should be 50-100ms for GraphRAG (1000x+ slower than vec)
