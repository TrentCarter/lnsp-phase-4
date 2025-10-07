# GraphRAG Quick Reference Card

**Last Updated:** 2025-10-05 | **Status:** ✅ Phase 1+2 Fixes Validated

---

## 🚀 One-Line Commands

```bash
# Quick baseline test
./scripts/benchmark_graphrag.sh baseline

# Full comprehensive benchmark
./scripts/benchmark_graphrag.sh with-tmd

# GraphRAG validation only
./scripts/benchmark_graphrag.sh graphrag-only
```

---

## 📊 Expected Performance

| Method | P@5 | nDCG@10 | Latency | Use Case |
|--------|-----|---------|---------|----------|
| **vecRAG + TMD** | **0.910** | **0.760** | 1928ms | 💎 **Max precision** |
| **vecRAG** | 0.890 | 0.747 | 0.05ms | ⚡ **Speed-critical** |
| **BM25** | 0.890 | 0.756 | 0.50ms | 🔍 Good baseline |
| **GraphRAG** | 0.890 | 0.747 | 63ms | 🔧 Research only |

---

## 🔧 Quick Tuning

```bash
# Test with half graph weight
GR_GRAPH_WEIGHT=0.5 ./scripts/benchmark_graphrag.sh graphrag-only

# More seeds for expansion
GR_SEED_TOP=20 ./scripts/benchmark_graphrag.sh graphrag-only

# Disable graph (pure vector)
GR_GRAPH_WEIGHT=0.0 ./scripts/benchmark_graphrag.sh graphrag-only

# Grid search
./scripts/benchmark_graphrag.sh tune-weights
```

---

## 🎯 Configuration Variables

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `GR_GRAPH_WEIGHT` | 1.0 | 0.0-5.0 | Graph influence (0=none, 5=max) |
| `GR_SEED_TOP` | 10 | 1-50 | Expansion seeds (more=slower) |
| `GR_SIM_WEIGHT` | 1.0 | 0.0-5.0 | Query similarity weight |
| `GR_RRF_K` | 60 | 10-100 | RRF parameter (standard=60) |

---

## 📁 Results Location

```bash
# Latest summary
cat $(ls -t RAG/results/summary_*.md | head -1)

# All recent results
ls -lt RAG/results/ | head -10
```

---

## ✅ Phase 1+2 Fixes (Oct 5, 2025)

**Safety Guarantee:**
- ✅ GraphRAG P@k ≥ vecRAG P@k (always)
- ✅ Re-ranks only within vector candidates
- ✅ Cannot perform worse than baseline

**Improvements:**
- ✅ Scale calibration (graph uses RRF, not raw confidence)
- ✅ Query similarity term added
- ✅ Configurable via environment variables
- ✅ Increased seeds from 5→10

---

## 🐛 Troubleshooting

**GraphRAG slower than expected?**
- ✅ Expected: 60-100ms (1000x+ slower than vec)
- Try reducing `GR_SEED_TOP` or `GR_GRAPH_WEIGHT`

**GraphRAG P@1 < 0.515?**
- ❌ Should NEVER happen with Phase 1+2 fixes
- File bug report immediately

**Neo4j errors?**
```bash
cypher-shell -u neo4j -p password "RETURN 1"
```

---

## 📚 Full Documentation

See [GraphRAG_Benchmark_Guide.md](GraphRAG_Benchmark_Guide.md) for:
- Complete command reference
- Manual benchmarking
- Historical results
- Detailed tuning guide
