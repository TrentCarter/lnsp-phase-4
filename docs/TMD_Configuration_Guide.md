# TMD Re-ranker Configuration Guide

## Overview

The Token-Matching Density (TMD) re-ranker improves retrieval by boosting results that contain query tokens. This guide explains the configuration parameters and their optimal values.

## Environment Variables

### Core TMD Parameters

```bash
# Alpha: Weight for TMD boost (0.0 = pure vector, 1.0 = pure TMD)
export TMD_ALPHA=0.3              # Default: 0.3 (30% TMD influence)

# Normalization: How to combine vector and TMD scores
export TMD_NORM=softmax           # Options: softmax, minmax, zscore
                                  # Default: softmax (most stable)

# Temperature: Controls softmax sharpness
export TMD_TEMP=1.0               # Default: 1.0 (balanced)
                                  # Higher = smoother, Lower = sharper
```

### Search Pool Configuration

```bash
# Search pool multiplier (how many candidates to consider)
export TMD_SEARCH_MULT=10         # Default: 10 (10x top-k)

# Maximum search pool size
export TMD_SEARCH_MAX=200         # Default: 200 candidates max
                                  # Prevents excessive computation
```

### Diagnostics & LLM

```bash
# Enable per-query diagnostics (Spearman correlation, rank changes)
export TMD_DIAG=1                 # 1=enabled, 0=disabled

# Use LLM for TMD extraction (vs simple tokenization)
export TMD_USE_LLM=1              # 1=enabled, 0=disabled

# LLM endpoint for TMD extraction
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

## Usage Examples

### Quick Test (Default Settings)
```bash
./scripts/run_lightrag_benchmark.sh 50
```

### Custom Alpha (More TMD Influence)
```bash
TMD_ALPHA=0.5 ./scripts/run_lightrag_benchmark.sh 200
```

### Disable Diagnostics (Faster)
```bash
TMD_DIAG=0 ./scripts/run_lightrag_benchmark.sh 200
```

### Pure TMD (No Vector)
```bash
TMD_ALPHA=1.0 TMD_SEARCH_MULT=20 ./scripts/run_lightrag_benchmark.sh 100
```

### Conservative Settings (Safer)
```bash
TMD_ALPHA=0.2 \
TMD_SEARCH_MULT=5 \
TMD_SEARCH_MAX=100 \
./scripts/run_lightrag_benchmark.sh 200
```

## Parameter Tuning Guide

### TMD_ALPHA (Primary Knob)

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Pure vector search | Semantic-only retrieval |
| 0.1-0.3 | Light TMD boost | General purpose (recommended) |
| 0.4-0.6 | Balanced hybrid | Token matching important |
| 0.7-1.0 | TMD-dominant | Lexical precision critical |

**Sweet spot**: `0.2-0.4` for most use cases

### TMD_NORM (Stability)

| Method | Pros | Cons |
|--------|------|------|
| `softmax` | Stable, probabilistic | Requires TMD_TEMP tuning |
| `minmax` | Simple, fast | Sensitive to outliers |
| `zscore` | Handles outliers | Assumes normal distribution |

**Recommended**: `softmax` with `TMD_TEMP=1.0`

### TMD_SEARCH_MULT (Recall)

| Value | Search Pool (for topk=10) | Trade-off |
|-------|---------------------------|-----------|
| 5 | 50 candidates | Faster, less exploration |
| 10 | 100 candidates | Balanced (default) |
| 20 | 200 candidates | Slower, more thorough |

**Recommended**: `10` (balances speed and quality)

## Diagnostics Output

When `TMD_DIAG=1`, the system generates:

```jsonl
{
  "query": "neural networks",
  "backend": "vec_tmd_rerank",
  "tmd_diagnostics": {
    "spearman": 0.85,              # Correlation: vec vs TMD scores
    "changed_at_1": 2,              # Rank changes in top-1
    "changed_at_5": 8,              # Rank changes in top-5
    "changed_at_10": 15,            # Rank changes in top-10
    "collapse_pct": 0.12            # % of candidates with identical TMD
  }
}
```

### Interpreting Diagnostics

- **spearman > 0.8**: TMD agrees with vector (safe to increase alpha)
- **spearman < 0.5**: TMD diverges from vector (use lower alpha)
- **changed_at_k**: Higher = more re-ranking activity
- **collapse_pct > 0.3**: Many ties (consider LLM extraction)

## Performance Impact

### Latency Benchmarks (200 queries)

| Backend | Mean Latency | Notes |
|---------|--------------|-------|
| vec | 0.05ms | Baseline FAISS |
| vec_tmd_rerank (alpha=0.3) | 1.5s | LLM extraction per query |
| vec_tmd_rerank (TMD_USE_LLM=0) | 50ms | Simple tokenization |

**Key Insight**: LLM extraction adds ~1.5s per query. For production, consider:
1. Pre-compute TMD vectors (store in DB)
2. Use simple tokenization (`TMD_USE_LLM=0`)
3. Batch LLM extraction offline

## Production Recommendations

### Development (Quality Focus)
```bash
export TMD_ALPHA=0.3
export TMD_NORM=softmax
export TMD_TEMP=1.0
export TMD_SEARCH_MULT=10
export TMD_SEARCH_MAX=200
export TMD_DIAG=1
export TMD_USE_LLM=1
```

### Production (Speed Focus)
```bash
export TMD_ALPHA=0.25
export TMD_NORM=softmax
export TMD_TEMP=1.0
export TMD_SEARCH_MULT=5
export TMD_SEARCH_MAX=100
export TMD_DIAG=0
export TMD_USE_LLM=0  # Use pre-computed TMD vectors
```

### High-Precision (Accuracy Focus)
```bash
export TMD_ALPHA=0.5
export TMD_NORM=softmax
export TMD_TEMP=0.5   # Sharper softmax
export TMD_SEARCH_MULT=20
export TMD_SEARCH_MAX=500
export TMD_DIAG=1
export TMD_USE_LLM=1
```

## Troubleshooting

### Issue: Low P@1 improvement
- **Check**: `spearman` in diagnostics
- **Fix**: Increase `TMD_ALPHA` if spearman > 0.7
- **Fix**: Decrease `TMD_ALPHA` if spearman < 0.5

### Issue: High latency
- **Fix**: Set `TMD_USE_LLM=0` (use simple tokenization)
- **Fix**: Reduce `TMD_SEARCH_MULT` to 5
- **Fix**: Pre-compute TMD vectors offline

### Issue: Many ties (high collapse_pct)
- **Fix**: Enable LLM extraction (`TMD_USE_LLM=1`)
- **Fix**: Use query expansion for longer queries
- **Fix**: Consider different tokenization strategy

### Issue: Unstable results
- **Fix**: Use `TMD_NORM=softmax` (most stable)
- **Fix**: Increase `TMD_TEMP` to 2.0 (smoother)
- **Fix**: Reduce `TMD_ALPHA` to 0.2 (less TMD influence)

## Advanced: Alpha Parameter Sweep

To find optimal alpha for your dataset:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6; do
  echo "Testing TMD_ALPHA=$alpha"
  TMD_ALPHA=$alpha ./scripts/run_lightrag_benchmark.sh 200
done

# Analyze results
grep '"summary": true' RAG/results/*.jsonl | \
  jq -r '[.backend, .metrics.p_at_1, .metrics.ndcg] | @tsv'
```

## See Also

- [TMD Alpha Tuning Postmortem](../TMD-Alpha-Rerank-Postmortem.md) - Lessons learned
- [LightRAG Benchmark Script](../scripts/run_lightrag_benchmark.sh) - Implementation
- [TMD Reranker Code](../RAG/vecrag_tmd_rerank.py) - Core algorithm

---

**Last Updated**: October 5, 2025
**Status**: Production-ready defaults configured
**Contact**: See project README for questions
