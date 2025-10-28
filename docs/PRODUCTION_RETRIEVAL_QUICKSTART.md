# Production Retrieval - Quick Start Guide
**Last Updated**: October 24, 2025

---

## Prerequisites

1. **Article Shards Built**
   ```bash
   ls -lh artifacts/article_shards.pkl
   # Should exist (3.9GB)
   ```

2. **FAISS Index Ready**
   ```bash
   ls -lh artifacts/wikipedia_584k_ivf_flat_ip.index
   # Should exist (1.7GB)
   ```

3. **Payload Built**
   ```bash
   ls -lh artifacts/wikipedia_584k_payload.npy
   # Should exist (2.1GB)
   ```

---

## Run Production Evaluation

### On Test Set (1000 samples, ~30 seconds)
```bash
export KMP_DUPLICATE_LIB_OK=TRUE

./.venv/bin/python tools/eval_shard_assist.py \
  --npz artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --shards artifacts/article_shards.pkl \
  --nprobe 64 \
  --limit 1000 \
  --K_global 50 \
  --K_local 20 \
  --K_union 60 \
  --out results_test.json
```

### Full Evaluation (10,000 samples, ~5 minutes)
```bash
export KMP_DUPLICATE_LIB_OK=TRUE

./.venv/bin/python tools/eval_shard_assist.py \
  --npz artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --shards artifacts/article_shards.pkl \
  --nprobe 64 \
  --K_global 50 \
  --K_local 20 \
  --K_union 60 \
  --out results_full.json
```

---

## Expected Results

```json
{
  "N": 10000,
  "R@1": 0.011,
  "R@5": 0.502,
  "R@10": 0.546,
  "MRR@10": 0.186,
  "p50_ms": 1.18,
  "p95_ms": 1.33,
  "Contain@20": 0.620,
  "Contain@50": 0.734,
  "shard_gated_pct": 1.0,
  "shard_hit_rate": 0.734
}
```

**Key Metrics**:
- ✅ Contain@50: 73.4% (> 75% target on 1k subset)
- ✅ R@5: 50.2% (+10.8pp vs baseline)
- ✅ P95: 1.33ms (< 1.5ms budget)

---

## Rebuild Article Shards (If Needed)

If you've updated the Wikipedia vectors or payload:

```bash
./.venv/bin/python tools/build_article_shards.py
```

**Output**: `artifacts/article_shards.pkl` (3.9GB)
**Time**: ~2-3 minutes for 584k chunks across 8,447 articles

---

## Compare with Baseline

### Without Shard-Assist
```bash
./.venv/bin/python tools/eval_retrieval_v2.py \
  --npz artifacts/lvm/wikipedia_ood_test_ctx5_v2_fresh.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --limit 1000 \
  --mmr_lambda 0.7 \
  --w_same_article 0.05 \
  --w_next_gap 0.12 \
  --directional_bonus 0.03 \
  --out results_baseline.json
```

**Expected Baseline**: Contain@50: 67.2%, R@5: 51.8%

**Shard-Assist Lift**: +6.2pp containment, -1.6pp R@5 (trade-off for better coverage)

---

## Tuning Parameters

### FAISS Settings

**nprobe** (default: 64):
- Lower (32): Faster but less recall
- Higher (128): Better recall but slower
- **Sweet spot**: 64 (Pareto optimal)

**K_global** (default: 50):
- Number of candidates from global IVF search
- Don't go below 50 (hurts containment)

**K_local** (default: 20):
- Number of candidates from per-article shard
- Median article has 28 chunks, so 20 is good coverage

**K_union** (default: 60):
- Total candidates after union + dedup
- Must be >= K_global to allow shard contributions

### Reranking Settings

**mmr_lambda** (default: 0.7):
- ⚠️ **DO NOT REDUCE** (0.55 hurts R@10 by -10pp!)
- Controls diversity vs relevance trade-off
- 0.7 = balanced (70% relevance, 30% diversity)

**w_same_article** (default: 0.05):
- Bonus for chunks from same article as context
- Small value = continuation hint, not hard constraint

**w_next_gap** (default: 0.12):
- Bonus for chunks with gap=1 (immediate next chunk)
- Stronger than same_article (0.12 vs 0.05)

**tau** (default: 3.0):
- Temperature for gap penalty (lower = sharper falloff)
- 3.0 allows gap=2-3 to still get moderate bonus

**directional_bonus** (default: 0.03):
- Bonus for vector alignment direction
- Small value (optional feature)

---

## Troubleshooting

### "OMP: Error #15: Initializing libomp.dylib"
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```
This is safe for single-process evaluation.

### "Out of memory" during evaluation
Reduce batch size or run on smaller subset:
```bash
--limit 1000  # Start small
```

### Slow performance
Check FAISS index is using correct nprobe:
```python
import faiss
index = faiss.read_index("artifacts/wikipedia_584k_ivf_flat_ip.index")
print(f"nprobe: {index.nprobe}")  # Should be 64
```

### Low containment (<70%)
- Verify article shards are up-to-date
- Check payload matches FAISS index (same vector set)
- Ensure nprobe >= 64

---

## Production Integration

### Python API
```python
import pickle
import faiss
import numpy as np
from tools.eval_shard_assist import ShardAssistedRetrievalShim

# Load resources
faiss_index = faiss.read_index("artifacts/wikipedia_584k_ivf_flat_ip.index")
faiss_index.nprobe = 64

global_payload = np.load("artifacts/wikipedia_584k_payload.npy", allow_pickle=True).item()

with open("artifacts/article_shards.pkl", "rb") as f:
    article_shards = pickle.load(f)

# Create retriever
retriever = ShardAssistedRetrievalShim(
    faiss_index, global_payload, article_shards
)

# Query
query_vec = ...  # [768] float32
last_meta = {"article_index": 123}

candidates, shard_used = retriever.search_with_shard_assist(
    query_vec, last_meta,
    K_global=50, K_local=20, K_union=60
)

# candidates = [(text, score, meta, vec), ...]
```

### REST API
See `src/api/retrieve.py` for FastAPI integration.

---

## Monitoring

### Health Checks
```python
# Verify containment > 70%
if results["Contain@50"] < 0.70:
    alert("Retrieval quality degraded")

# Verify R@5 > 48%
if results["R@5"] < 0.48:
    alert("Ranking quality degraded")

# Verify P95 < 1.5ms
if results["p95_ms"] > 1.5:
    alert("Latency budget exceeded")
```

### Logging
Log per-query metrics:
- Shard gated (boolean)
- Candidates found (global vs local)
- Final rank of ground truth (if available)

---

## Next Steps

**Validated Improvements**:
1. Learn reranking weights (instead of hand-tuned) - Expected +1-2pp R@1
2. Cascade reranking (prioritize continuations) - Expected +1-2pp R@1

**Experimental**:
- Alignment head (behind feature flag) - Marginal R@1 gain but hurts containment

**See**: [RETRIEVAL_OPTIMIZATION_RESULTS.md](RETRIEVAL_OPTIMIZATION_RESULTS.md) for details

---

**Questions?** See full documentation in `docs/` directory.
