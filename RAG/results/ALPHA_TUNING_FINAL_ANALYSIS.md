# TMD Alpha Tuning - Final Analysis (October 4, 2025)

## Executive Summary

✅ **Bug Fixed**: Metrics calculation had 0-based vs 1-based indexing error
✅ **TMD Re-ranking Works**: +1.5pp improvement in P@5 (75.5% → 77.0%)
⚠️  **Alpha Tuning Inconclusive**: All alpha values produce identical results

## Corrected Results

### Alpha Parameter Testing (200 queries)

| Alpha | TMD Weight | Vector Weight | P@1 | P@5 | P@10 | MRR |
|-------|------------|---------------|-----|-----|------|-----|
| 0.2 | 20% | 80% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.3 | 30% | 70% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.4 | 40% | 60% | 55.5% | 77.0% | 79.0% | 0.6555 |
| 0.5 | 50% | 50% | 55.0% | 77.0% | 79.0% | 0.6530 |
| 0.6 | 60% | 40% | 55.0% | 77.0% | 78.5% | 0.6524 |

### Baseline Comparison

| Method | P@1 | P@5 | P@10 | MRR | Notes |
|--------|-----|-----|------|-----|-------|
| **Baseline vecRAG** | 55.0% | 75.5% | 79.0% | 0.6490 | Pure vector similarity |
| **TMD re-rank (α=0.3)** | 55.5% | 77.0% | 79.0% | 0.6555 | +1.5pp P@5 improvement |
| **Improvement** | +0.5pp | +1.5pp | 0.0pp | +0.0065 | Modest but consistent |

## Key Findings

### 1. TMD Re-ranking Provides Small Improvement
- **P@5**: 75.5% → 77.0% (+1.5 percentage points)
- **P@1**: 55.0% → 55.5% (+0.5 percentage points)
- **MRR**: 0.6490 → 0.6555 (+1.0% relative improvement)

### 2. Alpha Parameter Makes No Difference
All alpha values from 0.2 to 0.6 produce **nearly identical results**. This indicates:
- TMD signal is much weaker than vector similarity
- Score normalization may be washing out TMD differences
- Query TMD extraction may not be varying enough between queries

### 3. Earlier Test Results Were Incorrect
The initial claim of P@5=97.5% was based on **metrics calculation bug**:
- Bug: Checked for `gold_rank == 0` (0-based)
- Reality: Benchmark uses `gold_rank == 1` (1-based)
- Fix: Updated metrics to use correct indexing

## Root Cause Analysis

### Why Doesn't Alpha Matter?

**Hypothesis 1: Weak TMD Signal**
```python
# From vecrag_tmd_rerank.py line 205:
combined_scores = alpha * vec_scores_norm + (1.0 - alpha) * tmd_similarities
```

If `tmd_similarities` are all similar (~0.8-0.9 for most docs), changing alpha won't affect ranking.

**Hypothesis 2: Score Normalization**
Vector scores are normalized to [0,1] range:
```python
vec_scores_norm = (vec_scores - vec_scores.min()) / (vec_scores.max() - vec_scores.min() + 1e-8)
```

This might be compressing the score range too much.

**Hypothesis 3: Uniform Query TMD**
If all queries extract similar TMD codes (e.g., all map to domain=9), TMD matching won't discriminate.

## Verification Tests

To determine which hypothesis is correct:

### Test 1: Check TMD Similarity Distribution
```python
# For each query, what's the TMD similarity distribution?
query_tmd = generate_tmd_for_query("material entity")
corpus_tmds = corpus_vectors[:, :16]
tmd_sims = compute_tmd_similarity(query_tmd, corpus_tmds)
print(f"TMD sim range: {tmd_sims.min():.3f} - {tmd_sims.max():.3f}")
print(f"TMD sim std: {tmd_sims.std():.3f}")
```

If std is very low (~0.01), TMD isn't discriminating.

### Test 2: Check Query TMD Diversity
```python
# Do different queries get different TMD codes?
queries = ["material entity", "continuant", "MAQC data"]
for q in queries:
    tmd = extract_tmd_with_llm(q)
    print(f"{q}: domain={tmd['domain_code']}, task={tmd['task_code']}")
```

If all queries map to same TMD code, re-ranking won't help.

### Test 3: Bypass Normalization
```python
# Try without normalizing vector scores
combined_scores = alpha * vec_scores + (1.0 - alpha) * tmd_similarities
```

If this changes results, normalization is the issue.

## Recommendations

### Immediate Actions

1. ✅ **Use alpha=0.2-0.3** (performs marginally better, minimal TMD weight)
2. ✅ **Accept +1.5pp P@5 improvement** (modest but real)
3. ⏳ **Skip corpus re-ingestion** (alpha doesn't matter, so LLM-based corpus TMD won't help)

### Future Investigation

1. **Debug TMD similarity distribution** (Test 1 above)
2. **Check query TMD diversity** (Test 2 above)
3. **Try unormalized scoring** (Test 3 above)
4. **Consider alternative TMD encoding** (current 16D may lose information)

### Alternative Approaches

Instead of TMD re-ranking, consider:
1. **GraphRAG improvements** (fix 10x edge expansion bug first)
2. **Hybrid retrieval** (BM25 + vector fusion)
3. **Query expansion** (use LLM to generate related terms)

## Conclusion

### What Worked
✅ TMD re-ranking provides **+1.5pp P@5 improvement** (75.5% → 77.0%)
✅ Metrics calculation bug fixed (was showing P@1=0% incorrectly)
✅ Infrastructure built for future alpha tuning experiments

### What Didn't Work
❌ Alpha parameter tuning inconclusive (all values perform identically)
❌ Cannot optimize beyond α=0.2-0.3 (TMD signal too weak)
❌ Initial P@5=97.5% claim was based on calculation bug

### Time Spent vs Value
- **Time invested**: ~30 minutes (alpha tuning) + debugging
- **Value gained**: +1.5pp P@5 improvement
- **Lessons learned**: Always verify metrics calculation, TMD signal weaker than expected

### Recommendation

**Use TMD re-ranking with alpha=0.2** for modest improvement, but prioritize:
1. Fixing GraphRAG (currently broken at P@1=8%)
2. Investigating why TMD signal is so weak
3. Exploring alternative re-ranking methods

---

**Date**: 2025-10-04
**Status**: Analysis complete
**Action**: Document findings, move to GraphRAG fixes
