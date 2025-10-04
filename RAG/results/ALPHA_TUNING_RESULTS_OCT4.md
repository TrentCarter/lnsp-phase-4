# TMD Alpha Tuning Results (October 4, 2025)

## Summary

Alpha parameter tuning completed testing 5 values (0.2, 0.3, 0.4, 0.5, 0.6) over 200 queries.

## Results

| Alpha | TMD Weight | Vector Weight | P@1 | P@5 | P@10 | MRR | nDCG |
|-------|------------|---------------|-----|-----|------|-----|------|
| 0.2 | 20% | 80% | 0.0% | 76.0% | 79.0% | 0.3463 | 0.4594 |
| 0.3 | 30% | 70% | 0.0% | 76.0% | 79.0% | 0.3463 | 0.4594 |
| 0.4 | 40% | 60% | 0.0% | 76.0% | 79.0% | 0.3463 | 0.4594 |
| 0.5 | 50% | 50% | 0.0% | 76.0% | 78.5% | 0.3455 | 0.4588 |
| 0.6 | 60% | 40% | 0.0% | 76.0% | 78.5% | 0.3450 | 0.4573 |

## Key Findings

1. **All alpha values perform similarly** - P@5 remains around 76% across all tested values
2. **Slight degradation at higher TMD weights** - Alpha 0.5-0.6 show minor drops in MRR/nDCG
3. **P@1 is 0%** across all configurations - suggests ranking issues

## Analysis

### Expected vs Actual Results

**Expected** (from earlier TMD re-rank test):
- Baseline vecRAG: P@5 = 95.6%
- TMD re-rank: P@5 = 97.5%

**Actual** (from alpha tuning):
- All alphas: P@5 = 76.0%

This 20-point discrepancy suggests either:
1. Different query sets between tests
2. Different evaluation methodology
3. Issue with TMD re-ranking implementation during alpha tuning

### P@1 = 0% Issue

The fact that P@1 is consistently 0% (gold document never ranks first) indicates a systematic ranking problem. Possible causes:
1. TMD re-ranking might be inverting scores
2. Normalization issue between vector and TMD scores
3. Query TMD extraction producing incorrect codes

## Recommendations

### Immediate Actions

1. **Verify query set consistency**
   ```bash
   # Compare query sets
   diff <(jq -r '.query' RAG/results/tmd_200_oct4.jsonl | sort) \
        <(jq -r '.query' RAG/results/tmd_alpha_0.3_oct4.jsonl | sort)
   ```

2. **Debug ranking scores**
   ```bash
   # Check a single query's ranking
   head -1 RAG/results/tmd_alpha_0.3_oct4.jsonl | jq '.hits[:5]'
   ```

3. **Test pure vector baseline**
   ```bash
   # Run baseline without TMD to establish ground truth
   PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
     --dataset self --n 10 --topk 10 --backends vec \
     --out RAG/results/baseline_debug.jsonl
   ```

### Investigation Priority

1. ✅ **HIGH**: Investigate P@1 = 0% issue
2. ✅ **HIGH**: Verify TMD score normalization
3. ✅ **MEDIUM**: Compare query sets between test runs
4. ⏳ **LOW**: Fine-tune alpha after fixing ranking issues

## Current Status

- ⚠️  **Alpha tuning inconclusive** - Results don't match earlier TMD re-rank test
- ⏳ **Need debugging** - P@1=0% suggests systematic issue
- ✅ **Infrastructure working** - Alpha tuning pipeline executed successfully

## Next Steps

1. Run debug baseline test (vec only, 10 queries)
2. Compare scores between vec and vec_tmd_rerank for same queries
3. Fix ranking issue
4. Re-run alpha tuning once ranking is verified correct

## Files

- **Results**: `RAG/results/tmd_alpha_*.jsonl` (5 files, ~166KB each)
- **Analysis**: `tools/compute_alpha_metrics.py`
- **Tuning script**: `tune_alpha.sh`

---

**Status**: ⚠️  Needs debugging
**Date**: 2025-10-04
**Time to complete**: 25 minutes (5 min × 5 alphas)
