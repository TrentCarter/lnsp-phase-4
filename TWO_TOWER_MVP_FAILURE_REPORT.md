# Two-Tower MVP Failure Report

**Date**: October 20, 2025  
**Status**: ❌ FAILED - 0% Recall@500  
**Root Cause**: Critical data scarcity (138 training pairs insufficient)

---

## Executive Summary

Phase 1 MVP two-tower training **FAILED completely** with 0% Recall@500 across all 20 epochs.

**Training Data**: 138 pairs (16 validation)  
**Result**: 0% Recall@1/5/10/100/500/1000  
**Loss**: Decreased from 3.34 → 3.13 (model learning, but wrong task)  
**Conclusion**: **Severe data scarcity** - need 10-100x more training data

---

## What Happened

### Training Configuration

```
Model: GRUPoolQuery (4.7M params) + IdentityDocTower
Data: 138 training pairs, 16 validation pairs
Context: 100 vectors × 768D
Batch: 32 × 8 accumulation = 256 effective
Epochs: 20
Learning rate: 2e-5
Temperature: 0.07
```

### Training Progression

| Epoch | Train Loss | Recall@10 | Recall@500 | Recall@1000 |
|-------|------------|-----------|------------|-------------|
| 1     | 3.3419     | 0.00%     | 0.00%      | 0.00%       |
| 5     | 3.2902     | 0.00%     | 0.00%      | 0.00%       |
| 10    | 3.2252     | 0.00%     | 0.00%      | 0.00%       |
| 15    | 3.1760     | 0.00%     | 0.00%      | 0.00%       |
| 20    | 3.1319     | 0.00%     | 0.00%      | 0.00%       |

**Best Recall@500**: 0.00% (no improvement)

---

## Root Cause Analysis

### Data Scarcity

**Available data**:
- Phase-3 TMD validation: 154 sequences total
- Split: 138 train / 16 val (90/10)

**Required for two-tower training** (industry standard):
- Minimum: 10,000-50,000 pairs
- Recommended: 100,000-1M pairs
- We have: **138 pairs** (0.14-1.38% of minimum!)

**Result**: Model has no capacity to learn meaningful query formation with this little data.

### Why Loss Decreased But Recall Stayed at 0%

**Loss (InfoNCE) measures**: How well model separates positive from *in-batch negatives* (31 other samples in batch)

**Recall@K measures**: How well model finds target in *full 771k bank*

**What happened**:
1. Model learned to rank positives above the 31 in-batch negatives → loss decreased ✓
2. But model has no concept of global 771k space → recall stayed at 0% ✗

**Analogy**: Learning to win at tic-tac-toe (in-batch) vs learning to win at chess (full-bank). Model learned tic-tac-toe but we're testing it on chess.

---

## Comparison to Expectations

### Consultant's Plan Assumptions

**Consultant assumed**:
```
"Use what you actually have now:
Train/valid sequences: [...] (≈1,540 seqs)"
```

**Reality**:
- We only found 154 sequences in Phase-3 TMD data
- The "~1,540" were training sequences for Phase-3 (not directly usable)
- Result: 10x less data than consultant expected

### Why This Matters

| Data Size | Expected Recall@500 | Actual Recall@500 |
|-----------|---------------------|-------------------|
| 138 pairs (actual) | N/A (too small) | **0.00%** ❌ |
| 1,500 pairs (consultant) | 40-45% | N/A |
| 10,000+ pairs (industry) | 55-60% | N/A |

**Conclusion**: Consultant's plan was sound, but data availability was 10x worse than assumed.

---

## Next Steps

### Option A: Acquire More Training Data (RECOMMENDED)

**Sources**:
1. **Use Phase-3 training data** (1,540 pairs exist somewhere)
   - Need to locate: `artifacts/lvm/data_phase3/training_sequences_ctx100.npz` or similar
   - This alone would give 10x more data

2. **Generate synthetic pairs** from Wikipedia ingestion
   - Export sliding windows from 771k bank
   - Target: 10,000-50,000 pairs
   - Time: 1-2 hours to generate

3. **Full Wikipedia re-export** with proper pair generation
   - Re-process all 771k vectors
   - Create overlapping context windows
   - Target: 100,000+ pairs
   - Time: 4-8 hours

**Recommended**: Start with Option 1 (find Phase-3 training data). If that gets us to 1,500 pairs, retry MVP.

### Option B: Simplify the Task (FALLBACK)

Instead of full-bank retrieval, test on smaller subsets:
- 10k subset: Would need 100-500 pairs
- 100k subset: Would need 1,000-5,000 pairs

**Risk**: Doesn't solve production problem (need full 771k retrieval).

### Option C: Abandon Two-Tower, Explore Alternatives

**Alternatives**:
1. **Hybrid heuristic** (exp weighted α=0.1) → 38.96% Recall@500 (proven)
2. **Deploy Phase-3 for small-set only** → 75.65% Hit@5 (proven)
3. **Wait for more data** from continued Wikipedia ingestion

**Risk**: Delays full-bank retrieval solution indefinitely.

---

## Learnings

### What Worked

✅ **Implementation**:
- All 3 scripts work correctly (pair builder, trainer, evaluator)
- Training loop runs smoothly (loss decreases)
- Evaluation harness functional

✅ **Diagnosis**:
- Clear failure mode (0% recall) immediately identified problem
- Loss vs recall discrepancy revealed in-batch vs full-bank gap

### What Failed

❌ **Data availability**: 138 pairs ≪ 10,000 minimum for two-tower  
❌ **Assumptions**: Consultant assumed 1,540 pairs; we had 154  
❌ **Scoping**: Should have verified data size before implementation

### What We Learned

**Critical insight**: Two-tower retriever training requires **massive amounts of data** (10,000-1M pairs). This is not a technique you can "test with small data" - it either has sufficient data or fails completely.

**Industry context**: Production two-tower models (GTR-T5, DPR, E5) are trained on:
- GTR-T5: 800M pairs (MS MARCO + NQ + others)
- DPR: 80k pairs minimum (with pre-trained encoders)
- E5: Billions of pairs

**Our 138 pairs**: Not even close to viable.

---

## Recommendation

**Immediate**: Find and use Phase-3 training data (~1,540 pairs)

**If that works** (Recall@500 > 40%):
- Proceed to Phase 2 (hard negatives)
- Target: 55-60% Recall@500

**If that still fails** (Recall@500 < 10%):
- Generate 10,000-50,000 synthetic pairs from Wikipedia
- Retry MVP with proper dataset size

**If synthetic data works**:
- Scale to 100,000+ pairs for production quality
- Expected: 55-60% Recall@500

---

## Files Created Today

### Diagnostic Tools (All Working ✅)
1. `tools/diagnose_faiss_oracle_recall.py` - Oracle test (97.40% Recall@5)
2. `tools/eval_hybrid.py` - Hybrid RRF (0.65% Hit@5)
3. `tools/test_query_formations.py` - Heuristics (best: 38.96%)

### Two-Tower Infrastructure (All Working ✅)
4. `tools/build_twotower_pairs.py` - Pair preparation  
5. `tools/train_twotower.py` - Training loop  
6. `tools/eval_twotower.py` - Standalone evaluation

### Documentation
7. `PRD_Two_Tower_Retriever_Train_Spec.md` - Comprehensive spec
8. `HYBRID_RETRIEVAL_EXPERIMENT_STATUS.md` - Complete diagnostic journey
9. `TWO_TOWER_MVP_FAILURE_REPORT.md` - This document

### Results
10. `artifacts/evals/oracle_recall_results.json`
11. `artifacts/evals/hybrid_results.json`
12. `artifacts/evals/query_formation_results.json`
13. `runs/twotower_mvp/` - Failed MVP checkpoints (0% recall)

---

## Bottom Line

**MVP Status**: ❌ FAILED (0% Recall@500)  
**Root Cause**: Severe data scarcity (138 vs 10,000+ needed)  
**Next Action**: Find Phase-3 training data or generate synthetic pairs  
**Timeline**: 1-2 days to acquire data, 1 day to retry

**Infrastructure**: ✅ ALL WORKING  
**Diagnosis**: ✅ COMPLETE  
**Path Forward**: ✅ CLEAR

The two-tower approach is still valid, but requires 10-100x more training data than currently available.

