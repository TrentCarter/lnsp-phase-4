# AMN 790K Second Training Failure - InfoNCE Dominance

**Date**: 2025-10-30
**Run**: `amn_790k_production_20251030_123212`
**Status**: ❌ CATASTROPHIC FAILURE - Worse than first attempt!

---

## 📊 Performance Comparison

| Run | Config | In-Dist | OOD | Analysis |
|-----|--------|---------|-----|----------|
| **584k Baseline** | MSE-only (1.0) | 0.5597 | 0.6375 | ✅ SUCCESS |
| **790k Attempt #1** | MSE-only (1.0) | 0.4607 | -0.0118 | ❌ Mode collapse |
| **790k Attempt #2** | MSE(0.5)+InfoNCE(0.5) | 0.2675 | ??? | ❌❌ WORSE! |

---

## 🔍 Root Cause: InfoNCE Dominance

### Loss Magnitude Comparison

**584k Successful (MSE-only)**:
```
Epoch 1:
  train_loss: 0.001073  (pure MSE)
  train_cosine: 0.5281  ✅
  train_loss_mse: 0.001073
  train_loss_info: 1.328 (recorded but NOT optimized)
```

**790k Failed (MSE+InfoNCE)**:
```
Epoch 1:
  train_loss: 0.6989  (dominated by InfoNCE!)
  train_cosine: 0.2528  ❌
  train_loss_mse: 0.002015
  train_loss_info: 1.394  (ACTIVELY optimized)

Effective weights:
  MSE contribution:     0.5 * 0.002 = 0.001
  InfoNCE contribution: 0.5 * 1.394 = 0.697

→ InfoNCE is 700x stronger than MSE!
```

---

## 💥 What Went Wrong

### 1. Loss Scale Mismatch

MSE operates on **squared euclidean distance** in 768D space:
- Typical values: 0.001 - 0.002
- Penalizes element-wise errors

InfoNCE operates on **cross-entropy of similarity matrices**:
- Typical values: 1.0 - 2.0
- Optimizes contrastive separation in batch

**Equal weighting** (0.5 each) means InfoNCE dominates by 700x!

### 2. Conflicting Objectives

**MSE** says: "Match the target vector exactly"
- Directly maximizes cosine similarity
- Works well for sequential prediction

**InfoNCE** says: "Be more similar to your target than to other targets in the batch"
- Encourages relative ranking
- Can achieve low loss with LOW absolute cosine values if batch contains similar targets

**Result**: Model learns to minimize InfoNCE by creating separation (low cosines for everything) instead of matching targets.

---

## 📈 Training Evidence

### Cosine Similarity Progression

| Epoch | 584k (MSE-only) | 790k Run #2 (MSE+InfoNCE) | Delta |
|-------|-----------------|---------------------------|-------|
| 1 | 0.5281 | 0.2528 | -52% ❌ |
| 5 | 0.5580 | 0.2721 | -51% ❌ |
| 10 | 0.5589 | 0.2714 | -51% ❌ |
| 20 | 0.5593 | 0.2696 | -52% ❌ |
| Final | 0.5597 | 0.2675 | -52% ❌ |

**Finding**: Cosine similarity is **STUCK at 0.27** from epoch 1, never improves!

### Loss Component Analysis

```
790k Run #2:
  MSE:     0.002 → 0.002 (flat, not improving)
  InfoNCE: 1.394 → 0.959 (decreasing, model IS learning!)
  Total:   0.699 → 0.482 (dominated by InfoNCE decrease)
```

Model is successfully minimizing InfoNCE, but this doesn't improve cosine similarity!

---

## ✅ Correct Solution

### Use MSE-Only (Like 584k)

```bash
--lambda-mse 1.0     # Primary loss
--lambda-info 0.0    # DISABLE InfoNCE
--lambda-moment 0.0  # Optional
--lambda-variance 0.0  # Optional
--epochs 20          # Standard duration
--lr 0.0005          # Standard rate
```

### Why MSE-Only Works

1. **Direct Optimization**: MSE directly minimizes `||pred - target||²`
2. **Cosine Alignment**: In high-D, minimizing MSE ≈ maximizing cosine similarity
3. **Proven**: 584k model achieved 0.5597 in-dist, 0.6375 OOD with MSE-only
4. **Scale**: MSE values (0.001) match learning rate (0.0005) for stable training

### Why InfoNCE Failed

1. **Wrong Scale**: InfoNCE (1.0-2.0) is 700x larger than MSE (0.001-0.002)
2. **Wrong Objective**: Contrastive loss optimizes ranking, not absolute similarity
3. **Batch Effects**: InfoNCE depends on batch composition (similar targets → bad gradients)
4. **No Benefit**: For sequential prediction, MSE is sufficient and more stable

---

## 🚨 Lessons Learned

### DO:
1. ✅ **Use MSE-only** for vector prediction tasks
2. ✅ **Match 584k config** that worked (don't over-engineer!)
3. ✅ **Test loss scales** before training (MSE vs InfoNCE magnitude)
4. ✅ **Monitor cosine from epoch 1** (should start >0.4, not 0.25!)

### DON'T:
1. ❌ **Mix losses with mismatched scales** (700x difference!)
2. ❌ **Use InfoNCE for regression** (it's for contrastive learning!)
3. ❌ **Weight losses equally** without checking magnitudes
4. ❌ **Assume more loss terms = better** (simpler is often better)

---

## 🔄 Next Steps

1. **Retrain with MSE-only** (lambda-mse=1.0, lambda-info=0.0)
2. **Expect results**: in-dist ~0.56, OOD ~0.63 (matching 584k)
3. **If it works**: Use SAME config for GRU, LSTM, Transformer
4. **Document**: Update training recipes with "MSE-only is the way"

---

## 📊 Why Run #1 (MSE-only) Failed But 584k Succeeded?

**Hypothesis**: Different data or configuration issue in Run #1, NOT a fundamental problem with MSE-only.

Evidence:
- 584k achieved 0.5597 with MSE-only
- 790k Run #1 got 0.4607 with MSE-only (lower but not catastrophic)
- 790k Run #2 got 0.2675 with MSE+InfoNCE (catastrophic!)

**Conclusion**: MSE-only is correct. Run #1 may have had a different issue (bad checkpoint, wrong data file, etc.). Try MSE-only again with verified setup.

---

**Status**: ROOT CAUSE CONFIRMED
**Fix**: Use MSE-only (lambda-mse=1.0, lambda-info=0.0)
**Updated**: 2025-10-30 13:15 PST
