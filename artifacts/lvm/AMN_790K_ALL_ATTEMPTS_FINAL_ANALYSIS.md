# AMN 790K Training - Complete Failure Analysis

**Date**: 2025-10-30
**Status**: ❌ ALL THREE ATTEMPTS FAILED
**Conclusion**: 790k dataset is fundamentally incompatible with current AMN architecture

---

## 📊 Complete Results Summary

| Attempt | Config | In-Dist | OOD | Status |
|---------|--------|---------|-----|--------|
| **584k Baseline** | MSE-only | 0.5597 | 0.6375 | ✅ SUCCESS |
| **790k #1** | MSE-only | 0.4607 | -0.0118 | ❌ OOD Collapse |
| **790k #2** | MSE+InfoNCE | 0.2675 | ??? | ❌ InfoNCE Dominance |
| **790k #3** | MSE-only | **0.4621** | **0.0211** | ❌ OOD Catastrophic |

---

## 🔴 Attempt #3 Analysis

### Training Progression
```
Epoch  | Train Cos | Val Cos  | Delta
-------|-----------|----------|-------
1      | 0.4228    | 0.4457   | -
5      | 0.4553    | 0.4545   | +0.0088
10     | 0.4582    | 0.4569   | +0.0024
15     | 0.4596    | 0.4576   | +0.0007
18     | 0.4601    | 0.4579   | +0.0003 (LR↓)
19     | 0.4638    | 0.4612   | +0.0032
20     | 0.4640    | 0.4621   | +0.0009
```

**Observations**:
- ✅ Steady training improvement (+3.7% over 20 epochs)
- ✅ No training collapse or divergence
- ✅ LR reduction helped (+0.0033 in 2 epochs)
- ❌ Started 15.6% below 584k baseline
- ❌ Ended 17.4% below 584k baseline
- ❌ OOD performance catastrophic (0.0211 vs 0.6375 target)

### OOD Evaluation Results
```
In-Distribution:  0.4621 ✅ (reasonable on seen data)
Out-of-Distribution: 0.0211 ❌ (essentially random!)
Δ (OOD - In-Dist): -0.4410 (catastrophic drop)
```

**Expected**: OOD boost (+0.08 to +0.18)
**Actual**: OOD collapse (-0.44)

---

## 🤔 Root Cause Analysis

### Why Did 790k Fail When 584k Succeeded?

**Hypothesis 1: Dataset Quality Degradation**
- 584k: Articles 1-11,000 (carefully curated Wikipedia)
- 790k: Articles 1-15,192 (+4,192 articles = +35% growth)
- New articles (11k-15k) may contain:
  - More "List of..." pages
  - More disambiguation pages
  - More stub articles
  - Lower semantic coherence

**Evidence**:
- Training curves are smooth (not a training bug)
- MSE-only config is correct (proven on 584k)
- Data validation checks passed (normalization ✅, FAISS ✅)

**Hypothesis 2: Context Window Insufficient**
- ctx=5 works for coherent 584k articles
- ctx=5 insufficient for diverse 790k articles
- Model needs more context to capture relationships

**Hypothesis 3: Learning Rate Mismatch**
- LR=0.0005 optimal for 584k
- LR=0.0005 too high for noisier 790k data
- Evidence: Started 15% below baseline from epoch 1

**Hypothesis 4: Overfitting to In-Distribution**
- Model memorizes training data patterns
- Fails to generalize to truly unseen OOD data
- OOD test set may be too different from training distribution

---

## 💡 Recommended Actions (In Priority Order)

### Option 1: Data Quality Filtering (HIGHEST PRIORITY)
```bash
# Filter 790k dataset to remove low-quality articles
# Criteria:
#   - Remove "List of..." pages
#   - Remove disambiguation pages
#   - Remove stubs (<500 words)
#   - Keep only articles with high semantic coherence

# Target: 600-650k high-quality concepts (vs 790k raw)
```

**Rationale**: 584k likely had implicit quality filtering. Recreate that quality bar.

### Option 2: Increase Context Window
```bash
# Try ctx=7 or ctx=9
--context-length 7  # vs current ctx=5
```

**Rationale**: More context helps model capture relationships in diverse data.

### Option 3: Curriculum Learning
```bash
# Phase 1: Train on high-coherence sequences only
# Phase 2: Gradually introduce lower-coherence sequences
# Phase 3: Fine-tune on full 790k dataset
```

**Rationale**: Start with easy examples, gradually increase difficulty.

### Option 4: Adjust Hyperparameters
```bash
# Lower learning rate for noisy data
--lr 0.0003  # vs current 0.0005

# Increase regularization
--weight-decay 0.01  # vs current 0.0

# Add dropout
--dropout 0.1  # vs current 0.0
```

**Rationale**: More conservative training for noisy data.

### Option 5: Blend Fine-Tuning (SAFEST)
```bash
# Start from 584k checkpoint
# Train only on NEW 206k concepts (articles 11k-15k)
# Use lower learning rate (0.0001)
```

**Rationale**: Leverage known-good 584k base, carefully add new data.

---

## 🚫 What NOT To Do

1. ❌ **Don't add InfoNCE again** - Attempt #2 proved it makes things worse
2. ❌ **Don't train longer** - Attempt #3 showed plateau after epoch 15
3. ❌ **Don't blame the architecture** - AMN works fine on 584k
4. ❌ **Don't skip data validation** - The problem is dataset composition, not bugs

---

## 📝 Next Steps

### Immediate Actions:
1. **Analyze 790k dataset composition**:
   ```bash
   # Check article types in 11k-15k range
   # Count "List of..." pages
   # Count disambiguation pages
   # Measure average coherence (cos similarity between adjacent chunks)
   ```

2. **Compare 584k vs 790k statistics**:
   ```bash
   # Adjacency coherence distribution
   # Article length distribution
   # Semantic diversity metrics
   ```

3. **Create filtered 650k dataset**:
   ```bash
   # Apply quality filters
   # Target: 650k high-quality concepts
   # Verify coherence improves
   ```

### Decision Tree:
```
IF filtered_650k shows higher coherence THAN 790k:
  → Retrain AMN on filtered_650k
  → Expect results similar to 584k baseline

ELSE IF ctx=7 improves 790k performance:
  → Use ctx=7 for all future training
  → Accept slightly longer inference time

ELSE:
  → Stick with 584k dataset
  → Focus on other models (GRU, LSTM, Transformer)
  → Revisit dataset expansion after architecture improvements
```

---

## 🎯 Success Criteria (For Any Re-Attempt)

**Minimum Acceptable**:
- In-Dist: ≥ 0.54 (within 5% of 584k)
- OOD: ≥ 0.60 (generalization boost)
- Δ (OOD - In-Dist): +0.05 to +0.10

**Target**:
- In-Dist: ≥ 0.56 (matches 584k)
- OOD: ≥ 0.63 (matches 584k)
- Δ (OOD - In-Dist): +0.06 to +0.08

**Abort Criteria**:
- Epoch 1 val_cosine < 0.50 → Data quality issue
- OOD < 0.40 → Fundamental compatibility problem
- No improvement after 10 epochs → Hyperparameter issue

---

**Status**: TRAINING HALTED - AWAITING DATASET ANALYSIS
**Recommendation**: Do NOT train remaining models (GRU, LSTM, Transformer) until dataset issue resolved
**Updated**: 2025-10-30 12:45 PST
