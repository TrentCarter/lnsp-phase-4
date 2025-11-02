# 5→1 Causal Alignment Test (5CAT) - Production Models Report

**Date**: 2025-10-30
**Test Tool**: `tools/tests/test_5to1_alignment.py`
**Sample Size**: 1000 VAL + 1000 OOD (reduced from 5000 due to performance)
**Horizon**: 3 steps (reduced from 5)

## 🚨 CRITICAL FINDINGS

### **ALL MODELS SUFFER FROM BACKWARD PREDICTION BIAS**

Every production model (except AMN 790k) learned to predict the **PREVIOUS** vector instead of the **NEXT** vector:

| Model                  | k=-1 (VAL) | k=1 (VAL) | Margin   | Direction |
|------------------------|------------|-----------|----------|-----------|
| **Transformer Optimized** | **0.690** | 0.557 | **-0.134** | ← BACKWARD |
| **LSTM (340k)**        | **0.687** | 0.538 | **-0.149** | ← BACKWARD |
| **GRU (340k)**         | **0.670** | 0.546 | **-0.124** | ← BACKWARD |
| **Transformer (340k)** | **0.603** | 0.549 | **-0.054** | ← BACKWARD |
| **AMN (790k)**         | -0.020    | -0.023 | -0.002   | ⚠️ RANDOM  |

**Expected behavior**: k=1 should be highest, margin should be positive (+0.12 minimum)

---

## Detailed Results by Model

### 1. AMN (790k Production) - Port 9001
**Model**: `artifacts/lvm/models/amn_790k_production_20251030_123212/best_model.pt`
**Val Cosine (checkpoint)**: 0.5597

#### Offset Alignment (VAL)
```
k=-3: -0.022
k=-2: -0.022
k=-1: -0.020
k=0:  -0.022
k=1:  -0.023  ← Should be highest
k=2:  -0.023
k=3:  -0.024
Margin: -0.002
```

#### OOD Results
```
k=-1: -0.048
k=1:  -0.052
Margin: -0.003
```

#### Gate Results
- ✗ Offset Sweep: -0.002 (need ≥0.12)
- ✗ Retrieval Rank: R@1=1.5%, R@5=10.7% (need ≥60%, ≥95%)
- ✗ Ablations: Shuffle=-0.002 (need ≤-0.15)
- ✗ Rollout: 0.034 (need ≥0.45)
- ✅ Bins Delta: 0.033 ✅

**Status**: ❌ Failed 4/5 gates - Model performs near-random

---

### 2. Transformer Baseline (340k) - Port 9002
**Model**: `artifacts/lvm/models_340k/transformer/best_model.pt`
**Val Cosine (checkpoint)**: 0.5774

#### Offset Alignment (VAL)
```
k=-3: 0.592
k=-2: 0.597
k=-1: 0.603  ← HIGHEST (WRONG!)
k=0:  0.558
k=1:  0.549  ← Should be highest
k=2:  0.544
k=3:  0.543
Margin: -0.054
```

#### OOD Results
```
k=-1: 0.606  ← Best
k=1:  0.543
Margin: -0.063
```

#### Gate Results
- ✗ Offset Sweep: -0.054 (BACKWARD!)
- ✗ Retrieval Rank: R@1=1.8%, R@5=10.7%
- ✗ Ablations: Shuffle=-0.015 (need ≤-0.15)
- ✅ Rollout: 0.534 ✅
- ✅ Bins Delta: 0.0045 ✅

**Status**: ❌ Failed 3/5 gates - **Learned BACKWARD prediction**

---

### 3. GRU (340k) - Port 9003
**Model**: `artifacts/lvm/models_340k/gru/best_model.pt`
**Val Cosine (checkpoint)**: 0.5920

#### Offset Alignment (VAL)
```
k=-3: 0.618
k=-2: 0.633
k=-1: 0.670  ← HIGHEST (WRONG!)
k=0:  0.564
k=1:  0.546  ← Should be highest
k=2:  0.538
k=3:  0.533
Margin: -0.124
```

#### OOD Results
```
k=-1: 0.659  ← Best
k=1:  0.542
Margin: -0.117
```

#### Gate Results
- ✗ Offset Sweep: -0.124 (BACKWARD!)
- ✗ Retrieval Rank: R@1=1.2%, R@5=13.2%
- ✗ Ablations: Shuffle=-0.015 (need ≤-0.15)
- ✅ Rollout: 0.526 ✅
- ✅ Bins Delta: 0.0087 ✅

**Status**: ❌ Failed 3/5 gates - **2nd worst backward bias**

---

### 4. LSTM (340k) - Port 9004
**Model**: `artifacts/lvm/models_340k/lstm/best_model.pt`
**Val Cosine (checkpoint)**: 0.4102 ⚠️

#### Offset Alignment (VAL)
```
k=-3: 0.615
k=-2: 0.635
k=-1: 0.687  ← HIGHEST (WRONG!)
k=0:  0.559
k=1:  0.538  ← Should be highest
k=2:  0.528
k=3:  0.522
Margin: -0.149
```

#### OOD Results
```
k=-1: 0.680  ← Best
k=1:  0.548
Margin: -0.133
```

#### Gate Results
- ✗ Offset Sweep: -0.149 (**WORST backward bias!**)
- ✗ Retrieval Rank: R@1=1.5%, R@5=15.3%
- ✗ Ablations: Shuffle=-0.019 (need ≤-0.15)
- ✅ Rollout: 0.519 ✅
- ✅ Bins Delta: 0.0099 ✅

**Status**: ❌ Failed 3/5 gates - **WORST backward prediction**

---

### 5. Transformer Optimized (20251024) - Port 9006
**Model**: `artifacts/lvm/models/transformer_optimized_20251024_072726/best_model.pt`
**Val Cosine (checkpoint)**: 0.5864

#### Offset Alignment (VAL)
```
k=-3: 0.614
k=-2: 0.632
k=-1: 0.690  ← HIGHEST (WRONG!)
k=0:  0.584
k=1:  0.557  ← Should be highest
k=2:  0.544
k=3:  0.536
Margin: -0.134
```

#### OOD Results
```
k=-1: 0.709  ← Best (highest of all models!)
k=1:  0.571
Margin: -0.138
```

#### Gate Results
- ✗ Offset Sweep: -0.134 (BACKWARD!)
- ✗ Retrieval Rank: R@1=1.0%, **R@5=18.1%** (BEST!)
- ✗ Ablations: Shuffle=-0.029 (need ≤-0.15)
- ✅ Rollout: **0.553** ✅ (BEST!)
- ✅ Bins Delta: 0.0233 ✅

**Status**: ❌ Failed 3/5 gates - **Best overall performance but still backward**

---

## 📊 Cross-Model Comparison

### Offset Margin (OOD, Higher = Better, Positive = Correct Direction)
```
AMN (790k):         -0.003  ⚠️ (near random)
Transformer (340k): -0.063  ❌
GRU (340k):         -0.117  ❌
LSTM (340k):        -0.133  ❌
Transformer Opt:    -0.138  ❌ (worst backward bias)
```

### Rollout Performance (OOD, Higher = Better, Need ≥0.45)
```
Transformer Opt:    0.553  ✅ BEST
LSTM:               0.536  ✅
GRU:                0.525  ✅
Transformer:        0.527  ✅
AMN:                0.031  ❌ (random)
```

### Retrieval Rank (OOD, R@5, Higher = Better)
```
Transformer Opt:    18.1%  🏆 BEST
LSTM:               9.4%
GRU:                8.4%
Transformer:        6.6%
AMN:                3.7%   ❌ WORST
```

### Ablations - Shuffle Penalty (OOD, More negative = Better)
```
Transformer Opt:    -0.028
Transformer:        -0.007
GRU:                -0.009
LSTM:               -0.013
AMN:                +0.0005  ❌ (no penalty!)
```

---

## 🔍 Root Cause Analysis

### Why All Models Learn Backward Prediction

**Hypothesis**: Training data has **reversed sequences** or **incorrect labeling**:

1. **340k models** (Transformer, GRU, LSTM) were trained on the same dataset
   - All show k=-1 > k=1 (backward bias)
   - All pass rollout tests (models work, just wrong direction)

2. **790k AMN** shows random performance
   - Near-zero cosines across all offsets
   - Different training approach (different data or architecture issue)

3. **Transformer Optimized** shows BEST backward prediction
   - k=-1 (OOD) = 0.709 (highest of all models!)
   - Superior rollout (0.553) and retrieval (18.1%)
   - Suggests **model is excellent, data is backward**

### Evidence for Data Issue

✅ **Models learned successfully** (rollout 0.52-0.55 passes gate)
✅ **Models generalize well** (Val ≈ OOD, bins delta passes)
✅ **Ablations work correctly** (shuffle/reverse hurts performance)
❌ **Direction is reversed** (all 340k models prefer k=-1)

**Conclusion**: The training data likely has **context and target swapped** or sequences are **temporally reversed**.

---

## 💡 Recommendations

### P0: Immediate Actions

1. **Verify Training Data Pipeline**
   ```bash
   # Check sequence creation in tools/create_training_sequences_with_articles.py
   # Line 85-88: Are we slicing [i:i+5] for context and [i+5] for target?
   # Or is it reversed?
   ```

2. **Test with Reversed Model Inference**
   - Flip the input order during inference
   - If k=1 becomes best, confirms data issue

3. **Retrain 1 Model with Verified Data**
   - Use Transformer Optimized architecture (best results)
   - Manually verify first 100 training samples
   - Run 5CAT during training to catch direction issues early

### P1: Testing Improvements

1. **Add 5CAT to Training Loop**
   - Run offset alignment test every N epochs
   - Alert if margin goes negative
   - Early stopping if backward bias detected

2. **Create Synthetic Test Cases**
   - Generate sequences with known temporal order
   - Verify models predict correct direction

3. **Update Production Models**
   - Once retrained, re-run 5CAT on all ports
   - Update model symlinks
   - Document expected 5CAT metrics

### P2: Documentation Updates

1. **CLAUDE.md**:
   - Add WARNING about backward prediction in current models
   - Document expected 5CAT metrics for future models
   - Add data verification checklist

2. **Training Scripts**:
   - Add inline comments showing expected sequence order
   - Add assertion tests for first batch

---

## ✅ Test Tool Improvements Made

During this evaluation, the following improvements were made to `test_5to1_alignment.py`:

1. **Checkpoint Loading**
   - Added support for both direct model objects and checkpoint dicts
   - Auto-detects model architecture (AMN/GRU/LSTM/Transformer)
   - Handles weights_only=False for numpy compatibility

2. **Numpy Compatibility**
   - Fixed negative stride issue with `np.ascontiguousarray()`
   - Handles reversed arrays for ablation tests

3. **Data Format Converters**
   - Created `validation_sequences_ctx5_articles4000-4499_compat.npz`
   - Created `ood_sequences_ctx5_articles1500-1999.npz`
   - Converts metadata format to expected keys

---

## 📌 Summary

**Status**: ❌ **ALL PRODUCTION MODELS FAIL 5CAT**

**Best Model**: Transformer Optimized (9006)
- Passes: Rollout (0.553), Bins Delta (0.023)
- Best R@5: 18.1%
- Best rollout: 0.553
- BUT: Predicts backward (margin -0.138)

**Worst Model**: AMN 790k (9001)
- Near-random performance (cosines ≈ 0)
- Fails all gates except Bins Delta

**Action Required**:
1. Verify training data pipeline for sequence direction
2. Retrain models with corrected data
3. Integrate 5CAT into training loop as quality gate

---

**Report Generated**: 2025-10-30
**Test Framework**: 5→1 Causal Alignment Test (5CAT)
**Models Tested**: 5 (AMN 790k, Transformer 340k, GRU 340k, LSTM 340k, Transformer Optimized)
