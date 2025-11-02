# P4 Rollout Failure Report

**Date**: 2025-11-01
**Model**: Transformer with Multi-Step Rollout + Adaptive Guards
**Verdict**: ❌ **FAILED - Same collapse pattern as V3, backward bias persists**

---

## Executive Summary

P4 Rollout approach failed catastrophically with the **SAME collapse pattern as V3**:
- ✅ Epochs 1-3 (pure MSE): Val cosine 0.540 (good, matches P1)
- ❌ **Epoch 4 (rollout starts): Val cosine collapsed to 0.338 (-37% in ONE epoch)**
- ❌ 5CAT Results: **2/5 gates passed**, margin **-0.149** (backward bias), R@1 **1.04%**

**Critical Finding**: The backward bias (-0.149 margin) exists in the **EPOCH 3 model** (pure MSE, before rollout/guards), suggesting the issue is in the base MSE training or data, NOT the directional losses.

---

## Training Results

### Epoch-by-Epoch Performance

| Epoch | Val Cosine | Val Loss | Train Cosine | Phase | Status |
|-------|-----------|----------|--------------|-------|--------|
| **1** | 0.455 | 0.001420 | 0.364 | Warm-up (MSE only) | ✅ Learning |
| **2** | 0.521 | 0.001249 | 0.507 | Warm-up (MSE only) | ✅ Improving |
| **3** | **0.540** | **0.001199** | 0.543 | Warm-up (MSE only) | ✅ **BEST MODEL (saved)** |
| **4** | **0.338** | 0.001724 | 0.391 | Rollout+Guards START | ❌ **COLLAPSE (-0.202 / 37%)** |
| **5** | 0.381 | 0.001611 | 0.340 | Rollout+Guards | ⚠️ Partial recovery |
| **10** | 0.404 | 0.001308 | 0.412 | Rollout+Guards+Future | ⚠️ Struggling |
| **15** | 0.397 | 0.001384 | 0.408 | Full pipeline | ⚠️ Unstable |
| **20** | 0.335 | 0.001731 | 0.405 | Full pipeline | ❌ Still collapsed |

### Key Metrics

- **Best Val Loss**: 0.001199 (epoch 3)
- **Best Val Cosine**: 0.540 (epoch 3)
- **Final Val Cosine**: 0.335 (epoch 20)
- **Final Train Cosine**: 0.405 (epoch 20)

---

## 5CAT Validation Results

**Model Tested**: `best_model.pt` (epoch 3, pure MSE, before rollout/guards)

### Gate Results: **2/5 PASSED**

| Gate | VAL Result | OOD Result | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| **A: Offset Sweep** | **Margin: -0.149** | **Margin: -0.152** | ≥+0.12 (VAL), ≥+0.10 (OOD) | ❌ **BACKWARD BIAS** |
| **B: Retrieval** | R@1: 1.04%, R@5: 22.12%, MRR: 0.129 | R@1: 0.86%, R@5: 18.04%, MRR: 0.104 | R@1 ≥60%, R@5 ≥95%, MRR ≥0.80 | ❌ **TERRIBLE** |
| **C: Ablations** | Shuffle: -0.028, Reverse: -0.049 | Shuffle: -0.023, Reverse: -0.038 | Deltas ≤ -0.15 | ❌ **STRUCTURE HURTS** |
| **D: Rollout** | 0.464 | 0.468 | ≥0.45 (VAL), ≥0.42 (OOD) | ✅ **PASSED** |
| **E: Generalization** | Low: 0.525, Normal: 0.548 | Low: 0.539, Normal: 0.555 | abs(Δ) ≤ 0.05 | ✅ **PASSED** |

### Detailed Offset Sweep (Gate A)

**VAL**:
```
k=-3: 0.578  k=-2: 0.608  k=-1: 0.670  k=0: 0.548  k=+1: 0.521  k=+2: 0.511  k=+3: 0.504
                           ↑ PEAK (should be at k=+1)
Margin(+1 vs -1): 0.521 - 0.670 = -0.149 ❌
```

**OOD**:
```
k=-3: 0.584  k=-2: 0.614  k=-1: 0.675  k=0: 0.550  k=+1: 0.523  k=+2: 0.510  k=+3: 0.501
                           ↑ PEAK (should be at k=+1)
Margin(+1 vs -1): 0.523 - 0.675 = -0.152 ❌
```

**Interpretation**: Model predicts **PREVIOUS** vector (k=-1) more accurately than **NEXT** vector (k=+1). This is the opposite of desired behavior.

---

## Comparison to Previous Approaches

| Approach | Epoch 3 Val Cos | Epoch 4 Val Cos | Final Val Cos | Final Margin | 5CAT Gates | Verdict |
|----------|----------------|----------------|---------------|--------------|------------|---------|
| **P1 Baseline** | 0.546 | 0.546 | 0.550 | ~0.0 (neutral) | Unknown | ✅ Stable |
| **V3 Guards** | 0.540 | 0.354 | 0.354 | -0.132 | 0/5 | ❌ Collapsed |
| **P4 Rollout** | 0.540 | 0.338 | 0.335 | -0.149 | 2/5 | ❌ **SAME COLLAPSE** |

**Key Insight**: P4 and V3 have **IDENTICAL collapse patterns**:
- Both reach val_cos ~0.540 at epoch 3
- Both **collapse to ~0.34** at epoch 4 when directional losses start
- Both fail 5CAT with negative margins

---

## Root Cause Analysis

### Finding 1: Rollout/Guards Cause Catastrophic Collapse

**Evidence**:
- Epochs 1-3: Val cosine improves (0.455 → 0.521 → 0.540) ✅
- **Epoch 4**: Val cosine collapses (0.540 → 0.338) ❌ **(-37% in ONE epoch!)**
- Epoch 4 is when rollout+guards activate per curriculum

**Mechanism**:
1. Multi-step rollout loss penalizes flat trajectories (ŷ₁≈ŷ₂≈ŷ₃)
2. Adaptive guards penalize high-similarity predictions
3. Combined, these push predictions TOO FAR from targets
4. Model can't recover even after 16 more epochs (stays at ~0.34)

**Comparison to V3**: V3 used strong static guards (λ=0.01), P4 used adaptive guards + rollout. Both caused the same collapse, suggesting **ANY strong directional penalty causes this pattern**.

### Finding 2: Backward Bias Exists BEFORE Rollout/Guards

**Critical Discovery**: The "best_model.pt" saved at **epoch 3** (pure MSE, NO rollout/guards) shows:
- Val cosine: 0.540 ✅ (good)
- **Margin: -0.149** ❌ (backward bias!)
- R@1: 1.04% ❌ (terrible retrieval)

**Implication**: The backward bias is NOT caused by directional losses! It exists in the base MSE training.

**Questions Raised**:
1. Why does P1 (pure MSE, 20 epochs) have margin ~0.0, but P4 epoch 3 (pure MSE, 3 epochs) has margin -0.149?
2. Is the backward bias baked into the training data?
3. Does MSE training need MORE epochs to converge to neutral margin?
4. Or is there a fundamental issue with how we're generating training sequences?

### Finding 3: Curriculum Implementation Was Correct

**Verified**: The curriculum code correctly sets `lambda_roll = 0.0` for epochs 1-3. The collapse at epoch 4 is NOT due to a curriculum bug.

---

## Why P4 Failed: Technical Breakdown

### The Theory (P4 Approach Document)

**Goal**: Multi-step rollout makes copying fail globally
- Copying → flat trajectories (ŷ₁≈ŷ₂≈ŷ₃) → high penalty
- Forward momentum → diverse predictions → low penalty
- Adaptive guards boost strength on high-similarity samples

**Curriculum**:
1. Warm-up (1-3): Pure MSE
2. Rollout (4-6): Add rollout loss (H=3, λ=0.05-0.10)
3. Rollout+Guards (7+): Add adaptive directional guards (λ_dir=0.002)
4. Full (10+): Add future ranking loss (λ_fut=0.002)

### The Reality (What Actually Happened)

**Epoch 4 Collapse**:
- Rollout loss activated with H=3, λ_roll=0.05
- Model predictions suddenly pushed far from targets
- Val cosine: 0.540 → 0.338 (-37%)
- Model never recovered (stuck at ~0.34 for remaining 16 epochs)

**Why Rollout Loss Backfired**:
1. **Too aggressive too soon**: Even λ=0.05 was too strong after only 3 epochs of MSE
2. **Destabilized learned patterns**: MSE had learned a stable (if backward-biased) pattern, rollout destroyed it
3. **No recovery mechanism**: Once predictions diverged, MSE alone couldn't pull them back
4. **Curriculum too fast**: Should have ramped up rollout weight MUCH more gradually (e.g., 0.001 → 0.005 → 0.01 over many epochs)

**Adaptive Guards Never Got a Chance**:
- Guards activate at epoch 7
- But model already collapsed at epoch 4
- Guards had nothing to work with (val_cos already at 0.38)

---

## Lessons Learned

### 1. Directional Losses Are Too Fragile

**Pattern Observed Across V3, P2, P3, P4**:
- **V3**: Strong guards (λ=0.01) → collapse at epoch 4
- **P2**: Residual architecture → margin -0.534 (worse than baseline)
- **P3**: Tiny guards (λ=0.002) → marginal improvement (-0.064) but still negative
- **P4**: Rollout + adaptive guards → **SAME collapse as V3** at epoch 4

**Conclusion**: ANY directional penalty strong enough to affect margin causes catastrophic collapse. Penalties weak enough to avoid collapse are too weak to fix backward bias.

### 2. Backward Bias May Be in the Data, Not the Model

**Evidence**:
- P4 epoch 3 (pure MSE): margin -0.149
- P1 baseline (pure MSE, 20 epochs): margin ~0.0
- Difference suggests data quality or sampling randomness

**Hypothesis**: Training sequences may have inherent backward bias:
- If chunk[i-1] is naturally more similar to context[0:5] than chunk[i+1]
- MSE will learn to predict closer to i-1
- This could be due to: topic drift, semantic coherence patterns, or data artifacts

**Next Step**: Run data diagnostic on training sequences:
```bash
./.venv/bin/python tools/tests/diagnose_data_direction.py \
  artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --n-samples 5000
```

### 3. Need to Compare P1 and P4 Epoch 3 on 5CAT

**Current Gap**: We don't have P1's 5CAT results to compare apples-to-apples

**Action Item**: Run 5CAT on P1 baseline to see:
- Is P1's margin truly 0.0 or also negative?
- Is P1's retrieval good (60%+ R@1) or also poor?
- Does P1 pass more than 2/5 gates?

```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_baseline_p1/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

---

## Recommendations

### SHORT-TERM: Validate Data Quality

1. **Run data diagnostic** on training sequences to check for inherent backward bias
2. **Run 5CAT on P1** to establish true baseline performance
3. **Inspect training data** for potential issues:
   - Are sequences properly ordered (chunk_i → chunk_i+1)?
   - Is there topic drift causing i-1 to be more similar than i+1?
   - Are there data contamination issues?

### MEDIUM-TERM: Rethink Approach

**If data diagnostic shows backward bias in data**:
- Fix data generation pipeline
- Regenerate training sequences
- Retrain P1 baseline on clean data
- Re-evaluate directional losses only AFTER baseline is healthy

**If data is clean but P1 also has negative margin**:
- MSE alone cannot learn forward directionality
- Need fundamentally different approach (not just loss modifications)
- Consider:
  - Autoregressive training (predict h steps ahead explicitly)
  - Contrastive learning (push away from k=-1, pull toward k=+1)
  - Data augmentation (flip sequences and negate margin)

**If P1 has neutral/positive margin but P4 doesn't**:
- Investigate why 3 epochs isn't enough to converge
- Try longer warm-up period (10-15 epochs pure MSE)
- Then introduce directional losses MUCH more gradually

### LONG-TERM: Architecture Changes

**If loss-based approaches continue to fail**:
- Explore **positional encoding** that encodes temporal direction
- Use **causal masking** in attention to enforce forward-only dependencies
- Try **sequence-to-sequence** architecture (explicit target = input[t+1])
- Consider **reinforcement learning** approach where forward = reward, backward = penalty

---

## Verdict

**P4 Rollout: ❌ FAILED**

**Status**: DO NOT DEPLOY - Same catastrophic collapse as V3

**Recommendation**:
1. Run data diagnostic and P1 5CAT validation first
2. Understand root cause of backward bias before attempting more fixes
3. If backward bias is in data, fix data pipeline
4. If backward bias is algorithmic, need fundamentally different approach (not loss tweaks)

**Preserved Model**: `artifacts/lvm/models/transformer_p4_rollout/best_model.pt` (epoch 3, pure MSE, for analysis)

**Production Model**: Continue using P1 Baseline (port 9007) until root cause is resolved

---

**Generated**: 2025-11-01
**Session**: Post-mortem analysis of P4 Rollout training
**See Also**:
- `artifacts/lvm/TRAINING_SESSION_2025_11_01.md` (Full session summary)
- `artifacts/lvm/P4_ROLLOUT_APPROACH.md` (Original approach design)
- `CLAUDE.md` (Updated repository guidance)
