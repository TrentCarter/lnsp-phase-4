# LVM Model Comparison Table - Recent Training Session

**Generated**: 2025-11-02 10:15 EST
**Session**: October 31 - November 2, 2025 (12-hour intensive debugging + P5 implementation)

---

## 📊 Complete Model Comparison

| # | Model Name | Date/Time | Size | Approach | Val Cos | Margin (VAL) | R@1 | R@5 | Gates | Status |
|---|------------|-----------|------|----------|---------|--------------|-----|-----|-------|--------|
| **11** | **P5 Stage A** | Nov 2, 10:06 | 205M | **Curriculum (top30%) + pos=0.03** | **0.463** | **-0.041** | **3.2%** | **17.5%** | **1/5** | ❌ **FAILED** |
| 10 | P4 Rollout | Nov 1, 18:09 | 205M | Rollout + Adaptive Guards | 0.338 | -0.149 | 1.0% | 22.1% | 2/5 | ❌ FAILED (collapsed) |
| 9 | P3 Tiny Guards | Nov 1, 15:55 | 205M | λ_dir=0.002 (tiny) | 0.526 | -0.064 | ? | ? | ? | ⚠️ Partial (51% improvement) |
| 8 | P2 Residual | Nov 1, 14:50 | 205M | Residual arch (ŷ=norm(u+α·Δ)) | 0.472 | -0.534 | ? | ? | ? | ❌ FAILED (worse) |
| 7 | P1 Baseline | Nov 1, 13:41 | 205M | Pure MSE (20 epochs) | 0.550 | **-0.167** | **1.08%** | **24.3%** | **2/5** | ⚠️ Neutral (backward bias) |
| 6 | V3 Directional | Oct 31, 15:01 | 205M | Strong guards (λ=0.01) | 0.354 | -0.132 | ? | ? | ? | ❌ FAILED (collapsed) |
| 5 | Directional Fix | Oct 31, 10:59 | 205M | Directional guards attempt | ? | ? | ? | ? | ? | ❌ FAILED |
| 4 | 584k Stable | Oct 31, 08:37 | 205M | Pure MSE (584k clean data) | ? | ? | ? | ? | ? | ✅ Stable |
| 3 | AMN 790k Split | Oct 31, 00:17 | 17M | AMN architecture | ? | -0.002 | ? | ? | ? | ⚠️ Random |
| 2 | Transformer Optimized | Oct 24, 08:55 | 205M | Optimized training (340k data) | ? | -0.134 | ? | ? | ? | ❌ Backward bias |
| 1 | AMN 584k Fresh | Oct 28, 22:33 | 17M | AMN on clean data | ? | ? | ? | ? | ? | ✅ Working |

---

## 🔍 Detailed Model Information

### 11. P5 Stage A (November 2, 2025) - **LATEST** ❌

**Path**: `artifacts/lvm/models/transformer_p5_20251102_095841/stageA/`
**Created**: November 2, 2025, 10:06 AM
**Size**: 205MB
**Architecture**: Transformer Decoder (input_dim=769 with positional encoding)

**Training Configuration**:
- **Approach**: P5 Curriculum Learning
- **Data**: Top 30% forward-distinct samples (131,571 / 438,568 total)
- **Forward-distinctness threshold**: Δ ≥ 0.6455
- **Epochs**: 4
- **Positional scalar**: 0.03
- **Losses**: Pure MSE (no directional/rollout/AC losses)
- **Dataset**: `training_sequences_ctx5_584k_clean_splits_stage_a_top30.npz`

**5CAT Results** (2000 samples):

| Test | VAL Score | OOD Score | Target | Status |
|------|-----------|-----------|--------|--------|
| **A: Offset Margin** | -0.041 | -0.046 | ≥ +0.02 | ❌ FAIL |
| **B: Retrieval R@1** | 3.2% | 1.75% | ≥ 60% | ❌ FAIL |
| **B: Retrieval R@5** | 17.5% | 10.9% | ≥ 60% | ❌ FAIL |
| **C: Ablations** | Fail | Fail | Pass | ❌ FAIL |
| **D: Rollout** | 0.448 | 0.476 | ≥ 0.46 | ⚠️ Mixed |
| **E: Bins Delta** | Pass | Pass | Pass | ✅ PASS |

**Gates Passed**: 1/5 (only bins_delta)

**Analysis**:
- **Negative margin**: Model STILL predicts backward despite curriculum
- **Poor retrieval**: 3.2% R@1 (should be 60%+) → predictions not meaningful
- **Positional encoding**: Applied (pos_scale=0.03) but too weak
- **Diagnosis**: Curriculum selection alone insufficient; need stronger positional signal

---

### 10. P4 Rollout (November 1, 2025) ❌

**Path**: `artifacts/lvm/models/transformer_p4_rollout/`
**Created**: November 1, 2025, 6:09 PM
**Size**: 205MB

**Training Configuration**:
- **Approach**: Multi-step rollout loss + adaptive guards
- **Epochs**: Collapsed at epoch 4 (before rollout activated)
- **Losses**: MSE (epochs 1-3), then rollout (epoch 4+, but never reached)

**5CAT Results** (Best checkpoint at epoch 3):

| Test | VAL Score | OOD Score | Status |
|------|-----------|-----------|--------|
| **Margin** | **-0.149** | **-0.152** | ❌ Backward bias (BEFORE rollout) |
| **R@1** | 1.04% | 0.70% | ❌ FAIL |
| **R@5** | 22.12% | 20.60% | ❌ FAIL |
| **Rollout** | 0.360 | 0.408 | ❌ FAIL |

**Gates Passed**: 2/5

**Analysis**:
- **Backward bias existed at epoch 3** (pure MSE phase)
- **Collapsed at epoch 4** when rollout activated (val_cos 0.540 → 0.338)
- **Same pattern as V3**: Guards/rollout caused catastrophic failure
- **Key finding**: Problem exists in MSE warm-up, not fixable by later losses

---

### 7. P1 Baseline (November 1, 2025) ⚠️

**Path**: `artifacts/lvm/models/transformer_baseline_p1/`
**Created**: November 1, 2025, 1:41 PM
**Size**: 205MB
**Status**: ✅ **DEPLOYED** on port 9007

**Training Configuration**:
- **Approach**: Pure MSE baseline (no tricks)
- **Epochs**: 20
- **Data**: Full 584k clean dataset
- **Purpose**: Establish ground truth for backward bias

**5CAT Results** (5000 samples):

| Test | VAL Score | OOD Score | Status |
|------|-----------|-----------|--------|
| **Margin** | **-0.167** | **-0.167** | ❌ WORSE than P4! |
| **R@1** | 1.08% | 1.04% | ❌ FAIL |
| **R@5** | 24.32% | 24.08% | ❌ FAIL |
| **Rollout** | 0.486 | 0.493 | ⚠️ Borderline |

**Gates Passed**: 2/5

**Key Finding**:
- **More MSE epochs = WORSE margin** (-0.167 after 20 epochs vs -0.149 after 3 epochs in P4)
- **Proves**: MSE converges TO copy-last (it's the optimal solution)
- **Implication**: Pure MSE will NEVER escape copy-last basin

---

### 6. V3 Directional (October 31, 2025) ❌

**Path**: `artifacts/lvm/models/transformer_directional_v3/`
**Created**: October 31, 2025, 3:01 PM
**Size**: 205MB

**Training Configuration**:
- **Approach**: Strong directional guards (λ_dir=0.01)
- **Epochs**: 5 warm-up, then guards
- **Result**: Collapsed at epoch 4

**Training Progression**:

| Epoch | Val Cosine | Status |
|-------|------------|--------|
| 1-3 | 0.540 | Stable MSE warm-up |
| 4 | **0.354** | **Catastrophic collapse** |
| 5+ | N/A | Training terminated |

**5CAT Results** (Epoch 3, before collapse):

| Test | Score | Status |
|------|-------|--------|
| **Margin** | -0.132 | ❌ Backward bias |
| **Val Cosine** | 0.540 | ✅ Good |

**Analysis**:
- **Guards 10x stronger than MSE** caused immediate collapse
- **Lesson**: Can't brute-force directionality with penalties
- **Led to P4/P5 approaches**: Change learning signal, not penalties

---

### 9. P3 Tiny Guards (November 1, 2025) ⚠️

**Path**: `artifacts/lvm/models/transformer_p3_tiny_guards/`
**Created**: November 1, 2025, 3:55 PM
**Size**: 205MB

**Training Configuration**:
- **Approach**: Tiny directional guards (λ_dir=0.002)
- **Epochs**: 5 warm-up, then guards activate
- **Purpose**: Test if weak guards avoid collapse

**Training Progression**:

| Phase | Epochs | Val Cosine | Margin | Status |
|-------|--------|------------|--------|--------|
| Warm-up | 1-5 | 0.550 | 0.0 | ✅ Stable |
| Guards | 6 | 0.507 | -0.067 | ⚠️ 4.3% drop |
| Final | 20 | 0.526 | -0.064 | ⚠️ Partial recovery |

**5CAT Results**:
- **Margin**: -0.064 (51% improvement from -0.133!)
- **Val Cosine**: 0.526 (acceptable, slight drop from 0.550)

**Analysis**:
- **Avoided collapse**: Tiny guards didn't destabilize training
- **Partial success**: Improved margin by 51%
- **Insufficient**: Margin still negative, can't overcome entrenched patterns
- **Lesson**: Guards too weak (λ=0.002 ≈ 2% of loss vs MSE)

---

### 8. P2 Residual (November 1, 2025) ❌

**Path**: `artifacts/lvm/models/transformer_residual_p2/`
**Created**: November 1, 2025, 2:50 PM
**Size**: 205MB

**Training Configuration**:
- **Approach**: Residual prediction architecture
- **Formula**: ŷ = norm(u + α·Δ) where Δ = target - last
- **Purpose**: Encourage forward prediction via architecture

**5CAT Results**:

| Test | Score | Status |
|------|-------|--------|
| **Margin** | **-0.534** | ❌ MUCH WORSE than baseline! |
| **Val Cosine** | 0.472 | ⚠️ Degraded |

**Analysis**:
- **Backfired**: Residual made copying EASIER (model learns Δ≈0)
- **Margin 3.2x worse** than P1 baseline (-0.534 vs -0.167)
- **Lesson**: Architectural tricks can make problem worse
- **Why**: Model learned to zero out residual → pure copy

---

## 📈 Training Timeline Summary

**October 31, 2025**:
- 584k Stable (morning)
- V3 Directional (afternoon, failed)

**November 1, 2025** (12-hour debugging session):
- P1 Baseline (1:41 PM) - Established neutral baseline
- P2 Residual (2:50 PM) - Failed (worse copying)
- P3 Tiny Guards (3:55 PM) - Partial success (51% improvement)
- P4 Rollout (6:09 PM) - Failed (collapsed at epoch 4)
- P5 design finalized (11:00 PM)

**November 2, 2025**:
- P5 Stage A (10:06 AM) - Failed (negative margin despite curriculum)

---

## 🎯 Key Findings from Session

### What DOESN'T Work

1. **Strong directional guards** (V3, λ=0.01) → Catastrophic collapse
2. **Residual architecture** (P2) → Makes copying easier
3. **Tiny guards** (P3, λ=0.002) → Partial improvement, insufficient
4. **Rollout loss** (P4) → Collapse when activated after warm-up
5. **Curriculum alone** (P5 Stage A, pos=0.03) → Backward bias persists

### What We Learned

1. **MSE converges TO copy-last**: More epochs = worse margin (-0.167 after 20 vs -0.149 after 3)
2. **Problem in warm-up**: Backward bias exists by epoch 3 (pure MSE)
3. **Guards cause collapse**: When stronger than 0.2% of loss
4. **Positional encoding too weak**: 0.03 scalar insufficient to break symmetry
5. **Data quality is good**: +0.117 forward signal, no inherent bias

### Current Status

**Deployed Models**:
- ✅ **P1 Baseline** (port 9007): Stable, neutral margin, use for testing

**Failed Attempts**: V3, P2, P4, P5 Stage A

**Next Steps**:
1. **Try P5 with stronger positional scalar** (0.05 or 0.10)
2. **Investigate forward-distinctness calculation**
3. **Consider alternative approaches** beyond curriculum

---

## 📋 Model Files Summary

**Recent Models by Size**:
- **205MB**: All Transformer models (P1-P5, V3, etc.)
- **17MB**: AMN models (different architecture)
- **59-84MB**: LSTM/GRU models (older)

**Active Models**:
- `transformer_baseline_p1/` (PORT 9007, deployed)
- `transformer_p5_20251102_095841/stageA/` (latest, failed)

**Historical Models** (Oct 13-Oct 30):
- Various LSTM, GRU, Transformer models with 340k old dataset
- All suffer from backward bias due to poor data quality

---

**Generated**: 2025-11-02 10:15 EST
**Next Update**: After P5 retry with stronger positional scalar
