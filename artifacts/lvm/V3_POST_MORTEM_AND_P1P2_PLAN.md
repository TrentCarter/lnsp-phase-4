# V3 Post-Mortem and P1/P2 Recovery Plan

**Date**: 2025-11-01
**Status**: P0 Triage Complete → P1/P2 Ready

---

## Executive Summary

V3 directional guardrails **FAILED** due to overly strong loss weights causing immediate collapse at epoch 4. Training logs show catastrophic performance drop the instant directional losses activated (val_cos 0.540 → 0.354). The model learned backward prediction bias instead of fixing it (5CAT margin: -0.132).

**Root Cause**: λ_dir=0.01 and λ_ac=0.01 are too strong, even with scheduled ramp-up.

**Recovery Plan**: P1 (MSE-only baseline) → P2 (residual architecture + tiny late guards λ≤0.002)

---

## P0 Triage Results

### P0.1: Training Data Quality ✅

```bash
./.venv/bin/python tools/sequence_direction_audit.py verify \
  --sequences artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --articles artifacts/wikipedia_584k_fresh.npz --max-samples 50000
```

**Results**:
- Mean sequence coherence: **0.468** ✅ (good)
- Monotonicity errors: **0** ✅
- Repeat pad count: **0** ✅

**Verdict**: Training data is fine. Issue is not in the data.

### P0.2: Positional Encoding Test ❌ (Not Viable)

Cannot test without positional encoding - architecture is baked in (input_proj expects 769D). Would need full retrain to test.

### P0.3: V3 Training Log Analysis 🚨 **SMOKING GUN**

**Epochs 1-3 (Warm-up, MSE-only)**:
```
Epoch 1: val_cos 0.459, train_cos 0.371
Epoch 2: val_cos 0.523, train_cos 0.511
Epoch 3: val_cos 0.540, train_cos 0.546 ← BEST PERFORMANCE
```

**Epoch 4 (Guards activated at λ=0.01)** - CATASTROPHIC COLLAPSE:
```
Epoch 4: val_cos 0.354 (-0.186 drop!), train_cos 0.356
         ↑
         The instant directional losses turned on!
```

**Epochs 5-20**: Never recovered, final val_cos 0.362

**Key Findings**:
1. Model was learning **perfectly** during warm-up (reaching 0.540 by epoch 3)
2. **Instant collapse** when guards activated at epoch 4
3. Guards at λ=0.01 are **way too strong** - destroyed learning immediately
4. Model couldn't optimize both MSE and directional constraints → collapsed

---

## 5CAT Results (V3 Model)

**Gates Passed**: 2/5 (Rollout, Bins Delta)
**Gates Failed**: 3/5 ❌

| Gate | Metric | VAL | OOD | Threshold | Status |
|------|--------|-----|-----|-----------|--------|
| **A: Offset Sweep** | Margin(+1) | **-0.132** | **-0.136** | ≥+0.12 | ❌ FAIL |
| **A: Peak** | k=-1 | **0.655** | **0.663** | k=+1 should be peak | ❌ BACKWARD |
| **B: Retrieval** | R@5 | 20.4% | 16.6% | ≥95% | ❌ FAIL |
| **C: Ablations** | Shuffle delta | -0.025 | -0.019 | ≤-0.15 | ❌ FAIL |
| **D: Rollout** | Avg cos@H=5 | 0.478 | 0.488 | ≥0.45 | ✅ PASS |
| **E: Bins Delta** | |VAL-OOD| | 0.031 | 0.017 | ≤0.05 | ✅ PASS |

**Critical Issue**: Model still predicts backward (k=-1 peak), despite all V3 guardrails!

---

## Root Cause Analysis

### What We Thought Would Happen
- Scheduled ramp-up prevents early collapse ✅
- Positional encoding breaks time symmetry ✅ (technically worked)
- Light losses (λ=0.01) nudge toward forward prediction ❌ **TOO STRONG**

### What Actually Happened
1. **Warm-up (epochs 1-3)**: Model learned basic MSE prediction, reached val_cos 0.540
2. **Guard activation (epoch 4)**:
   - Directional margin loss enforces cos(pred, next) > cos(pred, prev) + 0.03
   - Anti-copy hinge enforces cos(pred, next) > cos(pred, ctx) + 0.01
   - Combined with MSE loss, creates conflicting optimization pressures
3. **Collapse**: Model can't satisfy both MSE and guards → finds degenerate solution with low cosine to everything
4. **Never recovers**: Stuck in collapsed state for remaining 16 epochs

### Why Guards Were Too Strong

**Loss magnitude comparison** (estimated from training history):
- MSE loss at epoch 3: ~0.00119 (scale: ~10^-3)
- Directional loss at λ=0.01: Margin violations ~0.03-0.10 (scale: ~10^-2)
- **Guard losses are 10x larger than MSE!**

Even at λ=0.01, the guard gradients dominated and destroyed the MSE learning.

---

## P1: Clean MSE-Only Baseline (Next Step)

**Purpose**: Verify the pipeline can still learn without guards (sanity check).

**Script**: `./scripts/train_transformer_baseline_p1.sh`

**Configuration**:
- Model: Transformer (same architecture as V3)
- Epochs: 5 (just need to confirm learning works)
- Loss: MSE only (λ_dir=λ_ac=λ_fut=0)
- NO positional encoding (input_dim=768)
- NO context drop
- NO scheduling (pure baseline)

**Success Criteria** (by epoch 3):
- ✅ val_cos ≥ 0.50
- ✅ train_cos ≥ 0.48
- ✅ No collapse
- ✅ Peak at k=+1 (even if margin small)
- ✅ R@5 ≥ 90% on VAL

**Run Time**: ~10-15 minutes (5 epochs on MPS)

**If P1 Fails**: Pipeline is broken, need deeper debugging.
**If P1 Passes**: Proceed to P2 (residual architecture).

---

## P2: Residual Prediction Architecture (Surgical Fix)

### Core Idea: Predict Δ from Last Frame

**Problem**: Model can copy ctx[-1] with zero effort (identity shortcut).
**Solution**: Force model to predict **delta/residual** instead of absolute next vector.

### Architecture Changes

```python
# OLD (V3 and before):
y_pred = model(ctx)  # Direct prediction: (B, 5, 768) → (B, 768)
loss = MSE(y_pred, y_next)

# NEW (P2):
u = ctx[:, -1, :]           # Last frame (B, 768)
delta = model(ctx)          # Model outputs delta (B, 768)
alpha = nn.Parameter(0.5)   # Learnable scale (small init)
y_pred = F.normalize(u + alpha * delta, dim=-1)  # Compose
loss = MSE(y_pred, y_next)  # Same MSE target!
```

**Why This Helps**:
1. **Breaks identity copying**: If model outputs delta≈0 → y_pred≈u → MSE stays high unless y_next≈u
2. **Must learn forward "velocity"**: Non-zero delta required to improve MSE
3. **No architectural complexity**: Just one learnable scalar α and residual connection
4. **MSE-friendly**: Primary loss is still MSE (no competing objectives)

### Tiny Late Guards (Ultra-Conservative)

**Future Margin Loss** (start epoch ≥6):
```python
L_fut = ReLU(0.01 - (cos(y_pred, y_next) - cos(y_pred, y_p2)))
      + ReLU(0.01 - (cos(y_pred, y_next) - cos(y_pred, y_p3)))

λ_fut = 0.002  # 5x lighter than V3!
```

**Directional Margin Loss** (start epoch ≥6):
```python
L_dir = ReLU(0.01 - (cos(y_pred, y_next) - cos(y_pred, y_prev)))

λ_dir = 0.002  # 5x lighter than V3!
```

**Anti-Copy Loss** (optional, only if drift persists after epoch 10):
```python
L_ac = mean_i( ReLU(0.01 - (cos(y_pred, y_next) - cos(y_pred, ctx[i]))) )

λ_ac = 0.002  # Start disabled, enable only if needed
```

**Residual Regularization** (always on):
```python
L_res = (delta**2).mean() * 1e-4  # Prevent exploding residuals
```

### Micro-Scheduler

| Epochs | λ_dir | λ_ac | λ_fut | λ_res | Notes |
|--------|-------|------|-------|-------|-------|
| 1-5 | 0 | 0 | 0 | 1e-4 | MSE-only residual learning |
| 6-10 | 0.002 | 0 | 0.002 | 1e-4 | Add tiny future+directional |
| 11+ | 0.002 | 0* | 0.003 | 1e-4 | Optionally enable L_ac if drift |

\* Only enable λ_ac if 5CAT shows margin ≤0 after epoch 10

### Mini-5CAT Every Epoch (Early Warning System)

```python
# At end of each epoch, run on 500 VAL samples:
mini_5cat = {
    'peak_offset': argmax_k(cos(y_pred, y_{k})),  # Should be k=+1
    'margin': cos(y_pred, y_+1) - cos(y_pred, y_-1),
    'val_cos': mean(cos(y_pred, y_next))
}

# Abort/backoff rules:
if epoch >= 3 and mini_5cat['peak_offset'] != +1 and mini_5cat['margin'] <= 0:
    print("⚠️  Backward drift detected, dropping guards to zero next epoch")
    λ_dir, λ_ac, λ_fut = 0, 0, 0  # Emergency backoff
```

### P2 Success Criteria (Epoch-by-Epoch)

| Epoch | val_cos | Margin | Peak | R@5 | Status |
|-------|---------|--------|------|-----|--------|
| 1 | ≥0.48 | any | any | any | Baseline learning |
| 3 | ≥0.50 | ≥+0.06 | k=+1 | ≥88% | Residual working |
| 6 | ≥0.52 | ≥+0.08 | k=+1 | ≥92% | Guards on, no collapse |
| Final | ≥0.54 | ≥+0.10 | k=+1 | ≥95% | Production ready |

**Final 5CAT** (full 5000 samples):
- ✅ 4/5 gates pass (minimum)
- ✅ Margin ≥ +0.10 (VAL and OOD)
- ✅ |VAL - OOD| ≤ 0.05 (generalization)

---

## Implementation Files

### P1 Files
- `scripts/train_transformer_baseline_p1.sh` - Clean MSE-only baseline script ✅ CREATED

### P2 Files (To Create)
- `app/lvm/models_residual.py` - New model class with residual prediction
- `app/lvm/train_residual.py` - Training script with mini-5CAT integration
- `scripts/train_transformer_residual_p2.sh` - Main P2 training script
- `tools/mini_5cat.py` - Fast 500-sample 5CAT for per-epoch validation

---

## Key Lessons from V3 Failure

1. **Loss weight scale matters**: λ=0.01 seems small but was 10x larger than MSE
2. **Scheduling alone doesn't prevent collapse**: Need lighter weights too
3. **Monitor DURING training**: Should have caught epoch 4 collapse immediately
4. **Conflicting objectives need careful balancing**: MSE + guards created degenerate solution
5. **Residual architecture is better than loss-only fixes**: Change architecture to prevent copying

---

## Next Actions

**Immediate**:
```bash
# 1. Run P1 baseline (10-15 minutes)
./scripts/train_transformer_baseline_p1.sh

# 2. Check results
cat artifacts/lvm/models/transformer_baseline_p1/training_history.json | \
  jq '.history[] | {epoch, val_cosine, train_cosine}'

# Expected by epoch 3: val_cos ≥ 0.50
```

**If P1 Passes**:
```bash
# 3. Implement P2 residual architecture
# (Create models_residual.py with residual prediction)

# 4. Run P2 training with tiny late guards
./scripts/train_transformer_residual_p2.sh

# 5. Monitor mini-5CAT every epoch for early warnings
```

**If P1 Fails**:
- Investigate data loading (verify 584k clean splits)
- Check train_unified.py for regressions
- Verify FAISS/NPZ file integrity

---

## Timeline Estimate

- **P1 Baseline**: 10-15 minutes (5 epochs)
- **P2 Implementation**: 1-2 hours (code + testing)
- **P2 Training**: 40-60 minutes (20 epochs with mini-5CAT)
- **Final 5CAT**: 10 minutes (5000 samples)

**Total**: ~2-3 hours from start to deployed model

---

**Status**: Ready to execute P1
**Author**: Claude Code + User
**Date**: 2025-11-01
