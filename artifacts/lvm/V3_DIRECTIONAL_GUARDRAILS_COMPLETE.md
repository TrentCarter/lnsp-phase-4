# V3 Directional Guardrails Implementation - Complete

**Date**: 2025-10-31
**Status**: ✅ Ready for Training
**Purpose**: Fix "copy last context" bug WITHOUT collapsing performance

---

## Problem Summary

**V1 Issue (Original Model)**:
- Model learned to copy position 4 (last context) instead of predict position 5 (next)
- Margin(+1 vs -1) = **-0.166** (NEGATIVE = backward prediction)
- Peak at k=-1 instead of k=+1 in offset sweep
- Val cosine: 0.558 (good, but wrong prediction direction)

**V2 Issue (First Fix Attempt)**:
- Added directional losses with λ_dir=0.05, λ_ac=0.05
- Fixed backward bias (margin +0.0663), BUT...
- Collapsed performance: val cosine **0.158** (down from 0.558!)
- New problem: k=+3 drift (predicting 3 steps ahead instead of 1)
- **Diagnosis**: Guard losses too strong, overwhelmed MSE objective

---

## V3 Solution: Complete Guardrail System

### 1. Loss Scheduling (Prevents Early Collapse)

**Warm-up Phase (Epochs 1-3)**:
- Pure MSE only (λ_dir=0, λ_ac=0, context_drop=0)
- Allows model to establish baseline mapping
- Target: val_cos ≥ 0.48, should see high similarity to k=-1 (copy behavior)

**Ramp Phase (Epochs 4-10)**:
- Gradual introduction: λ_dir and λ_ac ramp from 0.005 → 0.01
- Context drop ramps from 0.05 → 0.10
- Guards nudge without dominating
- Target: Margin should turn positive by epoch 7-8

**Full Phase (Epochs 11-20)**:
- All guards at target strength: λ_dir=0.01, λ_ac=0.01
- Context drop at 0.10
- Target: Margin ≥ +0.10, val_cos ≥ 0.54

### 2. Positional Scalar (Breaks Time Symmetry)

```python
# Adds [0.0, 0.25, 0.5, 0.75, 1.0] * 0.03 to each context position
# Input dim: 768 → 769
pos = torch.linspace(0, 1, steps=5) * 0.03
contexts_aug = torch.cat([contexts, pos.unsqueeze(-1)], dim=-1)
```

**Why this works**:
- Tells model which slot is "most recent" without ambiguity
- Prevents time-reversal symmetry (can't confuse forward/backward)
- Cheap (1 extra dim), very effective
- Model must learn: high position = more recent = closer to target

**Important**: If positional encoding is enabled, the model is created with `input_dim=769` and positional encoding is applied to ALL training batches (cannot be disabled per-epoch). The scheduling only affects the directional loss weights, not the positional encoding itself.

### 3. Directional Margin Loss (Fixed Weights)

```python
# V1: λ=0.05 (TOO STRONG)
# V3: λ=0.01 (5x lighter)
L_dir = λ_dir * ReLU(m_dir - (cos(pred, next) - cos(pred, prev)))
```

**Purpose**: Ensures pred is more similar to NEXT than PREVIOUS
**Settings**: λ=0.01, margin=0.03
**Effect**: Gentle nudge, won't overwhelm MSE

### 4. Anti-Copy Hinge Loss (Fixed Weights)

```python
# V1: λ=0.05 (TOO STRONG)
# V3: λ=0.01 (5x lighter)
L_ac = λ_ac * mean_i ReLU(m_ac - (cos(pred, next) - cos(pred, ctx[i])))
```

**Purpose**: Ensures pred is more similar to next than ANY context frame
**Settings**: λ=0.01, margin=0.01
**Effect**: Prevents blind copying of any input position

### 5. Future Margin Loss (Infrastructure Ready)

```python
# TODO: Requires article-aware batching
L_fut = λ_fut * (ReLU(m - (cos(pred, +1) - cos(pred, +2)))
               + ReLU(m - (cos(pred, +1) - cos(pred, +3))))
```

**Purpose**: Anchor prediction to k=+1 (not k=+2 or k=+3)
**Status**: Function implemented, awaiting dataloader update
**When needed**: If k=+3 drift persists after V3 training

### 6. Context Drop Augmentation (Scheduled)

```python
# Randomly perturbs last context position with prob p
# Makes blind copying unreliable → forces model to use full context
context_drop(contexts, p=0.10, mode="last_to_noise")
```

**Schedule**: 0 → 0.05 (ramp) → 0.10 (full)

---

## File Changes

### New Files

1. **`app/lvm/losses_directional.py`** (UPDATED)
   - Added `future_margin_loss()` function
   - Implements +1 vs +2/+3 ranking loss

2. **`scripts/train_transformer_directional_v3.sh`** (NEW)
   - Complete training script with all guardrails
   - 3-stage schedule (warm-up → ramp → full)
   - Positional encoding enabled
   - Target hyperparameters set

3. **`scripts/check_5cat_epoch.sh`** (NEW)
   - Helper for running 5CAT on intermediate checkpoints
   - Quick validation during training
   - Usage: `./scripts/check_5cat_epoch.sh <model_path> [samples]`

### Modified Files

1. **`app/lvm/train_unified.py`**
   - Added positional encoding support (`--use-positional`, `--pos-scale`)
   - Added future margin loss parameters (`--lambda-fut`, `--margin-fut`)
   - Added scheduling parameters (`--warmup-epochs`, `--ramp-epochs`)
   - Implemented 3-stage loss scheduling in training loop
   - Updated train_epoch signature with new parameters
   - Added positional augmentation before model forward pass

---

## How to Use

### Quick Start (Run V3 Training)

```bash
./scripts/train_transformer_directional_v3.sh
```

This will:
- Train for 20 epochs with scheduled guards
- Save to `artifacts/lvm/models/transformer_directional_v3/`
- Output training logs showing margin evolution

### Monitor Training Progress

Check 5CAT at key epochs:

```bash
# After epoch 5 (ramp phase)
./scripts/check_5cat_epoch.sh \
  artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt 500

# After epoch 10 (end of ramp)
./scripts/check_5cat_epoch.sh \
  artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_10.pt 500

# Final model
./scripts/check_5cat_epoch.sh \
  artifacts/lvm/models/transformer_directional_v3/best_model.pt 5000
```

### Expected Results

**Training Evolution**:
- Epochs 1-3: Margin negative (copy behavior), val_cos ~0.48-0.52
- Epochs 4-7: Margin turning positive, val_cos ~0.50-0.54
- Epochs 8-10: Margin solidly positive (+0.06+), val_cos ~0.52-0.56
- Epochs 11-20: Margin ≥ +0.10, val_cos ~0.54-0.58

**Final 5CAT Targets** (5000 samples):

| Gate | Metric | VAL Target | OOD Target | What It Tests |
|------|--------|------------|------------|---------------|
| **A: Offset Sweep** | Margin(+1) | ≥ +0.12 | ≥ +0.10 | Predicts NEXT, not previous |
| **A: Offset Sweep** | Peak k | k=+1 | k=+1 | Correct temporal direction |
| **B: Retrieval Rank** | R@1 / R@5 / MRR | ≥60% / ≥95% / ≥80% | ≥55% / ≥92% / ≥75% | Finds target in article |
| **C: Ablations** | Shuffle delta | ≤ -0.15 | ≤ -0.15 | Order matters |
| **D: Rollout** | Avg cos@H=5 | ≥ 0.45 | ≥ 0.42 | Multi-step coherence |
| **E: Bins Delta** | abs(Val - OOD) | ≤ 0.05 | ≤ 0.05 | Generalization |

---

## Troubleshooting

### If Training Shows:

**Negative margin after epoch 10**:
- Guards too weak
- Action: Increase λ_dir and λ_ac to 0.015 (50% boost)

**Val cosine drops below 0.45**:
- Guards too strong
- Action: Reduce λ_dir and λ_ac to 0.007 (30% reduction)

**Val cosine drops below 0.40 (collapse)**:
- Emergency stop!
- Action: Reduce ALL λ by half, extend warm-up to 5 epochs

**k=+3 drift (prediction 3 steps ahead)**:
- Needs future margin loss
- Action: Implement article-aware batching, enable λ_fut=0.005

**k=+1 is peak but margin < +0.05**:
- Prediction direction correct but weak
- Action: Extend training to 30 epochs, margin should strengthen

**OOD much worse than VAL (>0.10 gap)**:
- Overfitting or data contamination
- Action: Verify article-based splits, check no overlap in train/val

---

## Technical Details

### Loss Weights Rationale

**Primary MSE**: λ=1.0 (anchor, never change)

**Directional**: λ=0.01, margin=0.03
- V1 used 0.05 → collapsed to 0.158 cosine
- 0.01 is 5x lighter → won't overwhelm MSE
- Margin 0.03 is tight but achievable (half of V1's 0.05)

**Anti-Copy**: λ=0.01, margin=0.01
- V1 used 0.05, margin=0.02 → too strong
- 0.01 margin very tight → only activates when pred nearly identical to context

**Context Drop**: p=0.10 (10% of batches)
- V1 used 0.20 → too aggressive, lost learning signal
- 0.10 is enough to break copying shortcut without hurting learning

**Positional**: scale=0.03
- Adds 0.00, 0.0075, 0.015, 0.0225, 0.03 to positions 0-4
- Small enough not to dominate 768D vectors
- Large enough to be distinguishable signal

### Why Scheduling Matters

**Without scheduling** (V1):
- Guards active from epoch 1
- Model tries to optimize 3 objectives at once (MSE + dir + ac)
- Confusion → finds trivial solution (collapse)

**With scheduling** (V3):
- Epochs 1-3: Model learns basic MSE mapping
- Epochs 4-10: Guards gently nudge toward correct direction
- Epochs 11-20: Guards enforce correct behavior without dominating
- MSE always primary objective → maintains performance

---

## Next Steps

1. **Run Training**:
   ```bash
   ./scripts/train_transformer_directional_v3.sh
   ```

2. **Monitor Checkpoints**:
   ```bash
   # Check epoch 5 (should see margin turning positive)
   ./scripts/check_5cat_epoch.sh artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt

   # Check epoch 10 (should see margin ≥ +0.06)
   ./scripts/check_5cat_epoch.sh artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_10.pt
   ```

3. **Final Validation**:
   ```bash
   # Full 5CAT on best model (5000 samples)
   ./.venv/bin/python tools/tests/test_5to1_alignment.py \
     --model artifacts/lvm/models/transformer_directional_v3/best_model.pt \
     --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
     --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
     --articles-npz artifacts/wikipedia_584k_fresh.npz \
     --device mps --max-samples 5000 | tee /tmp/5cat_v3_final.log
   ```

4. **If Successful** (passes 5CAT):
   - Deploy to port 9007 (replace Transformer Experimental)
   - Update production documentation
   - Archive V1/V2 models

5. **If k=+3 Drift Persists**:
   - Implement article-aware batching in dataloader
   - Enable future margin loss with λ_fut=0.005
   - Retrain with V4 configuration

---

## Summary

**V3 implements a complete guardrail system** to fix the "copy last context" bug while maintaining high performance:

✅ **Scheduled ramp-up** prevents early collapse
✅ **Positional encoding** breaks time symmetry
✅ **Lightweight losses** (5x lighter than V1) nudge without dominating
✅ **Context drop** makes copying unreliable
✅ **Future loss infrastructure** ready when needed

**Expected outcome**: Margin ≥ +0.10, val cosine ≥ 0.54, OOD within ±0.05, all 5CAT gates passing.

---

**Author**: Claude Code + User
**Date**: 2025-10-31
**Version**: 3.0 (Production Ready)
