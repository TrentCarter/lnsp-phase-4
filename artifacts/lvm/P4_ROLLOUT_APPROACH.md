# P4: Rollout Loss + Adaptive Guards

**Status**: ✅ READY TO LAUNCH

## Executive Summary

P3 proved that tiny guards alone can't overcome an entrenched copy-last pattern once MSE has locked it in during warm-up. **P4 changes the game**: instead of fighting symptoms with stronger guards, we **change the learning signal** so that copying becomes fundamentally bad over multiple timesteps.

## The Problem with P1-P3

### What Went Wrong

1. **P1 Baseline**: Clean MSE → model learns copy-last is "good enough" (val_cos 0.550, margin 0.0)
2. **V3**: Strong guards (λ=0.01) → catastrophic collapse (val_cos 0.540 → 0.354)
3. **P2 Residual**: Architectural change → made copying EASIER (margin -0.534)
4. **P3 Tiny Guards**: Gentle nudges (λ=0.002) → margin improved but still negative (-0.064)

### Root Cause

**MSE is indifferent to k=-1 vs k=+1** when both are close. During warm-up:
- Copying ctx[-1] gives low MSE on high-similarity Wikipedia sequences
- By epoch 5, this pattern is deeply embedded in weight space
- Tiny guards can't overcome 10-50x stronger MSE gradient

## P4 Solution: Multi-Step Rollout Loss

### Core Insight

**Don't fight the entrenched pattern. Change the objective landscape.**

If the model is predicting k=-1 (copying last frame), it will fail catastrophically over 2-3 steps:
- Step 1: Copy ctx[-1] → okay (if next ≈ last)
- Step 2: Copy predicted step 1 → flat trajectory (no forward momentum)
- Step 3: Copy predicted step 2 → completely flat (high penalty)

Rollout loss **makes copying globally bad**, while guards **decide the direction** (+1 vs -1).

### Implementation

**Autoregressive Rollout** (H=3 steps):

```python
# Step 1: Predict ŷ₁ from ctx
y_pred_1 = model([ctx[0], ctx[1], ctx[2], ctx[3], ctx[4]])

# Step 2: Predict ŷ₂ from shifted context
y_pred_2 = model([ctx[1], ctx[2], ctx[3], ctx[4], ŷ₁])

# Step 3: Predict ŷ₃ from shifted context
y_pred_3 = model([ctx[2], ctx[3], ctx[4], ŷ₁, ŷ₂])

# Rollout loss: Penalize flat trajectories
# - Step 1: MSE against true target
# - Steps 2-3: Penalize high similarity (cos > 0.95 = flat trajectory)
L_roll = (MSE(ŷ₁, target) + flat_penalty(ŷ₁, ŷ₂) + flat_penalty(ŷ₂, ŷ₃)) / 3
```

**Why This Works**:
- Pure copying creates flat trajectories → high rollout penalty
- Model must learn forward dynamics over multiple steps
- Guards then guide direction (next vs prev)

## Training Curriculum

### Phase 1: Warm-Up (Epochs 1-3)

**Pure MSE** - Let model learn basic representations

```python
loss = MSE(pred, target)
```

**Expected**:
- Epoch 3: val_cos ≥ 0.50
- Model learns general patterns, no directional bias yet

### Phase 2: Rollout Activation (Epochs 4-6)

**Add multi-step consistency**

```python
loss = MSE(pred, target) + 0.05 * L_roll
```

**Why epoch 4?**
- Basic representations learned (epoch 3 val_cos ~0.50)
- Early enough to prevent copy-last from cementing
- Rollout changes objective landscape before bad habits form

**Expected**:
- Epoch 4-5: Small dip (≤2-3%) is normal as model adjusts
- Epoch 6: Margin should start moving toward zero

### Phase 3: Guards + Stronger Rollout (Epochs 7+)

**Full curriculum active**

```python
# Epoch 7+: Increase rollout weight
loss = MSE(pred, target) + 0.10 * L_roll + guards

# Guards (adaptive):
if cos(ctx[-1], target) > 0.60:  # High similarity = tempting to copy
    lambda_dir_eff = 0.002 * (1.0 + boost)  # Boost guard
else:
    lambda_dir_eff = 0.002  # Normal guard

loss += lambda_dir_eff * L_dir(pred, target, prev)

# Epoch 10+: Add future ranking
loss += 0.002 * L_fut(pred, target, {+2, +3})
```

**Expected**:
- Epoch 8-10: Margin crosses positive (≥ +0.04)
- Epoch 20: val_cos 0.54-0.56, margin ≥ +0.10

## Adaptive Directional Guards

### Problem

Not all samples need equal guard strength. High-similarity cases (cos(ctx[-1], target) > 0.60) are where copying is most tempting.

### Solution

**Adaptive lambda_dir** based on sample similarity:

```python
sim = cos(ctx[-1], target)  # Per-sample similarity
boost = sigmoid((sim - 0.60) / 0.05)  # Sharp boost above 0.60
lambda_dir_eff = lambda_dir * (1.0 + boost.mean())
```

**Effect**:
- Low similarity (< 0.60): Normal guard (λ ≈ 0.002)
- High similarity (> 0.70): Strong guard (λ ≈ 0.004)
- Focuses guard effort where it matters most

## Configuration

### P4 Hyperparameters

```bash
# Rollout settings
--rollout-h 3                  # Predict 3 steps ahead
--lambda-roll 0.05             # Start at 0.05, increases to 0.10 at epoch 7
--rollout-start-epoch 4        # Activate after basic learning

# Guards (adaptive)
--guards-start-epoch 6         # After rollout establishes landscape
--lambda-dir 0.002             # Tiny base guard
--lambda-fut 0.002             # Future ranking (epoch 10+)
--lambda-ac 0.0                # Anti-copy disabled (most destabilizing)
--adaptive-dir                 # Enable adaptive weighting

# Regularization
--context-drop-p 0.05          # Prevent overfitting to exact last frame
--margin-dir 0.01              # Directional margin threshold
--margin-fut 0.008             # Future margin threshold
```

### Why These Values?

**λ_roll = 0.05 → 0.10**:
- 0.05 ≈ 5% of total loss (gentle introduction)
- 0.10 ≈ 10% of total loss (strong multi-step signal)
- 2x ramp at epoch 7 (after model adapts to rollout)

**λ_dir = 0.002** (adaptive):
- 5x weaker than V3's failed λ=0.01
- Adaptive boost to 0.004 on high-similarity samples
- Guards decide direction, rollout handles magnitude

**H = 3 steps**:
- Long enough to expose copying (flat trajectory)
- Short enough to be computationally efficient
- Can increase to H=4 if margin stalls at epoch 10

**Epoch scheduling**:
- 1-3: Pure MSE (basic learning)
- 4-6: Rollout only (change landscape)
- 7+: Rollout + guards (full curriculum)
- 10+: Add future ranking (if needed)

## Expected Results

### By Epoch Checkpoints

| Epoch | Expected val_cos | Expected margin | What's Happening |
|-------|------------------|-----------------|------------------|
| 3 | ≥ 0.50 | ~0.0 | Warm-up complete |
| 4 | 0.48-0.50 | -0.05 to 0.0 | Rollout adjusting |
| 6 | 0.50-0.52 | -0.02 to +0.02 | Guards activate |
| 8 | 0.52-0.54 | +0.04 to +0.08 | Margin positive! |
| 10 | 0.53-0.55 | +0.06 to +0.10 | Stable forward |
| 20 | 0.54-0.56 | ≥ +0.10 | Target achieved |

### 5CAT Gates (Final)

| Gate | Metric | VAL Target | OOD Target | P4 Expected |
|------|--------|------------|------------|-------------|
| **A: Offset Sweep** | Margin(+1) | ≥ +0.12 | ≥ +0.10 | +0.10 to +0.15 |
| **B: Retrieval** | R@5 / MRR | ≥95% / ≥80% | ≥92% / ≥75% | Pass |
| **C: Ablations** | Shuffle delta | ≤ -0.15 | ≤ -0.15 | Pass |
| **D: Rollout** | Cos@H=5 | ≥ 0.45 | ≥ 0.42 | 0.50-0.55 |
| **E: Generalization** | abs(Val-OOD) | ≤ 0.05 | ≤ 0.05 | Pass |

**Expected**: Pass 4/5 or 5/5 gates

## Monitoring & Tripwires

### Watch For (Early Warning)

**Epoch 4-5** (rollout activation):
- ✅ **Normal**: val_cos dips 2-3% (0.50 → 0.48), recovers by epoch 6
- ⚠️ **Tripwire**: val_cos drops > 5% → reduce lambda_roll to 0.03

**Epoch 6** (guards activate):
- ✅ **Normal**: Smooth transition, no sudden drops
- ⚠️ **Tripwire**: val_cos drops > 3% → guards too strong, reduce to λ=0.001

**Epoch 8-10**:
- ✅ **Target**: Margin crosses positive (≥ +0.04)
- ⚠️ **Tripwire**: Margin still < 0 → increase rollout_h to 4, keep λ same

**Epoch 10-20**:
- ✅ **Target**: Steady improvement, margin ≥ +0.08
- ⚠️ **Tripwire**: Margin stalls < +0.06 → add future loss (already scheduled)

### Emergency Rollback

If val_cos < 0.48 after epoch 8:
1. Stop training
2. Load epoch 3 checkpoint (pre-rollout)
3. Restart with λ_roll=0.03 (weaker)

## Comparison to Previous Approaches

| Model | Approach | Final val_cos | Final margin | Outcome |
|-------|----------|---------------|--------------|---------|
| **P1** | Pure MSE | 0.550 | 0.0 | ✅ Baseline (neutral) |
| **V3** | Strong guards (λ=0.01) | 0.354 | -0.132 | ❌ Collapse |
| **P2** | Residual arch | 0.472 | -0.534 | ❌ Worse copying |
| **P3** | Tiny guards (λ=0.002) | 0.526 | -0.064 | ⚠️ Stable but biased |
| **P4** | Rollout + adaptive | **0.54-0.56** | **+0.10** | 🎯 **Target** |

**Key Advantages**:
1. **Changes learning signal**, not just adds penalties
2. **Curriculum-based**: rollout first, then guards
3. **Adaptive guards**: focuses effort where needed
4. **Multi-step objective**: makes copying fail globally

## Launch Command

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./scripts/train_transformer_p4_rollout.sh
```

**Estimated time**: ~4-5 hours (20 epochs @ 12-15 min/epoch on MPS)

**Logs**: `artifacts/lvm/models/transformer_p4_rollout/training.log`

## Next Steps After Training

### 1. Check Training Progress

```bash
# View training history
cat artifacts/lvm/models/transformer_p4_rollout/training_history.json | \
  jq '.history[] | {epoch, val_cosine, train_cosine}'

# Check margin progression
grep "Margin(+1 vs last)" artifacts/lvm/models/transformer_p4_rollout/training.log | tail -20
```

### 2. Run Full 5CAT Validation

```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_p4_rollout/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

### 3. If P4 Succeeds

- Deploy to production (port 9007?)
- Document final metrics
- Update training procedures
- Celebrate! 🎉

### 4. If P4 Fails (margin still < 0 at epoch 20)

**Diagnostic**:
- Check rollout loss values (should be decreasing)
- Check if margin improved at all (any upward trend?)
- Check epoch 4-6 transition (did rollout work?)

**Next steps**:
- Increase rollout horizon: H=4 or H=5
- Stronger rollout weight: λ_roll=0.10 → 0.15
- Earlier activation: rollout_start_epoch=3
- Consider data quality audit (are sequences truly sequential?)

## Technical Implementation Notes

### Rollout Loss Computation

```python
# Autoregressive rollout with teacher forcing
current_ctx = contexts.clone()  # (B, 5, 768)
rollout_losses = []

for step in range(H):
    # Predict next vector
    y_pred = model(current_ctx)

    if step == 0:
        # First step: MSE against true target
        loss_step = MSE(y_pred, target)
    else:
        # Later steps: Penalize flat trajectory
        prev_pred = rollout_losses[-1]
        trajectory_sim = cos(prev_pred, y_pred)
        # If similarity > 0.95, trajectory too flat
        loss_step = relu(trajectory_sim - 0.95) * 10.0

    rollout_losses.append(loss_step)

    # Teacher forcing: shift context
    current_ctx = cat([current_ctx[:, 1:], y_pred.unsqueeze(1)], dim=1)

L_roll = mean(rollout_losses)
```

### Adaptive Lambda Computation

```python
# Compute per-batch similarity
last_ctx = normalize(contexts[:, -1, :])  # Last frame
target_norm = normalize(target)
sim = (last_ctx * target_norm).sum(dim=-1)  # (B,)

# Sigmoid boost: sharp transition at 0.60
boost = sigmoid((sim - 0.60) / 0.05)  # (B,)

# Apply to lambda_dir
lambda_dir_eff = lambda_dir * (1.0 + boost.mean())
```

### Curriculum Schedule

```python
if epoch < 4:
    # Phase 1: Warm-up
    lambda_roll = 0.0
    lambda_dir = 0.0
elif epoch < 6:
    # Phase 2: Rollout only
    lambda_roll = 0.05
    lambda_dir = 0.0
elif epoch < 7:
    # Phase 3: Rollout + guards (weak)
    lambda_roll = 0.05
    lambda_dir = 0.002
else:
    # Phase 4: Rollout + guards (strong)
    lambda_roll = 0.10
    lambda_dir = 0.002
    lambda_fut = 0.002 if epoch >= 10 else 0.0
```

## Files Modified

1. `app/lvm/train_unified.py`:
   - Added rollout loss implementation (lines 245-295)
   - Added adaptive lambda_dir (lines 310-318)
   - Added curriculum scheduling (lines 703-738)
   - Added new arguments (lines 459-467)

2. `scripts/train_transformer_p4_rollout.sh`:
   - Complete training script with P4 configuration

3. `artifacts/lvm/P4_ROLLOUT_APPROACH.md`:
   - This document (comprehensive guide)

## Success Criteria

### Minimum Acceptable

- ✅ Final val_cos ≥ 0.52
- ✅ Final margin ≥ +0.05
- ✅ Pass 3/5 5CAT gates
- ✅ No collapse during training

### Target

- 🎯 Final val_cos ≥ 0.54
- 🎯 Final margin ≥ +0.10
- 🎯 Pass 4/5 or 5/5 5CAT gates
- 🎯 OOD within ±0.03 of VAL

### Stretch

- 🚀 Final val_cos ≥ 0.56
- 🚀 Final margin ≥ +0.15
- 🚀 Pass all 5 5CAT gates
- 🚀 Rollout cos@H=5 ≥ 0.55

## Why P4 Should Work

1. **Addresses Root Cause**: Changes learning signal, not just adds penalties
2. **Curriculum-Based**: Gives model time to adapt at each phase
3. **Multi-Step Objective**: Makes copying fail over 2-3 steps (not just 1)
4. **Adaptive Guards**: Focuses effort on high-similarity cases
5. **Proven Components**: All pieces work individually (MSE, guards, rollout)
6. **Safety Mechanisms**: Early tripwires, gradual ramp-up, rollback plan

**Bottom Line**: P3 showed tiny guards can't fight entrenched MSE patterns. P4 doesn't fight - it **changes the game** so copying becomes fundamentally incompatible with the objective.

Let's see if the optimizer agrees! 🚀
