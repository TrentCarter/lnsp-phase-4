# P6b v2.2 Implementation Summary (Nov 2, 2025)

## 🎯 Objective

**Flip margin positive** while maintaining v2.1's stability by using controlled stronger directional pressure.

---

## 📊 v2.1 Results (Why We Need v2.2)

**P6b v2.1 Final Results** (12 epochs):
- Margin: -0.047 (improved 43% from -0.082, but still negative ❌)
- R@5: 77% (excellent ✅)
- Val cosine: 0.488 (healthy ✅)
- ρ: 0.10-0.25 (stable ✅)
- **Passed 2/5 5CAT gates** (Rollout, Bins Delta)

**Root Cause**: Guardrails too conservative
- ρ capped at 25% → directional loss too weak vs data's -7% backward bias
- Margin improved significantly but not enough to flip positive
- All 6 guardrails worked perfectly (no collapse)

**Verdict**: Stability proven ✅, but need ~50% more directional pressure

---

## 🚀 P6b v2.2 Changes (Surgical Escalation)

### 1. ρ-Controller (NEW)

**v2.1** (passive cap):
```python
if rho > 0.25:  # Just cap it
    lambda_eff *= (0.25 / rho)
```

**v2.2** (active control):
```python
# Target-based control (not just capping)
rho_target = 0.35  # GOAL: 35% of loss from directional
rho_cap = 0.50     # SAFETY: max 50%

lambda_eff = (mse_val * rho_target) / dir_val

# If too low: push up
if rho < rho_target * 0.8:
    lambda_eff *= (rho_target / max(rho, 1e-6))

# If too high: safety clamp
elif rho > rho_cap:
    lambda_eff *= (rho_cap / rho)
```

**Impact**: Actively pushes ρ to 35% (vs passive cap at 25%)

---

### 2. Epoch-Gated Schedule (STRONGER)

| Component | v2.1 | v2.2 | Change |
|-----------|------|------|--------|
| **ρ target (E1-2)** | N/A (cap only) | 0.15 | NEW |
| **ρ target (E3-4)** | N/A (cap only) | 0.25 | NEW |
| **ρ target (E5+)** | N/A (cap only) | 0.35 | NEW |
| **ρ cap (all)** | 0.25 | 0.35-0.50 | +40-100% |
| **margin_gap (E9+)** | 0.05 | 0.07 | +40% |
| **λ_max** | 0.02 | 0.03 | +50% |

**Impact**: Higher directional pressure throughout training

---

### 3. Stronger Anchors

| Anchor | v2.1 | v2.2 | Impact |
|--------|------|------|--------|
| **pos_floor τ** | 0.10 | 0.12 | +20% threshold |
| **pos_floor β** | 1e-3 | 2e-3 | 2x weight |
| **Orthogonality κ** | N/A | 5e-4 | NEW penalty |

**Impact**: Harder for model to "win" with negative cosines

---

### 4. New Loss Component: Orthogonality Penalty

```python
def orthogonality_penalty(pred, y_prev):
    """
    Gently penalizes similarity to previous vector.
    penalty = (cos(pred, prev))^2
    """
    cos_prev = (pred * y_prev).sum(dim=-1)
    return cos_prev.pow(2).mean()
```

**Usage**:
```python
orth_pen = orthogonality_penalty(pred, y_prev)
loss = loss + 5e-4 * orth_pen  # κ = 5e-4
```

**Impact**: Chips away at backward bias without breaking legitimate similarity

---

## 📈 Expected Results

### Epoch-by-Epoch Trajectory

| Epochs | Margin | ρ | Directional Pressure |
|--------|--------|---|----------------------|
| 1-2 | -0.04 | 0.15 | Baseline (gentle) |
| 3-4 | -0.02 to -0.01 | 0.25 | Climbing |
| **5-6** | **0.00 → +0.01** | **0.35** | **FLIP POINT!** |
| 7-9 | +0.02 to +0.04 | 0.35 | Stabilize |
| 10-12 | +0.03 to +0.05 | 0.35 | **TARGET** |

### Why Epochs 5-6 for Flip?

**Math**:
- v2.1: ρ ≈ 0.10-0.25, margin improved +0.035 (-0.082 → -0.047)
- v2.2: ρ ≈ 0.35-0.50, expected improvement +0.05 to +0.07
- Need to overcome: -0.047 (current) + 0.000 (target) = +0.047 improvement
- With ρ=0.35 + stronger anchors + orthogonality: ~+0.05 to +0.07 ✅
- By epoch 6: 50% through training, should see ~75% of improvement

---

## 🔧 Implementation Details

### Files Modified

1. **app/lvm/losses_directional.py** (lines 488-525):
   - Added `orthogonality_penalty()` function

2. **app/lvm/train_unified.py** (lines 639-754):
   - Added P6b v2.2 training logic block
   - Implemented ρ-controller (lines 687-710)
   - Epoch-gated schedule (lines 650-662)
   - Enhanced logging with ρ_target, ρ_cap

3. **scripts/train_transformer_p6b_v22.sh** (NEW):
   - Training script with --p6b-v22 flag
   - Enhanced diagnostics output

### CLI Usage

```bash
# Launch v2.2 training
./scripts/train_transformer_p6b_v22.sh

# Or manually
./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --data artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device mps \
  --p6b-v22 \
  --fivecat-every-epoch 1 \
  --output-dir artifacts/lvm/models/transformer_p6b_v22_TEST
```

---

## 🎯 Success Criteria

### Training Metrics (Per Epoch)

- [ ] ρ tracks ρ_target (within ±0.05)
- [ ] skip = 0 (no collapse warnings)
- [ ] pos > 0.0 AND neg > 0.0 (both cosines positive)
- [ ] Margin climbing +0.01 to +0.02 every 2-3 epochs
- [ ] R@5 ≥ 70% (maintained)

### Final 5CAT Validation

- [ ] **Margin > 0** (POSITIVE!)
- [ ] Pass ≥ 3/5 gates
- [ ] Gate A (Offset): Margin ≥ +0.10 (VAL), ≥ +0.08 (OOD)
- [ ] Gate B (Retrieval): R@1 ≥ 55%, R@5 ≥ 92%
- [ ] Gate D (Rollout): avg_cos@H=5 ≥ 0.45

---

## 🔬 Monitoring Guide

### Every 200 Steps (Watch Log Output)

```
[P6b v2.2] λ_eff=0.00215 pos=0.512 neg=0.518 gap=-0.006 ratio=-0.008
           ρ=0.348 ρ_tgt=0.35 ρ_cap=0.50 frac=0.35 margin_gap=0.06 skip=0
```

**Key Metrics**:
- **ρ vs ρ_tgt**: Should be close (±0.05) → controller working
- **skip=0**: No collapse warnings → model stable
- **pos, neg > 0**: Both cosines positive → no death spiral
- **gap climbing**: Should increase over training

### Every Epoch (Mini-5CAT)

```
[Mini-5CAT] Margin: -0.0268 | R@5: 0.742
```

**Watch For**:
- Margin should climb +0.01 to +0.02 every 2-3 epochs
- R@5 should stay ≥ 70%
- If margin drops or R@5 crashes: training auto-saves and may exit

---

## 🚨 Troubleshooting

### If ρ doesn't track ρ_target

**Symptom**: ρ stays far from ρ_target (±0.10 or more)

**Cause**: Controller logic may need tuning

**Fix**: Check λ_eff adjustment logic (train_unified.py:700-704)

### If margin stays negative after epoch 6

**Symptom**: Margin < -0.01 at epoch 6

**Likely cause**: Still not enough directional pressure

**Options**:
1. Continue training to epoch 12 (may flip later)
2. Try v2.3 with exponential ramp
3. Remove adaptive guard (nuclear option)

### If skip rate > 1%

**Symptom**: Many "P6b v2.2 SKIP" warnings in logs

**Cause**: Model producing negative cosines (bad predictions)

**Fix**:
1. Check data quality (diagnose_data_direction.py)
2. Lower learning rate (5e-4 → 3e-4)
3. Increase warmup epochs

---

## 📚 Related Documentation

- **Session Handoff**: `artifacts/lvm/SESSION_HANDOFF_P6B_V22_2025_11_02.md`
- **v2.1 Implementation Guide**: `artifacts/lvm/P6B_V21_IMPLEMENTATION.md` (500+ lines)
- **Root Cause Analysis**: `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md`
- **5CAT Test**: `tools/tests/test_5to1_alignment.py`

---

## ✅ Verification Tests

```bash
# Import test
./.venv/bin/python -c "from app.lvm.losses_directional import orthogonality_penalty; print('✅')"

# Component test
./.venv/bin/python -c "
from app.lvm.losses_directional import (
    directional_margin_loss_v21,
    positive_floor_penalty,
    norm_regularization,
    orthogonality_penalty
)
import torch
pred, target, prev = torch.randn(32, 768), torch.randn(32, 768), torch.randn(32, 768)
dir_result = directional_margin_loss_v21(pred, target, prev)
pos_floor = positive_floor_penalty(pred, target, tau=0.12)
norm_pen = norm_regularization(pred)
orth_pen = orthogonality_penalty(pred, prev)
print('✅ All components working')
"

# CLI test
./.venv/bin/python app/lvm/train_unified.py --help | grep -A2 "p6b-v22"
```

**All tests passed** ✅

---

**Generated**: 2025-11-02
**Status**: ✅ READY TO TRAIN
**Command**: `./scripts/train_transformer_p6b_v22.sh`
