# P6b Implementation Complete ✅

**Date**: 2025-11-02
**Status**: Ready for training

---

## What Was Implemented

### 1. Smooth Directional Margin Loss (`app/lvm/losses_directional.py`)

**Function**: `directional_margin_loss_smooth()`

```python
def directional_margin_loss_smooth(
    pred: torch.Tensor,
    y_next: torch.Tensor,
    y_prev: torch.Tensor,
    *,
    margin: float = 0.05,
    gamma: float = 8.0,
):
    """
    P6b: Smooth directional margin loss using softplus.

    Loss = softplus(gamma * (margin - gap)) / gamma
    where gap = cos(pred, next) - cos(pred, prev)

    Returns: (loss, pos_mean, neg_mean, gap_mean)
    """
```

**Key Features**:
- Smooth hinge (softplus) instead of hard ReLU
- Scale-friendly (works with auto-scaling λ_eff)
- Returns diagnostic metrics (pos, neg, gap)

**Test Results**:
```
✅ Loss: 0.112429
   Pos (next): 0.0056
   Neg (prev): -0.0004
   Gap: 0.0060
   Expected: Gap ≈ 0 (random vectors) ✓
```

---

### 2. P6b Training Logic (`app/lvm/train_unified.py`)

**Location**: Lines 480-526 (inside `train_epoch()`)

**Auto-Scaled Directional Loss**:
```python
if p6b_directional:
    # Extract y_prev (hard negative: previous chunk)
    y_prev = F.normalize(contexts_orig[:, -2, :], dim=-1, p=2)

    # Ramped margin schedule
    if epoch < 2:
        dir_margin_curr = 0.02
        target_frac = 0.10     # 10% of MSE
    elif epoch < 5:
        dir_margin_curr = 0.04
        target_frac = 0.20     # 20% of MSE
    else:
        dir_margin_curr = 0.05
        target_frac = 0.25     # 25% of MSE (steady-state)

    # Compute raw directional loss
    dir_raw, pos_mu, neg_mu, gap_mu = directional_margin_loss_smooth(
        pred_cos, targets, y_prev,
        margin=dir_margin_curr,
        gamma=8.0
    )

    # Auto-scale λ_eff to keep directional term proportional to MSE
    with torch.no_grad():
        mse_val = loss.detach().clamp_min(1e-8)
        dir_val = dir_raw.detach().clamp_min(1e-8)
        lambda_eff = (mse_val * target_frac) / dir_val
        lambda_eff = lambda_eff.clamp(1e-4, 5e-2)  # Safety: [0.0001, 0.05]

    # Add scaled directional term
    loss = loss + lambda_eff * dir_raw
```

**Logging** (every 200 steps):
```python
print(
    f"[P6b dir] λ_eff={lambda_eff.item():.5f} "
    f"pos={pos_mu:.3f} neg={neg_mu:.3f} gap={gap_mu:.3f} "
    f"frac_of_mse={frac.item():.2f} margin={dir_margin_curr:.3f}"
)
```

**CLI Argument**:
```python
parser.add_argument('--p6b-directional', action='store_true',
    help='Enable P6b smooth directional loss (auto-scaled to MSE)')
```

---

### 3. Training Script (`scripts/train_transformer_p6b_directional.sh`)

**Usage**:
```bash
./scripts/train_transformer_p6b_directional.sh [DATA] [VAL] [OOD] [ART] [DEVICE]
```

**Defaults**:
```bash
DATA: artifacts/lvm/training_sequences_ctx5_p6_next_token.npz
VAL:  artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz
OOD:  artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz
ART:  artifacts/wikipedia_584k_fresh.npz
DEVICE: mps
```

**Training Configuration**:
- Model: Transformer (4-layer, 512D, 8 heads)
- Epochs: 12
- Batch size: 32
- Learning rate: 5e-4
- Device: MPS (or CPU)
- P6b directional loss: ENABLED
- Mini-5CAT: Every epoch (2000 samples)

**Outputs**:
- Best model: `artifacts/lvm/models/transformer_p6b_YYYYMMDD_HHMMSS/best_model.pt`
- 5CAT results: `artifacts/lvm/models/transformer_p6b_YYYYMMDD_HHMMSS/5cat_results.json`
- Training history: `artifacts/lvm/models/transformer_p6b_YYYYMMDD_HHMMSS/training_history.json`

---

## Why P6b Will Work

### Problem (P1-P6 All Failed)

**Root Cause**: Wikipedia data has inherent backward temporal bias
- Forward (ctx[-1] → target_next): 0.3876
- Backward (ctx[-1] → target_prev): 0.4569
- **Δ = -0.0692** (backward is 7% stronger!)

**Evidence**:
- P1 Baseline (MSE only): margin = -0.167
- P2-P4 (Directional): Collapsed or negative
- P5.1 (Curriculum): margin = -0.046
- P6 (NEXT token): margin = **-0.082** (still negative!)

**P6 Proved**: Problem is data, not architecture
- Removed identity path (cos(ctx[4], target_next) = 0.395)
- Yet margin still negative → **data teaches backward**

### Solution (P6b)

**Two-Part Fix**:
1. **P6 Architecture**: Removes copy-last shortcut
   - Predicts target_next (not target)
   - Cannot copy ctx[4] (too dissimilar)

2. **Directional Margin Loss**: Overrides backward data signal
   - Explicitly enforces: `cos(pred, next) > cos(pred, prev) + margin`
   - Auto-scaled to MSE magnitude (prevents collapse)
   - Ramped margin (0.02 → 0.05 over 5 epochs)

**Expected Behavior**:
- Epochs 1-2: Margin climbs from -0.08 → 0.0 (neutralizing backward bias)
- Epochs 3-5: Margin goes positive (+0.02 → +0.05)
- Epochs 6-12: Margin stabilizes (+0.05 → +0.10)
- R@5 stays high throughout (≥ 70%)
- Val cosine good (≥ 0.50)

---

## Expected Results (10-12 Epochs)

### Mini-5CAT (Per-Epoch Validation)

| Epoch | Margin | R@5 | Notes |
|-------|--------|-----|-------|
| 1-2   | -0.05 → 0.0 | 0.60-0.65 | Climbing from negative |
| 3-5   | 0.0 → +0.05 | 0.65-0.70 | **Flips positive!** |
| 6-12  | +0.05 → +0.10 | 0.70-0.75 | Stable positive margin |

### Full 5CAT (Final Evaluation)

**Gate A: Offset Sweep** ✅
- k=-1: ~0.45 (prev, lower)
- k=0: ~0.52 (next, higher!)
- **Margin(+1): +0.05 to +0.10** (POSITIVE!)

**Gate B: Retrieval** ✅
- R@1: ≥ 60%
- R@5: ≥ 70%

**Gate D: Rollout** ✅
- avg_cos@H=5: ≥ 0.45

**Gate E: Bins Delta** ✅
- |Val - OOD| ≤ 0.05

**Overall**: Pass 4/5 gates (A, B, D, E) ✅

---

## Training Logs to Watch

### Good Signs ✅

```
[P6b dir] λ_eff=0.00235 pos=0.520 neg=0.468 gap=0.052 frac_of_mse=0.18 margin=0.050
[Mini-5CAT] Margin: +0.052 | R@5: 0.702
✅ Positive margin! Model predicts NEXT, not last context
```

**What to check**:
- λ_eff stays in [0.001 - 0.02] (auto-scaling works)
- gap increases over epochs (0.0 → +0.05 → +0.10)
- frac_of_mse around 0.15-0.35 (directional is 15-35% of MSE)
- Mini-5CAT margin goes positive by epoch 3-5

### Warning Signs ⚠️

```
[P6b dir] λ_eff=0.00001 pos=0.402 neg=0.456 gap=-0.054 frac_of_mse=0.01 margin=0.020
[Mini-5CAT] Margin: -0.080 | R@5: 0.450
⚠️  Negative margin - still copying last context
```

**What this means**:
- λ_eff too small → increase target_frac (0.25 → 0.35)
- gap still negative → check data quality (should be P6 format!)
- R@5 dropping → model struggling, increase margin ramp slower

---

## How to Run

### Quick Start (Default Settings)

```bash
./scripts/train_transformer_p6b_directional.sh
```

### Custom Settings

```bash
./scripts/train_transformer_p6b_directional.sh \
  artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
  artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz \
  artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz \
  artifacts/wikipedia_584k_fresh.npz \
  cpu  # Use CPU instead of MPS
```

### Manual Training (With Custom Args)

```bash
./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --data artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device mps \
  --p6b-directional \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir artifacts/lvm/models/transformer_p6b_test
```

---

## Success Criteria

### Training Metrics ✅

- [ ] Val cosine ≥ 0.50 by epoch 5
- [ ] Mini-5CAT margin goes positive by epoch 5
- [ ] R@5 ≥ 70% by epoch 8
- [ ] λ_eff stays in [0.001 - 0.02] throughout
- [ ] Gap increases steadily (0.0 → +0.05 → +0.10)

### 5CAT Gates (Pass 3/5 Minimum) ✅

- [ ] **A: Offset Sweep** → margin(+1) ≥ +0.05
- [ ] **B: Retrieval** → R@5 ≥ 70%
- [ ] **D: Rollout** → avg_cos ≥ 0.45
- [ ] **E: Bins Delta** → |Val-OOD| ≤ 0.05
- [ ] **C: Ablations** (optional, may fail due to weak context dependency)

### Deployment Criteria ✅

- [ ] Margin is **POSITIVE** (≥ +0.05)
- [ ] R@5 ≥ 70% on OOD data
- [ ] No backward bias in any test

---

## Files Modified

### Created
- `scripts/train_transformer_p6b_directional.sh` (126 lines)
- `artifacts/lvm/P6B_IMPLEMENTATION_COMPLETE.md` (this file)

### Modified
- `app/lvm/losses_directional.py`
  - Added `directional_margin_loss_smooth()` (lines 320-362)

- `app/lvm/train_unified.py`
  - Import smooth loss (line 49)
  - Add `epoch` param to `train_epoch()` (line 271)
  - Add `p6b_directional` param (line 300)
  - P6b training logic (lines 480-526)
  - CLI argument `--p6b-directional` (line 878)
  - Pass `epoch` to `train_epoch()` (line 1142)
  - Pass `p6b_directional` flag (line 1171)

---

## Next Steps

### 1. Launch Training

```bash
./scripts/train_transformer_p6b_directional.sh
```

### 2. Monitor Logs (First 3 Epochs)

Watch for:
- λ_eff in range [0.001 - 0.02]
- gap climbing toward 0 then positive
- Mini-5CAT margin going positive

### 3. Check 5CAT Results

After training completes:
- Margin should be **positive** (≥ +0.05)
- R@5 should be high (≥ 70%)
- Pass 3/5 gates minimum

### 4. Deploy or Iterate

**If successful** (margin ≥ +0.05, pass 3/5 gates):
- ✅ Deploy model to production
- ✅ Update CLAUDE.md with P6b as recommended approach
- ✅ Document results in session summary

**If margin still negative** (unlikely):
- Increase target_frac: 0.25 → 0.35 (more directional signal)
- Extend epochs: 12 → 15 (more time to flip)
- Check data quality (verify P6 format with `diagnose_p6_direction.py`)

---

## Theoretical Background

### Why Auto-Scaling Works

**Problem with Fixed λ**:
- Too small: Directional loss ignored, backward bias wins
- Too large: Dominates MSE, training collapses

**Solution: Auto-Scale λ_eff**:
```python
λ_eff = (MSE * target_frac) / dir_loss
```

**Effect**:
- Early epochs (high MSE): λ_eff large (strong directional signal)
- Late epochs (low MSE): λ_eff small (gentle refinement)
- Directional term stays proportional to MSE (15-35%)

### Why Ramped Margin Works

**Problem with Fixed Margin**:
- High margin early: Model can't satisfy, loss explodes
- Low margin late: Model doesn't improve beyond baseline

**Solution: Ramp Margin**:
- Epochs 1-2: margin=0.02 (easy target, model learns basics)
- Epochs 3-5: margin=0.04 (moderate target, pushing forward)
- Epochs 6+: margin=0.05 (final target, stable training)

**Effect**:
- Model gradually learns forward preference
- Avoids early collapse or late stagnation
- Smooth training curve

---

## References

- Root Cause Analysis: `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md`
- Session Summary: `artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md`
- Session Handoff: `artifacts/lvm/SESSION_HANDOFF_P6b_2025_11_02.md`
- P6 Diagnostics: `tools/diagnose_p6_direction.py`
- CLAUDE.md: Lines 191-279 (P6b status)

---

**Implementation Date**: 2025-11-02
**Status**: ✅ Ready for training
**Estimated Training Time**: 2-3 hours on MPS, 4-6 hours on CPU
**Expected Outcome**: Margin flips positive, passes 3/5 gates minimum
