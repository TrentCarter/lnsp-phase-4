# Session Handoff: P5.1 + Curriculum Fix Complete

**Date**: November 2, 2025
**Status**: ✅ Implementation Complete, Ready to Train
**Next Action**: Run P5.1 training and evaluate results

---

## What Was Accomplished

### 1. P5.1 Landscape Reshaping Implementation ✅

**Added to `app/lvm/train_unified.py`:**
- Scheduler functions: `lin_ramp()`, `current_pos_scalar()`, `current_last_bias()`, `current_lambda_dir()`
- Noise function: `maybe_corrupt_last_slot()` (p=0.15, σ=0.03, swap=0.05)
- Attention bias builder: `build_lastcol_bias_mask()` (ready for future use)
- Mini-5CAT validation: `mini_5cat_epoch_metrics()` (margin + R@5, 2000 samples)
- Integrated all P5.1 features into training loop with proper scheduling

**Added to `app/lvm/losses_directional.py`:**
- `micro_directional_loss()` - Gentle ranking loss (softplus, γ=5.0, m=0.02, λ=0.001)

**New CLI Arguments (16 total):**
```bash
# Positional encoding ramp
--positional-scalar, --positional-ramp-epochs

# Attention bias
--attn-last-bias-max, --attn-last-bias-warmup-epochs

# Last-slot noise
--last-slot-noise-p, --last-slot-noise-sigma, --last-slot-swap-p

# Micro-directional guard
--lambda-dir, --dir-gamma, --dir-margin, --dir-warmup-epochs

# Mini-5CAT validation
--fivecat-every-epoch, --fivecat-max-samples

# Stage gates
--gate-min-margin, --gate-min-r-at5, --gate-min-rollout
```

### 2. Curriculum Fix Implementation ✅

**Upgraded `tools/compute_forward_distinctness.py`:**
- Old: Simple similarity `cos(target, ctx[-1])`
- New: Forward-advantage metrics
  - `sim_prev`: cos(target, ctx[-1])
  - `sim_prev2`: cos(target, ctx[-2])
  - `sim_other_max`: max(cos(target, ctx[i])) for i∈[0..3]
  - **`adv_prev`**: sim_prev - sim_other_max (KEY METRIC!)
  - `delta_prev2`: sim_prev - sim_prev2

**Upgraded `tools/build_curriculum_splits.py`:**
- Old: Top 30% by percentile
- New: Threshold-based selection
  - Stage A: `sim_prev ≥ 0.66 AND adv_prev ≥ 0.08`
  - Stage B: `sim_prev ≥ 0.58 OR adv_prev ≥ 0.05`
- Added fail-fast validation (rejects Stage A if mean adv < 0.02)
- Added CLI args: `--tau-sim-A`, `--tau-adv-A`, `--tau-sim-B`, `--tau-adv-B`

### 3. Integrated Training Script ✅

**Created `scripts/train_transformer_p5.1_curriculum.sh`:**
- Computes forward-advantage metrics
- Builds curriculum with validation
- Trains Stage A with all P5.1 enhancements
- Runs 5CAT validation after Stage A
- Auto-exits if Stage A fails gates

### 4. Documentation Updated ✅

- **CLAUDE.md**: Updated LVM training section with P5.1 status
- **P5.1_IMPLEMENTATION_SUMMARY.md**: Complete technical reference (139 KB)
- **SESSION_HANDOFF_2025_11_02.md**: This file

---

## Testing Completed

### Unit Tests ✅
- `micro_directional_loss()` - Returns valid loss tensor
- `lin_ramp()` - Schedules correctly (0→0.10 over 3 epochs)
- `maybe_corrupt_last_slot()` - Corrupts context safely
- `build_lastcol_bias_mask()` - Creates valid attention bias

### Integration Tests ✅
- `compute_forward_distinctness.py` - Computes all 5 metrics correctly
- `build_curriculum_splits.py` - Fail-fast validation works (correctly rejects synthetic data)

### Synthetic Data Test ✅
```
✅ Random data correctly rejected by curriculum builder
❌ FAIL: Stage A has ZERO samples!
[CURR/ERROR] Stage A is empty. Relax thresholds or check score calculation.
```
**This is expected behavior** - random data has no forward-prediction signal.

---

## Next Steps

### Immediate: Run P5.1 Training

```bash
./scripts/train_transformer_p5.1_curriculum.sh
```

**Expected Stage A Results (4 epochs):**
- Epoch 1-2: Margin should flip from negative to positive
- Epoch 3-4: Margin ≥ +0.02, R@5 ≥ 60%
- Gates: ✅ PASS → Proceed to Stage B

### If Stage A Passes

1. **Celebrate!** 🎉 Backward bias is broken
2. Proceed to Stage B (uncomment in script)
3. Proceed to Stage C (uncomment in script)
4. Run full 5CAT validation (5000 samples)
5. Deploy if 5CAT passes ≥3/5 gates

### If Stage A Fails

Try stronger settings (in order):
1. **Nuclear positional**: `--positional-scalar 0.15` (was 0.10)
2. **Stronger bias**: `--attn-last-bias-max 0.8` (was 0.6)
3. **P6 [NEXT] token**: Architectural change (removes identity path by design)

---

## Key Files

### Training
- **Main script**: `scripts/train_transformer_p5.1_curriculum.sh`
- **Training code**: `app/lvm/train_unified.py` (P5.1 integrated)
- **Loss functions**: `app/lvm/losses_directional.py` (micro-directional)

### Curriculum
- **Score computation**: `tools/compute_forward_distinctness.py`
- **Curriculum builder**: `tools/build_curriculum_splits.py`

### Documentation
- **Quick reference**: `CLAUDE.md` (LVM training section)
- **Technical details**: `artifacts/lvm/P5.1_IMPLEMENTATION_SUMMARY.md`
- **Root cause**: `LVM_CURRICULUM_FIX_ANALYSIS.md`

---

## Technical Summary: Why This Should Work

| Component | What It Does | Why It Helps |
|-----------|--------------|--------------|
| **Forward Advantage (Data)** | Selects samples where ctx[-1] is uniquely best | Copy-last is CORRECT for these samples! |
| **Positional Ramp (0→0.10)** | Makes time visible to model | Can't ignore temporal order |
| **Attention Bias (0→0.6)** | Reduces attention to ctx[-1] | Makes copying mechanically harder |
| **Last-Slot Noise (p=0.15)** | Corrupts ctx[-1] during training | Copy-last becomes unreliable |
| **Micro-Dir Guard (λ=0.001)** | Gentle nudge to prefer next | Doesn't destabilize MSE |
| **Mini-5CAT Every Epoch** | Validates margin/R@5 | Catches problems at epoch 1-2 |

**Synergy**: Curriculum ensures "copy-last" is *locally correct* for training samples, while P5.1 ensures model learns *generalizable* forward prediction instead of memorizing the shortcut.

---

## Comparison: What Changed from P5

| Feature | P5 (Failed) | P5.1 (New) |
|---------|-------------|------------|
| Positional | Fixed 0.03 | 0 → 0.10 (ramped, 3.3x stronger) |
| Curriculum | Top 30% by 1.0-sim | Threshold-based by adv_prev |
| Validation | None | Fail-fast (mean adv ≥ 0.02) |
| Mini-5CAT | After stage | **Every epoch** |
| Noise | None | p=0.15, σ=0.03 |
| Attention | None | 0 → 0.6 bias |
| Guard | None | Micro-dir λ=0.001 |

**P5 Result**: Margin -0.041, R@5 17.5% (FAILED)
**P5.1 Expected**: Margin ≥ +0.02, R@5 ≥ 60% (target)

---

## Diagnostic Commands

### Check Curriculum Quality
```bash
./.venv/bin/python tools/compute_forward_distinctness.py \
  --npz artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz

# Look for:
# - adv_prev mean > 0.02
# - % adv_prev > 0 should be > 50%
```

### Verify Stage A Split
```bash
./.venv/bin/python tools/build_curriculum_splits.py \
  --train-npz artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --scores-npz artifacts/lvm/training_sequences_ctx5_584k_clean_splits_forward_scores.npz \
  --tau-sim-A 0.66 --tau-adv-A 0.08

# Should pass all validation checks:
# ✅ PASS: Stage A has N forward-advantaged samples
# ✅ PASS: Stage A mean(adv_prev) = 0.XXX ≥ 0.02
# ✅ PASS: Stage A has XX.X% samples with adv_prev > 0
```

### Monitor Training
```bash
# Watch for margin flip in mini-5CAT output
tail -f artifacts/lvm/models/transformer_p5.1_*/training.log

# Expected:
# Epoch 1: [Mini-5CAT] Margin: -0.015  ← Still backward
# Epoch 2: [Mini-5CAT] Margin: +0.008  ← Margin flips!
# Epoch 3: [Mini-5CAT] Margin: +0.028  ← Above threshold
```

---

## Clean Context Reset Ready

✅ All code implemented and tested
✅ Documentation updated (CLAUDE.md, technical summary, handoff)
✅ Training script ready to run
✅ Clear next steps defined

**Safe to `/clear` and start fresh session.**

---

**Last Updated**: November 2, 2025
**Next Session**: Run P5.1 training and analyze Stage A results
