# V3 Quick Reference Card

**Last Updated**: 2025-10-31
**Status**: ✅ READY FOR TRAINING

---

## 🚀 Quick Start

```bash
# Start training (one command)
./scripts/train_transformer_directional_v3.sh

# Monitor progress (optional, separate terminal)
./scripts/monitor_training_v3.sh

# Check at epoch 5 (recommended)
./scripts/check_5cat_epoch.sh \
  artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt 500
```

**Training Time**: ~20-30 minutes on MPS

---

## 📋 What is V3?

**Problem**: Transformer 584k model predicts BACKWARD (k=-1) instead of FORWARD (k=+1)

**Solution**: V3 directional guardrails with:
1. Scheduled ramp-up (prevents collapse)
2. Positional encoding (breaks time symmetry)
3. Lightweight directional losses (λ=0.01)

**Expected Result**: Margin ≥ +0.10, val cosine ≥ 0.54, passes 3/5 5CAT gates

---

## 📖 Documentation

### Start Here
`artifacts/lvm/V3_COMPLETE_IMPLEMENTATION_SUMMARY.md` - Complete overview

### Technical Details
- `artifacts/lvm/V3_DIRECTIONAL_GUARDRAILS_COMPLETE.md` - Loss formulas, hyperparameters
- `artifacts/lvm/POSITIONAL_ENCODING_FIX.md` - Dimension mismatch fixes
- `artifacts/lvm/SESSION_SUMMARY_2025_10_31.md` - This session's work

### Updated
- `CLAUDE.md` - Added V3 status section (line 191)

---

## 🎯 Success Criteria

Training is successful if:
- ✅ Completes all 20 epochs
- ✅ Final margin ≥ +0.10 (positive = forward prediction)
- ✅ Final val cosine ≥ 0.54 (maintains performance)
- ✅ k=+1 is peak in offset sweep
- ✅ Passes 3/5 5CAT gates

---

## 🔧 Key Files

### Training
- `scripts/train_transformer_directional_v3.sh` - Main training script
- `app/lvm/train_unified.py` - Training loop (scheduling implemented)
- `app/lvm/losses_directional.py` - Loss functions

### Validation
- `tools/tests/test_5to1_alignment.py` - 5CAT test
- `scripts/check_5cat_epoch.sh` - Quick helper

### Models
- `app/lvm/models.py` - 4 architectures updated (input_dim ≠ output_dim)

---

## ⚠️ If Training Fails

### Dimension Mismatch
Should not happen - 3 fixes applied. If it does, see:
`artifacts/lvm/POSITIONAL_ENCODING_FIX.md`

### Negative Margin
Guards too weak → retrain with λ=0.015

### Low Val Cosine (<0.45)
Guards too strong → retrain with λ=0.007

### k=+3 Drift
Enable future margin loss (requires article-aware batching)

---

## 🎓 What Changed

### Before
- Model copied position 4 (last context)
- Margin = -0.166 (negative = backward)
- Val cosine = 0.558 (good, wrong direction)

### V1 Attempt
- Added strong losses (λ=0.05)
- Result: Fixed direction BUT collapsed (val cosine 0.158)

### V2 Design
- Lighter losses (λ=0.01)
- No scheduling
- Not tested (went to V3)

### V3 (Current)
- Scheduled ramp-up (warm-up → ramp → full)
- Positional encoding (breaks symmetry)
- Lightweight losses (λ=0.01)
- Expected: Fixes direction WITHOUT collapse

---

## 📊 Training Schedule

| Phase | Epochs | Guards | Expected |
|-------|--------|--------|----------|
| Warm-up | 1-3 | OFF (λ=0) | Val ~0.48-0.52, margin negative |
| Ramp | 4-10 | 0.005→0.01 | Margin turns positive by epoch 7-8 |
| Full | 11-20 | λ=0.01 | Margin ≥+0.10, val ≥0.54 |

---

## 🛠️ Helper Commands

```bash
# Training
./scripts/train_transformer_directional_v3.sh

# Monitor
./scripts/monitor_training_v3.sh

# Quick 5CAT (epoch 5)
./scripts/check_5cat_epoch.sh \
  artifacts/lvm/models/transformer_directional_v3/checkpoint_epoch_5.pt 500

# Full 5CAT (final)
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_directional_v3/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

---

## 💡 Key Insights

1. **Scheduling prevents collapse** - V1 without scheduling failed immediately
2. **Positional encoding is cheap and effective** - 1 extra dim, breaks time symmetry
3. **Light losses are crucial** - λ=0.05 too strong, λ=0.01 just right
4. **Dimension consistency matters** - 3 fixes required for positional encoding

---

**Status**: ✅ Ready to train
**Next**: Run `./scripts/train_transformer_directional_v3.sh`
**Docs**: `artifacts/lvm/V3_COMPLETE_IMPLEMENTATION_SUMMARY.md`

🚀 Good luck!
