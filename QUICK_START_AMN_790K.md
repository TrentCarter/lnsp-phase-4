# Quick Start: AMN 790K Production Training

**Status**: ✅ Ready to launch
**Duration**: 9-12 hours (30 epochs)
**Critical Fix**: InfoNCE re-enabled (was disabled in failed run)

---

## 🚀 Launch (Copy-Paste Ready)

### Terminal 1: Start Training
```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
bash scripts/train_amn_790k_production.sh
```

### Terminal 2: Live Monitor (Optional)
```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
bash tools/monitor_training_live.sh artifacts/lvm/models/amn_790k_production_*/training.log
```

---

## 📊 Quick Health Checks

### Latest Metrics (Run Anytime)
```bash
# Last 5 epochs
tail -100 artifacts/lvm/models/amn_790k_production_*/training.log | grep "val_cosine"

# OOD sentinel status
tail -20 artifacts/lvm/models/amn_790k_production_*/ood_sentinel.log
```

### Stoplight Gates (Expected Values)
```bash
# Epoch 3: val_cosine ≥ 0.48
grep "Epoch 3" artifacts/lvm/models/amn_790k_production_*/training.log | grep val_cosine

# Epoch 6: val_cosine ≥ 0.50
grep "Epoch 6" artifacts/lvm/models/amn_790k_production_*/training.log | grep val_cosine

# Epoch 20: val_cosine ≥ 0.54
grep "Epoch 20" artifacts/lvm/models/amn_790k_production_*/training.log | grep val_cosine

# Epoch 30: val_cosine ~0.56-0.58 (target)
grep "Epoch 30" artifacts/lvm/models/amn_790k_production_*/training.log | grep val_cosine
```

---

## ✅ Success Criteria

| Metric | Target | Check Command |
|--------|--------|---------------|
| **Epoch 3 Val** | ≥ 0.48 | `grep "Epoch 3" */training.log \| grep val` |
| **Epoch 6 Val** | ≥ 0.50 | `grep "Epoch 6" */training.log \| grep val` |
| **Epoch 30 Val** | 0.56-0.58 | `grep "Epoch 30" */training.log \| grep val` |
| **Final OOD** | 0.63-0.65 | `python tools/eval_model_ood.py --model */best_model.pt` |

---

## 🚨 Red Flags (Stop Training If...)

1. **Epoch 3 val < 0.45** → InfoNCE not working
2. **OOD Sentinel kills training** → Check `ood_sentinel.log`
3. **Loss = NaN/Inf** → Numerical instability
4. **No improvement after epoch 10** → Stuck in local minimum

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `scripts/train_amn_790k_production.sh` | Main training script (corrected config) |
| `artifacts/lvm/AMN_790K_PRODUCTION_RUNBOOK.md` | Detailed monitoring guide |
| `artifacts/lvm/AMN_790K_FAILURE_ANALYSIS.md` | Root cause analysis of failed run |
| `tools/monitor_training_live.sh` | Live dashboard (updates every 30s) |
| `tools/eval_model_ood.py` | Post-training OOD evaluation |

---

## 🔧 What Was Fixed

```diff
# Previous (FAILED):
--lambda-mse 1.0
- --lambda-info 0.0   ❌ InfoNCE DISABLED!
--epochs 20

# Current (CORRECTED):
--lambda-mse 0.5
+ --lambda-info 0.5   ✅ InfoNCE ENABLED!
+ --lambda-moment 0.001
+ --lambda-variance 0.001
+ --tau 0.07
--epochs 30
```

**Result**: MSE-only → MSE+InfoNCE (prevents mode collapse in 768D space)

---

## 📞 If Training Fails Again

1. **Check logs first**:
   ```bash
   tail -100 artifacts/lvm/models/amn_790k_production_*/training.log
   ```

2. **Verify InfoNCE is non-zero**:
   ```bash
   grep "train_loss_info" artifacts/lvm/models/amn_790k_production_*/training.log | tail -10
   # Should see values like 1.5-2.5, NOT 0.0!
   ```

3. **Check OOD sentinel**:
   ```bash
   cat artifacts/lvm/models/amn_790k_production_*/ood_sentinel.log
   ```

4. **Compare with failure analysis**: See `AMN_790K_FAILURE_ANALYSIS.md` lines 59-88

---

## ⏱️ Timeline

| Time | Event | Action |
|------|-------|--------|
| **T+0** | Launch training | Start monitoring |
| **T+1.5h** | Epoch 3 complete | Check val ≥ 0.48 |
| **T+3h** | Epoch 6 complete | Check val ≥ 0.50 |
| **T+4.5h** | Epoch 10 complete | Verify steady climb |
| **T+9h** | Epoch 20 complete | Check val ≥ 0.54 |
| **T+12h** | Training done | Run OOD eval |

---

## 🎯 Expected Final Results

| Metric | Failed Run | Target | Improvement |
|--------|-----------|--------|-------------|
| In-Dist | 0.4607 ❌ | 0.56-0.58 ✅ | +21-26% |
| OOD | -0.0118 ❌ | 0.63-0.65 ✅ | +6300%! |

---

**Created**: 2025-10-30 11:50 PST
**Ready**: ✅ All systems go!
