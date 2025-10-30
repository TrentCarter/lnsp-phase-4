# AMN 790K Production Training Runbook

**Date**: 2025-10-30
**Objective**: Retrain AMN on 790k dataset with corrected config (MSE+InfoNCE)
**Expected Duration**: 9-12 hours (30 epochs)

---

## 🚨 CRITICAL: What Was Fixed

The failed 790k training (OOD: -0.0118) was caused by **MSE-only training**:
```bash
# ❌ BROKEN CONFIG (used in failed run):
--lambda-mse 1.0
--lambda-info 0.0   # InfoNCE DISABLED!
```

**Root Cause**: MSE-only collapses in 768D space on diverse data. Model learned to output vectors orthogonal to targets (cosine ~0.004-0.064).

```bash
# ✅ CORRECTED CONFIG (for this run):
--lambda-mse 0.5    # Balanced MSE
--lambda-info 0.5   # InfoNCE ENABLED (THE FIX!)
--lambda-moment 0.001
--lambda-variance 0.001
--tau 0.07
```

---

## 🎯 Expected Results

| Metric | Failed (MSE-only) | Target (MSE+InfoNCE) | Recovery |
|--------|------------------|---------------------|----------|
| In-Dist | 0.4607 ❌ | 0.56-0.58 ✅ | +21-26% |
| OOD | -0.0118 ❌❌❌ | 0.63-0.65 ✅ | +6300%! |

---

## 🚀 Launch Command

### Terminal 1: Start Training
```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Make script executable
chmod +x scripts/train_amn_790k_production.sh

# Launch training
bash scripts/train_amn_790k_production.sh
```

**NOTE**: Warm-start from 584k checkpoint is not yet implemented in `train_unified.py`. Training will start from scratch with corrected config. This is still expected to work - InfoNCE provides the directional guidance needed.

### Terminal 2: Monitor Progress (Optional)
```bash
# Live monitoring dashboard (updates every 30s)
bash tools/monitor_training_live.sh artifacts/lvm/models/amn_790k_production_*/training.log
```

---

## 📊 Monitoring Commands

### Quick Health Check
```bash
# Latest metrics
tail -20 artifacts/lvm/models/amn_790k_production_*/training.log

# Val cosine trend (last 10 epochs)
grep "val_cosine" artifacts/lvm/models/amn_790k_production_*/training.log | tail -10

# Loss breakdown
grep "loss_info\|loss_mse" artifacts/lvm/models/amn_790k_production_*/training.log | tail -20
```

### Stoplight Gate Checks

#### Epoch 3 Gate (Early Signal)
```bash
# Should see: val_cosine ≥ 0.48
grep "Epoch 3/" artifacts/lvm/models/amn_790k_production_*/training.log -A 5 | grep val_cosine
```
**RED FLAG**: If val < 0.45, InfoNCE may not be working. Check logs for errors.

#### Epoch 6 Gate (Confirmation)
```bash
# Should see: val_cosine ≥ 0.50
grep "Epoch 6/" artifacts/lvm/models/amn_790k_production_*/training.log -A 5 | grep val_cosine
```
**RED FLAG**: If val < 0.48, training is trending wrong. Consider stopping.

#### Epoch 10 Gate (Mid-Point)
```bash
# Should see: val_cosine ≥ 0.50, steady improvement
grep "Epoch 10/" artifacts/lvm/models/amn_790k_production_*/training.log -A 5 | grep val_cosine
```

#### Epoch 20 Gate (Late Stage)
```bash
# Should see: val_cosine ≥ 0.54
grep "Epoch 20/" artifacts/lvm/models/amn_790k_production_*/training.log -A 5 | grep val_cosine
```

#### Epoch 30 Gate (Final)
```bash
# Target: val_cosine 0.56-0.58
grep "Epoch 30/" artifacts/lvm/models/amn_790k_production_*/training.log -A 5 | grep val_cosine
```

---

## 🔬 OOD Sentinel (Automatic)

The training script includes an automatic OOD sentinel that:
- Runs OOD evaluation at epochs 3, 6, 10, 20
- **Kills training** if gates fail:
  - Epoch 3: OOD < 0.30 → ABORT
  - Epoch 6: OOD < 0.45 → ABORT

Check sentinel log:
```bash
tail -f artifacts/lvm/models/amn_790k_production_*/ood_sentinel.log
```

---

## 📈 Expected Training Progression

### Healthy Training (MSE+InfoNCE):
```
Epoch 1:  val_cosine ~0.44-0.46  (good start, InfoNCE working)
Epoch 3:  val_cosine ~0.47-0.49  (✅ gate passed)
Epoch 6:  val_cosine ~0.49-0.51  (✅ gate passed)
Epoch 10: val_cosine ~0.51-0.53  (steady climb)
Epoch 20: val_cosine ~0.54-0.56  (✅ gate passed)
Epoch 30: val_cosine ~0.56-0.58  (🎯 target reached!)
```

### Unhealthy Training (If InfoNCE Fails):
```
Epoch 1:  val_cosine ~0.42-0.43  (low start)
Epoch 3:  val_cosine ~0.43-0.44  (❌ not improving)
Epoch 6:  val_cosine ~0.44-0.45  (❌ flat/slow)
→ ABORT: Same mode collapse as before
```

---

## 🚦 Real-Time Diagnostics

### Loss Component Trends
```bash
# MSE should decrease steadily
grep "train_loss_mse" artifacts/lvm/models/amn_790k_production_*/training.log | tail -20

# InfoNCE should decrease then stabilize
grep "train_loss_info" artifacts/lvm/models/amn_790k_production_*/training.log | tail -20
```

**Healthy pattern**:
- MSE: 0.0012 → 0.0010 → 0.0009 (gradual decrease)
- InfoNCE: 2.5 → 1.8 → 1.5 → 1.3 (faster initial drop, then stable)

### Cosine Similarity Check
```bash
# Training cosine (batch-level)
grep "train_cosine" artifacts/lvm/models/amn_790k_production_*/training.log | tail -10
```

**Healthy**: Should be within 0.02 of val_cosine (not diverging)

---

## ⚠️ Troubleshooting

### Problem: Val Cosine Stuck Below 0.45 After Epoch 5
**Diagnosis**: InfoNCE not engaging properly
**Fix**: Check log for errors related to InfoNCE computation. Verify tau=0.07 is being used.

### Problem: Val Cosine Climbing But OOD Dropping
**Diagnosis**: Overfitting to sequential patterns
**Fix**: (For next run) Increase context window to 7 or add dropout

### Problem: Training Crashes with "Abort trap: 6"
**Diagnosis**: OpenMP duplicate library issue
**Fix**: Verify `KMP_DUPLICATE_LIB_OK=TRUE` is set (script includes this)

### Problem: OOD Sentinel Kills Training at Epoch 3
**Diagnosis**: Catastrophic failure - model not learning
**Actions**:
1. Check training log for errors in InfoNCE calculation
2. Verify data file is correct (726k sequences)
3. Verify lambda-info=0.5 is actually being used

---

## 📁 Output Files

After training completes:
```
artifacts/lvm/models/amn_790k_production_<timestamp>/
├── best_model.pt                 # Best checkpoint by val loss
├── epoch_1.pt, epoch_2.pt, ...   # Per-epoch checkpoints
├── training.log                  # Full training output
├── ood_sentinel.log              # OOD monitoring log
└── training_history.json         # Structured metrics
```

---

## ✅ Success Criteria

Training is successful if:
1. ✅ Val cosine ≥ 0.56 by epoch 30
2. ✅ OOD cosine ≥ 0.63 (run full eval after training)
3. ✅ No crashes or sentinel aborts
4. ✅ Loss components converging (not NaN/Inf)

---

## 📝 Post-Training Evaluation

Once training completes successfully:

### 1. Full OOD Evaluation
```bash
./.venv/bin/python tools/eval_model_ood.py \
    --model artifacts/lvm/models/amn_790k_production_*/best_model.pt \
    --ood-data artifacts/lvm/wikipedia_ood_test_ctx5.npz \
    --device mps
```

**Expected**: OOD cosine 0.63-0.65

### 2. Compare with 584k Baseline
```bash
# 584k baseline:
#   In-Dist: 0.5597
#   OOD: 0.6375

# 790k results:
#   In-Dist: (check training_history.json)
#   OOD: (check eval output)
```

### 3. Update Leaderboard
```bash
# Add results to:
artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md

# Format:
| AMN_790k | 0.XXX | 0.XXX | 0.XXms | 1.5M | MSE+InfoNCE | 2025-10-30 |
```

---

## 🔄 Next Steps (After AMN Success)

1. ✅ Mark AMN training complete in `TRAINING_RECOVERY_LOG.md`
2. 🔄 Train GRU on 790k: `bash scripts/train_gru_790k.sh`
3. 🔄 Train LSTM on 790k: `bash scripts/train_lstm_790k.sh`
4. 🔄 Train Transformer on 790k: `bash scripts/train_transformer_790k.sh`
5. 📊 Create comprehensive comparison table (584k vs 790k, all models)

---

## 📞 Emergency Contact

If training fails again with same symptoms (OOD < 0):
1. **DO NOT** retrain without diagnosis
2. Check `AMN_790K_FAILURE_ANALYSIS.md` for diagnostic steps
3. Verify InfoNCE is actually enabled in the command being run
4. Check logs for InfoNCE loss values (should be non-zero!)

---

**Created**: 2025-10-30 11:45 PST
**Status**: Ready to launch
**Estimated Completion**: 2025-10-30 20:45 PST (9 hours from start)
