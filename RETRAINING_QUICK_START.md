# LVM Retraining - Quick Start Guide

**Created**: 2025-10-30
**Issue**: All production models have backward prediction bias
**Solution**: Retrain on clean 584k data with 5CAT validation

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Train One Model (Recommended - Test First)

```bash
# Train Transformer with 5CAT validation (~1.5 hours)
./scripts/train_with_5cat_validation.sh transformer
```

**What happens**:
- Loads clean 584k training data (coherence 0.457)
- Trains for 20 epochs with MSE loss
- Runs 5CAT test every 5 epochs
- Alerts if backward bias detected
- Saves best model + best 5CAT model
- Complete in ~1.5-2 hours

### Option 2: Train All Models (6-8 Hours Total)

```bash
# Train all 4 models sequentially
./scripts/retrain_all_production_models.sh
```

**Trains**:
1. Transformer (~2 hours)
2. GRU (~1.5 hours)
3. LSTM (~1.5 hours)
4. AMN (~1 hour)

---

## 📊 What You'll See

### During Training (Every 5 Epochs)

```
============================================
🧪 Running 5CAT Validation (Epoch 5)
============================================

📊 5CAT Results:
   VAL Margin: +0.145 (need ≥+0.12)  ✅
   OOD Margin: +0.138 (need ≥+0.10)  ✅
   VAL Rollout: 0.521 (need ≥0.45)   ✅
   OOD Rollout: 0.514 (need ≥0.42)   ✅

✅ Good forward prediction!
   Margin is +0.145
```

### Warning Signs (Would Trigger Alerts)

```
🚨 WARNING: BACKWARD BIAS DETECTED!
   Margin is -0.084 (negative)
   Model is learning to predict PREVIOUS vector instead of NEXT!
```

### After Training

```
✅ TRAINING COMPLETE!

Output directory: artifacts/lvm/models/transformer_5cat_20251030_220000/
Best val loss: 0.001234

Models saved:
  - best_model.pt        (lowest val loss)
  - best_5cat_model.pt   (best 5CAT metrics) ← Use this one!

Next steps:
  1. Run full 5CAT test (max-samples 5000)
  2. Update production symlinks
  3. Restart LVM services
```

---

## ✅ Expected Results

| Model | Val Cosine | Val Margin | OOD Margin | Training Time |
|-------|-----------|-----------|-----------|---------------|
| **Transformer** | 0.58-0.62 | **+0.12 to +0.18** ✅ | **+0.10 to +0.16** ✅ | ~2 hours |
| **GRU** | 0.56-0.60 | **+0.10 to +0.15** ✅ | **+0.09 to +0.14** ✅ | ~1.5 hours |
| **LSTM** | 0.55-0.59 | **+0.10 to +0.15** ✅ | **+0.09 to +0.14** ✅ | ~1.5 hours |
| **AMN** | 0.54-0.58 | **+0.08 to +0.12** ✅ | **+0.07 to +0.11** ✅ | ~1 hour |

**Key**: Positive margins mean forward prediction (correct!)

---

## 🔧 After Training

### 1. Run Full 5CAT Test

```bash
# Test final model with 5000 samples (10-15 min)
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/transformer_5cat_*/best_5cat_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

### 2. Update Production Symlinks

```bash
# Backup old models
mv artifacts/lvm/models/transformer_v0.pt artifacts/lvm/models/transformer_v0_OLD.pt

# Link new model
ln -s transformer_5cat_20251030_220000/best_5cat_model.pt \
      artifacts/lvm/models/transformer_v0.pt

# Repeat for gru_v0.pt, lstm_v0.pt, amn_v0.pt
```

### 3. Restart LVM Services

```bash
# Stop old services
./scripts/stop_lvm_services.sh

# Start with new models
./scripts/start_lvm_services.sh
```

### 4. Test Chat Endpoints

```bash
# Test Transformer (port 9002)
curl -X POST http://localhost:9002/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Hello, how are you?"], "temperature": 0.7}'

# Should now predict FORWARD (next text) instead of backward!
```

---

## 🚨 Troubleshooting

### If Training Shows Backward Bias

**Symptoms**:
```
🚨 WARNING: BACKWARD BIAS DETECTED!
   Margin is -0.124 (negative)
```

**Actions**:
1. Check data quality:
   ```bash
   ./.venv/bin/python tools/tests/diagnose_data_direction.py \
     artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz
   ```
2. Verify coherence ≥ 0.40, signal ≥ +0.08
3. If data is bad, regenerate training sequences
4. If data is good, check model architecture (context/target swap?)

### If Training is Slow

**On MPS (Apple Silicon)**:
- Expected: ~8 min/epoch for Transformer
- If slower: Check Activity Monitor for background processes

**Switch to CPU**:
```bash
./scripts/train_with_5cat_validation.sh transformer 20 cpu
```

### If Out of Memory

**Reduce batch size** (edit script line 125):
```python
batch_size=128  # Change from 256 to 128
```

---

## 📁 Output Files

After training, you'll find:

```
artifacts/lvm/models/transformer_5cat_20251030_220000/
├── best_model.pt              # Lowest val loss (may not have best 5CAT)
├── best_5cat_model.pt         # Best 5CAT metrics ← DEPLOY THIS ONE
├── checkpoint_epoch5.pt       # Checkpoint at epoch 5
├── checkpoint_epoch10.pt
├── checkpoint_epoch15.pt
├── checkpoint_epoch20.pt
├── training_history.json      # All metrics by epoch
└── training.log              # Full training output
```

**Use `best_5cat_model.pt`** for production (optimized for forward prediction)

---

## 📚 Documentation

### Created in This Investigation

1. **Training Scripts**:
   - `scripts/train_with_5cat_validation.sh` - Train single model
   - `scripts/retrain_all_production_models.sh` - Train all models

2. **Test Tools**:
   - `tools/tests/test_5to1_alignment.py` - 5CAT validation
   - `tools/tests/diagnose_data_direction.py` - Data quality check

3. **Reports**:
   - `artifacts/lvm/5CAT_PRODUCTION_MODELS_REPORT.md` - Test results
   - `artifacts/lvm/BACKWARD_BIAS_ROOT_CAUSE_REPORT.md` - Full investigation
   - `RETRAINING_QUICK_START.md` - This file

4. **Updated Documentation**:
   - `CLAUDE.md` - Added data quality requirements and 5CAT standards

---

## ✅ Success Criteria

Model is ready for production when:

- ✅ Val Margin: **≥ +0.10** (positive!)
- ✅ OOD Margin: **≥ +0.08** (positive!)
- ✅ Val Rollout: **≥ 0.50**
- ✅ Pass at least 3/5 5CAT gates
- ✅ Training history shows no backward bias alerts

---

## 🎯 Next Steps After Retraining

1. **Document Results**
   - Update `docs/LVM_DATA_MAP.md` with new model metrics
   - Add 5CAT results to model metadata
   - Compare to old 340k models

2. **Performance Testing**
   - Run chat interface tests
   - Measure inference latency
   - Compare text generation quality

3. **Production Deployment**
   - Update all 4 model symlinks (9002-9004, 9006)
   - Restart services
   - Monitor for issues

4. **Establish Standards**
   - Make 5CAT mandatory for all future models
   - Require data quality checks before training
   - Document in team wiki

---

**Ready to start?**

```bash
# Train your first model (Transformer recommended)
./scripts/train_with_5cat_validation.sh transformer
```

Good luck! 🚀
