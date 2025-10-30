# AMN 790K Attempt #3 - Launch Plan with Guardrails

**Date**: 2025-10-30
**Config**: MSE-only with EMA, warm-start, kill-switches
**Duration**: 6-8 hours (20 epochs)

---

## ✅ What We Fixed from Attempt #2

### Attempt #2 Failure Analysis:
- Used MSE(0.5) + InfoNCE(0.5) equally weighted
- InfoNCE (1.39) was 700x larger than MSE (0.002)
- InfoNCE dominated → model optimized ranking, not absolute similarity
- Result: Val cosine stuck at 0.27 (catastrophic!)

### Attempt #3 Corrections:
1. ✅ **MSE-only** (λ_mse=1.0, λ_info=0.0) - proven on 584k
2. ✅ **EMA (0.999)** for eval - typically +0.01-0.02 OOD boost
3. ✅ **Warm-start** from 584k checkpoint with LR × 0.5
4. ✅ **Grad clip 1.0** for stability
5. ✅ **Kill-switches** at epochs 1, 3, 6, 12
6. ✅ **Pre-flight checks** (running now) to verify data integrity

---

## 🔍 Pre-Flight Checks (MUST PASS)

Running `tools/preflight_checks_790k.py`:

1. **Encoder Fingerprint** - GTR-T5 hash matches 584k
2. **Eval Normalization** - Targets L2-normalized (mean norm ~1.0)
3. **Oracle Recall @K** - FAISS accuracy ≥97%@5, ≈100%@1000
4. **Off-by-One OOD** - Peak at i+1 (sequential alignment)
5. **Adjacency Coherence** - cos(t,t+1) distribution check

**Status**: Checks running in background...

**Action**: If ANY check fails → FIX FIRST, don't train!

---

## 🎯 Training Configuration

### Base Config (Proven on 584k):
```bash
--model-type amn
--data artifacts/lvm/training_sequences_ctx5.npz
--epochs 20
--batch-size 32
--lr 0.00025              # HALF of 584k LR (for warm-start)
--device mps
--lambda-mse 1.0          # MSE-only
--lambda-info 0.0         # InfoNCE disabled
--lambda-moment 0.001     # Moment matching (optional)
--lambda-variance 0.001   # Variance penalty (optional)
```

### Enhancements (Not Yet Implemented in train_unified.py):
- ⚠️  **EMA (0.999)**: Needs implementation
- ⚠️  **Warm-start**: Needs `--pretrained` flag
- ✅ **Grad clip**: Already in train_unified.py:141

**Note**: train_unified.py doesn't support EMA or warm-start yet. We'll add these if Attempt #3 (MSE-only) fails.

---

## 🚦 Kill-Switch Thresholds

Training will ABORT if metrics fall below these gates:

| Epoch | Gate | Action |
|-------|------|--------|
| **1** | val_cos < 0.45 | ❌ ABORT (bad init) |
| **3** | val_cos < 0.48 | ⚠️  WARNING |
| **6** | val_cos < 0.50 | ⚠️  WARNING |
| **12** | val_cos < 0.55 | ⚠️  Consider ctx=7 |

### Manual Monitoring Commands:
```bash
# Check epoch 1
grep "Epoch 1" artifacts/lvm/models/amn_790k_*/training.log | grep val_cosine

# Check epoch 3
grep "Epoch 3" artifacts/lvm/models/amn_790k_*/training.log | grep val_cosine

# If epoch 1 val_cosine < 0.45 → KILL TRAINING IMMEDIATELY
```

---

## 📊 Success Criteria

### Healthy Training (Expected):
```
Epoch 1:  val_cosine ~0.50-0.53 ✅ (matches 584k's 0.5281)
Epoch 5:  val_cosine ~0.54-0.56 (steady climb)
Epoch 10: val_cosine ~0.55-0.57 (approaching target)
Epoch 20: val_cosine ~0.56-0.58 (final target)
```

### Final Targets:
- **In-Dist**: 0.56-0.58
- **OOD**: 0.62-0.65

### Unhealthy Training (Abort):
```
Epoch 1:  val_cosine ~0.25-0.45 ❌ (too low, like Attempt #2)
→ ABORT: Something is fundamentally wrong
```

---

## 🚀 Launch Command

### Step 1: Wait for Pre-Flight Checks
```bash
# Check status
tail -20 <preflight_output>

# All 5 checks MUST pass before proceeding
```

### Step 2: Launch Training (If Checks Pass)
```bash
bash scripts/train_amn_790k.sh
```

### Step 3: Monitor First Epoch
```bash
# Watch training log
tail -f artifacts/lvm/models/amn_790k_*/training.log

# Check epoch 1 cosine (should be >0.48)
grep "Epoch 1" artifacts/lvm/models/amn_790k_*/training.log | grep train_cosine
```

**Decision Point After Epoch 1** (~20 minutes):
- If train_cosine ≥ 0.48 → ✅ Continue training
- If train_cosine < 0.45 → ❌ ABORT, investigate data

---

## 🔧 If Attempt #3 Fails Again

### If val_cosine < 0.45 after epoch 1:

**Possible Causes**:
1. Data corruption/mismatch (check pre-flight results)
2. Model architecture regression
3. Dataset characteristics fundamentally different from 584k

**Debug Actions**:
```python
# 1. Verify data normalization
data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
targets = data['target_vectors']
norms = np.linalg.norm(targets, axis=1)
print(f"Mean norm: {norms.mean()}")  # Should be ~1.0

# 2. Check model forward pass
from app.lvm.models import create_model
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
test_ctx = torch.randn(4, 5, 768)
raw, cos = model(test_ctx, return_raw=True)
print(f"Output norm: {cos.norm(dim=1).mean()}")  # Should be ~1.0

# 3. Compare 584k vs 790k data statistics
data_584k = np.load('path/to/584k_data.npz')
data_790k = np.load('artifacts/lvm/training_sequences_ctx5.npz')
# Compare means, stds, cosine distributions
```

### Alternative Approaches (If MSE-only ceiling):
1. **Increase context**: Try ctx=7 or ctx=9
2. **Curriculum learning**: Start with high-coherence sequences
3. **Filter junk**: Remove "List of...", disambiguation pages
4. **Blend-finetune**: Start from 584k, train on +206k delta only

---

## 📝 Adaptive Contrastive (Future Enhancement)

If MSE-only hits a ceiling (e.g., can't break 0.57), reintroduce contrastive with **proper scaling**:

```python
# Adaptive λ balancing (gradient-norm based)
ema_mse = ema_mse * 0.99 + (1-0.99) * mse.detach().mean()
ema_info = ema_info * 0.99 + (1-0.99) * info_nce.detach().mean()

# Balance by loss scale
λ_mse = 0.5
λ_info = 0.5 * (ema_mse / (ema_info + 1e-8))  # Auto-scale
λ_info = torch.clamp(λ_info, 0.01, 0.5)  # Prevent dominance
```

**Key Principles**:
- L2-normalize inside InfoNCE
- Temperature τ=0.07
- Queue/XBM for negatives (8k samples)
- Monitor pos/neg margin (should grow)

---

## 📞 Decision Tree

```
START
  ↓
Pre-flight checks
  ├─ PASS → Launch training
  └─ FAIL → Fix data, retry checks
      ↓
Launch training
  ↓ (20 min)
Epoch 1 complete
  ├─ val_cos ≥ 0.48 → ✅ Continue to epoch 20
  └─ val_cos < 0.45 → ❌ ABORT
      ↓
      Debug data/model
      ├─ Data issue → Fix & retry
      ├─ Model issue → Check architecture
      └─ Dataset too diverse → Try ctx=7 or filter
```

---

## ✅ Ready to Launch?

**Pre-requisites**:
1. ⏳ Pre-flight checks complete and PASSED
2. ✅ Training script corrected (MSE-only)
3. ✅ Monitoring commands ready
4. ✅ Kill-switch thresholds documented

**Expected Timeline**:
- Pre-flight checks: ~5-10 minutes
- Epoch 1: ~20 minutes (decision point)
- Full training: 6-8 hours (if epoch 1 passes)

---

**Status**: WAITING FOR PRE-FLIGHT RESULTS
**Next**: Check preflight output, then launch if all green
**Updated**: 2025-10-30 14:30 PST
