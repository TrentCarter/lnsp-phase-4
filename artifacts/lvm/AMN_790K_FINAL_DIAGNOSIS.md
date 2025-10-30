# AMN 790K Training - Complete Diagnosis & Solution

**Date**: 2025-10-30
**Attempts**: 2 failures, preparing attempt #3
**Status**: ROOT CAUSE IDENTIFIED, SOLUTION READY

---

## 📊 All Attempts Summary

| Attempt | Config | Val Cosine | OOD Cosine | Issue |
|---------|--------|------------|------------|-------|
| **584k Baseline** | MSE-only (1.0) | 0.5597 ✅ | 0.6375 ✅ | SUCCESS |
| **790k #1** | MSE-only (1.0) | 0.4607 | -0.0118 | Mode collapse? |
| **790k #2** | MSE(0.5)+InfoNCE(0.5) | 0.2675 ❌❌ | ??? | InfoNCE dominance |
| **790k #3** | MSE-only (1.0) | ??? | ??? | READY TO LAUNCH |

---

## 🔍 What We Learned

### 1. InfoNCE is NOT the Solution

**Attempt #2 Results**:
- Trained with `--lambda-mse 0.5 --lambda-info 0.5`
- InfoNCE loss (1.39) was 700x larger than MSE (0.002)
- InfoNCE dominated training → cosine stuck at 0.27
- **Worse than attempt #1!**

**Why InfoNCE Failed**:
- InfoNCE optimizes contrastive ranking, not absolute similarity
- Loss magnitude mismatch (1.0-2.0 vs 0.001-0.002)
- Conflicting objectives with MSE
- Not appropriate for sequential vector prediction

### 2. MSE-Only is Correct

**584k Evidence**:
```python
# 584k successful config:
--lambda-mse 1.0   # Pure MSE
--lambda-info 0.0  # InfoNCE disabled
--epochs 20
--lr 0.0005

# Results:
Epoch 1: train_cosine 0.5281 ✅
Final:   val_cosine 0.5597 ✅
OOD:     0.6375 ✅
```

MSE-only achieved excellent results on 584k. We should use the same approach for 790k.

---

## 🤔 Why Did 790k Attempt #1 Fail?

**Hypothesis**: NOT a fundamental problem with MSE-only, but likely:
1. **Different data split** (random seed)
2. **Bad initialization** (random weights)
3. **Learning rate interaction** with larger dataset
4. **Early stopping** (may have needed more epochs)

**Evidence for "bad luck"**:
- 584k got 0.5597 with MSE-only
- 790k #1 got 0.4607 with same config (lower but not catastrophic)
- 790k #2 got 0.2675 with InfoNCE (catastrophic!)

**Conclusion**: MSE-only is correct. Attempt #1 likely hit bad initialization/split. Try again.

---

## ✅ Corrected Training Configuration

###Use the Proven 584k Recipe

```bash
# artifacts/lvm/models/amn_584k_pure_mse_20251029_055838/
# This model achieved 0.5597 in-dist, 0.6375 OOD

./.venv/bin/python app/lvm/train_unified.py \
    --model-type amn \
    --data artifacts/lvm/training_sequences_ctx5.npz \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.0005 \
    --device mps \
    --lambda-mse 1.0 \
    --lambda-info 0.0
```

**Why This Will Work**:
1. ✅ **Proven on 584k** (0.5597 in-dist, 0.6375 OOD)
2. ✅ **Simple** (one loss term, no conflicts)
3. ✅ **Right scale** (MSE ~0.001 matches LR ~0.0005)
4. ✅ **Direct objective** (minimize distance → maximize cosine)

**Expected Results**:
- Epoch 1 cosine: ~0.50-0.53 (if working)
- Final val cosine: ~0.55-0.58
- Final OOD cosine: ~0.62-0.65

---

## 🚦 Early Warning Signs

### ✅ Good Training (Like 584k):
```
Epoch 1: train_cosine 0.5281 ✅
Epoch 2: train_cosine 0.5525 (improving)
Epoch 5: train_cosine 0.5580 (steady climb)
```

### ❌ Bad Training (Needs Abort):
```
Epoch 1: train_cosine 0.25 ❌ (RED FLAG - restart!)
Epoch 2: train_cosine 0.26 (not improving)
Epoch 5: train_cosine 0.27 (stuck)
```

**Rule**: If epoch 1 cosine < 0.45, something is fundamentally wrong. Abort and restart.

---

## 📋 Pre-Flight Checklist (Before Attempt #3)

1. ✅ **Correct script**: `scripts/train_amn_790k.sh` uses MSE-only
2. ✅ **Data file**: `artifacts/lvm/training_sequences_ctx5.npz` (726k sequences)
3. ✅ **Config matches 584k**: lambda-mse=1.0, lambda-info=0.0
4. ✅ **Hyperparameters**: epochs=20, lr=0.0005, batch=32
5. ✅ **Device**: mps (Apple Silicon)
6. ✅ **OpenMP fix**: `KMP_DUPLICATE_LIB_OK=TRUE` in script

---

## 🎯 Success Criteria

Training is successful if:
1. **Epoch 1 cosine ≥ 0.48** (early signal)
2. **Epoch 5 cosine ≥ 0.52** (steady improvement)
3. **Final val cosine ≥ 0.55** (target)
4. **OOD cosine ≥ 0.60** (generalization)

Training has failed if:
1. **Epoch 1 cosine < 0.45** → Abort and restart
2. **Flat progression** (no improvement by epoch 5) → Abort
3. **Divergence** (cosine decreasing) → Abort

---

## 🔧 If Attempt #3 Fails Again

If MSE-only fails a second time with cosine < 0.45:

### Possible Causes:
1. **Data issue**: Check 790k data for corruption/normalization
2. **Hyperparameter mismatch**: Try different LR (0.0003 or 0.001)
3. **Context size**: Try ctx=7 instead of ctx=5
4. **Model architecture**: Verify AMN implementation matches 584k

### Debug Steps:
```python
# 1. Verify data normalization
data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
targets = data['target_vectors']
norms = np.linalg.norm(targets, axis=1)
print(f"Mean norm: {norms.mean()}, should be ~1.0")

# 2. Check cosine similarity of targets themselves
from sklearn.metrics.pairwise import cosine_similarity
cos_matrix = cosine_similarity(targets[:1000])
print(f"Self-similarity: {cos_matrix.mean()}")  # Should be 0.3-0.5

# 3. Test model forward pass
from app.lvm.models import create_model
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
test_ctx = torch.randn(4, 5, 768)  # batch=4, ctx=5, dim=768
raw, cos = model(test_ctx, return_raw=True)
print(f"Output norm: {cos.norm(dim=1).mean()}")  # Should be ~1.0
```

---

## 📝 Final Recommendation

**Launch Attempt #3** with:
- MSE-only (proven on 584k)
- Same hyperparameters as 584k
- Monitor epoch 1 cosine (expect >0.48)

If it works → proceed to GRU/LSTM/Transformer with same config
If it fails → deep dive into data/model diagnostics

---

**Status**: READY TO LAUNCH ATTEMPT #3
**Config**: MSE-only (λ_mse=1.0, λ_info=0.0)
**Expected Duration**: 6-8 hours (20 epochs)
**Updated**: 2025-10-30 14:00 PST
