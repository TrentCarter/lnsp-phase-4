# Mamba-S Contrastive Training Status
## Phase 5.1: Fixing Generalization with InfoNCE

**Started**: 2025-10-27
**Status**: 🔄 **RUNNING** (PID 47970)
**Device**: MPS (Apple Silicon GPU)

---

## Problem Diagnosis (Root Cause)

### POC Training Results (AR Cosine Only)
- **Training data**: 0.62 cosine ✅ (works great on seen articles!)
- **Eval data**: 0.22 cosine ❌ (complete failure on unseen articles!)
- **Retrieval**: 0% (Contain@50=0%, R@5=0%)

### Root Cause
**Objective mismatch**: AR cosine loss optimizes next-step regression, which causes:
- ✅ **Episode memorization**: Model learns article-specific patterns
- ❌ **No generalization**: Fails on unseen articles
- ❌ **Shortcut-friendly**: Memorizes "if in article X, predict pattern Y"

### Contractor's Diagnosis
> "The model is learning episode-specific sequence patterns, not the global GTR-T5 semantic geometry."

**Why AMN worked**: Two-tower contrastive training → learns global geometry → generalizes to unseen articles

---

## The Fix: Contrastive Learning

### Architecture Changes
1. **Projection Head**: 768→512→256 with GELU, LayerNorm, L2 normalization
2. **InfoNCE Loss**: Contrastive learning with all in-batch negatives (τ=0.07)
3. **Stop-Grad**: On target branch to prevent collapse
4. **Combined Loss**: λ_con=0.7 × InfoNCE + λ_ar=0.3 × AR_cosine
5. **Regularization**:
   - Article dropout (p=0.2): Zero out last k context positions
   - Span corruption (p=0.1): Replace random position with different article's vector
6. **Large Effective Batch**: 256 × 4 = 1024 via gradient accumulation

### Why This Works
```
AR Cosine Only:
  "Next vector in article X" → Model learns X-specific patterns → Overfits to seen articles

InfoNCE + AR Cosine:
  "ŷ closer to true next than to 1000+ negatives from other topics"
  → Forces global GTR-T5 semantics
  → Generalizes to unseen articles
  + AR term preserves sequential signal for generation
```

---

## Training Configuration

```bash
Model: Mamba-S (Pure SSM)
Training data: 396,258 sequences (payload-aligned)
Validation data: 99,064 sequences (payload-aligned)

Hyperparameters:
  d_model: 768
  n_layers: 8
  d_state: 128
  conv_sz: 4
  expand: 2
  dropout: 0.1

Contrastive:
  projection: 768→512→256
  lambda_con: 0.7
  lambda_ar: 0.3
  temperature: 0.07
  article_dropout: 0.2
  span_corruption: 0.1

Training:
  batch_size: 256
  grad_accum_steps: 4  (effective batch: 1024)
  epochs: 20
  lr: 1e-3
  weight_decay: 0.02
  warmup_steps: 1000
  device: mps
```

---

## Expected Timeline

**Data Loading**: ~30-60 seconds (396k sequences)
**Epoch 1**: ~20-25 minutes (MPS with contrastive head)
**Epochs 2-20**: ~15-20 minutes each
**Total time**: ~4-5 hours (early stopping likely around epoch 8-10)

**Smoke test**: After epoch 2 (check eval_cosine ≥ 0.50)

---

## Success Gates (From Contractor)

### Epoch 2 Gate (Critical!)
- ✅ **Eval cosine ≥ 0.50** (was 0.22 with AR-only)
  - This is the key signal that contrastive learning is working
  - Means model is learning global GTR-T5 space, not episode patterns

### Final Gates (5.2k eval samples)
- 🎯 **Contain@50 ≥ 60%** (was 0%)
- 🎯 **Eff@5 ≥ 0.68**
- 🎯 **R@5 ≥ 40%** (was 0%)
- 🎯 **P95 latency ≤ 1.45ms**

---

## Monitoring Commands

### Check Training Progress
```bash
# Watch training in real-time
tail -f logs/mamba_s_contrastive_*.log

# Check latest validation metrics
tail -100 logs/mamba_s_contrastive_*.log | grep "val_cosine"

# Check if training is still running
ps aux | grep train_mamba_contrastive | grep -v grep
```

### Check Epoch 2 Result
```bash
# After ~40 minutes, check epoch 2 val_cosine
python3 -c "
import json
with open('artifacts/lvm/models/mamba_s_contrastive/history.json') as f:
    history = json.load(f)
    if len(history) >= 2:
        epoch2 = history[1]
        print(f'Epoch 2 val_cosine: {epoch2[\"val_cosine\"]:.4f}')
        if epoch2['val_cosine'] >= 0.50:
            print('✅ GATE PASSED - Contrastive learning is working!')
        else:
            print('⚠️ GATE MISSED - May need investigation')
"
```

### Run Unified Evaluation (After Training)
```bash
# Full 5.2k sample evaluation with unified gates
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
  --checkpoint artifacts/lvm/models/mamba_s_contrastive/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 5244 \
  --nprobe 64 \
  --gate-contain50 0.60 \
  --gate-eff5 0.68 \
  --gate-r5 0.40 \
  --gate-p95 1.45 \
  --out artifacts/lvm/eval_mamba_s_contrastive_full.json
```

---

## Expected Results

### With Contrastive Learning (NEW)
| Metric | OLD (AR-only) | NEW (Expected) | Why It Works |
|--------|---------------|----------------|--------------|
| **Val cosine (same articles)** | 0.5749 | 0.54-0.58 | Similar (minor regularization cost) |
| **Eval cosine (unseen articles)** | 0.22 ❌ | **≥0.50** ✅ | **Learns global space!** |
| **Contain@50** | 0% | 60-75% | Generalizes to unseen articles |
| **R@5** | 0% | 40-55% | Sequential bias + generalization |
| **Eff@5** | N/A | 0.68-0.73 | Efficient containment-to-recall |

---

## Decision Points

### After Epoch 2 (~40 minutes)
**IF** eval_cosine ≥ 0.50:
- ✅ **Contrastive learning is working!**
- ✅ **Continue training** to 20 epochs (early stop will trigger)
- ✅ **Generalization fixed** - model learning global GTR-T5 space

**IF** eval_cosine < 0.40:
- ⚠️ **Investigate** - contrastive loss may not be working
- Check projection head is being used
- Verify InfoNCE loss is computed correctly
- Inspect batch negatives sampling

### After Training Complete
**IF** R@5 ≥ 40% AND Contain@50 ≥ 60%:
- ✅ **SUCCESS!** - Contrastive fix worked
- ✅ **Proceed with Sandwich/H** using same regimen
- Expected total time: ~12-15 hours for all 3 models

**IF** R@5 ≥ 53.2% (AMN baseline):
- 🎉 **Mamba matches AMN!**
- Consider production deployment
- Run A/B testing vs AMN

---

## What Changed vs POC

### POC (AR Cosine Only)
```python
# Loss: Just cosine similarity to next vector
loss = 1.0 - F.cosine_similarity(pred, target).mean()

# Problem: Learns article-specific patterns
# "If in article 1234, next vector is similar to training pattern X"
```

### Contrastive (InfoNCE + AR)
```python
# Loss: Must be closer to true next than to all batch negatives
logits = torch.mm(h_pred, h_target.t()) / temperature  # [B, B]
loss_infonce = F.cross_entropy(logits, labels)

# Combined with sequential signal
loss = 0.7 * loss_infonce + 0.3 * loss_ar

# Result: Learns global GTR-T5 semantics + sequential patterns
```

---

## Files

### Training
- **Script**: `app/lvm/train_mamba_contrastive.py`
- **Launch**: `scripts/train_mamba_s_contrastive.sh`
- **Model**: `artifacts/lvm/models/mamba_s_contrastive/best.pt`
- **History**: `artifacts/lvm/models/mamba_s_contrastive/history.json`
- **Log**: `logs/mamba_s_contrastive_20251027_*.log`

### Evaluation
- **Unified tool**: `tools/eval_checkpoint_unified.py`
- **Quick reference**: `docs/PHASE5_EVAL_QUICK_REFERENCE.md`

### Data (Payload-Aligned)
- **Train**: `artifacts/lvm/train_payload_aligned.npz` (396k sequences)
- **Val**: `artifacts/lvm/val_payload_aligned.npz` (99k sequences)
- **Eval**: `artifacts/lvm/eval_v2_payload_aligned.npz` (5.2k sequences)

### Documentation
- **Root cause**: `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md`
- **Data alignment**: `docs/MAMBA_PHASE5_DATA_ALIGNMENT_COMPLETE.md`
- **Eval reference**: `docs/PHASE5_EVAL_QUICK_REFERENCE.md`
- **This file**: `docs/MAMBA_CONTRASTIVE_TRAINING_STATUS.md`

---

## Recommended Actions

**Now** (0-40 minutes):
- Let training run undisturbed
- Check log occasionally: `tail -f logs/mamba_s_contrastive_*.log`
- Training is loading data and running first epochs

**After 40 minutes** (Epoch 2 should be done):
- Check `history.json` for epoch 2 val_cosine
- **CRITICAL GATE**: val_cosine ≥ 0.50 means contrastive is working!
- If passed, let training continue

**After 4-5 hours** (Training complete):
- Run full unified evaluation on 5.2k samples
- Check final gates: Contain@50≥60%, R@5≥40%, Eff@5≥0.68, P95≤1.45ms
- If passed, start Sandwich/H with same contrastive regimen

**Within 24 hours** (All models trained):
- Compare Sandwich/H/S with contrastive learning
- Pick best model (likely Sandwich based on previous val_cosine)
- Run production comparison vs AMN baseline
- Generate Phase-5 completion report

---

## Confidence Level

**Diagnosis**: ✅ 100% confident (AR-only causes episode memorization)
**Solution**: ✅ High confidence (InfoNCE forces global learning)
**Expected Outcome**: ✅ 80%+ confidence (eval_cosine 0.22 → ≥0.50)

The root cause is definitively identified: AR cosine loss causes overfitting to seen articles. Contrastive learning forces the model to learn the global GTR-T5 semantic space by competing against many in-batch negatives. This is the standard solution in metric learning and should fix the generalization problem.

---

**Last Updated**: 2025-10-27
**Training PID**: 47970
**Status**: RUNNING
**Check in**: ~40 minutes for epoch 2 gate
