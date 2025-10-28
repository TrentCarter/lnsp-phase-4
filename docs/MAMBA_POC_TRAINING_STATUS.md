# Mamba-S POC Training Status
## Payload-Aligned Data Retrain

**Started**: 2025-10-26 21:20:57
**Status**: 🔄 **RUNNING** (PID 29676)
**Device**: MPS (Apple Silicon GPU)

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
  epochs: 20
  batch_size: 256
  lr: 1e-3
  device: mps
```

---

## Expected Timeline

**Data Loading**: ~30-60 seconds (396k sequences)
**Epoch 1**: ~15-20 minutes (first epoch, model initialization)
**Epochs 2-20**: ~10-15 minutes each
**Total POC time**: ~3-4 hours (with early stopping likely around epoch 10-12)

**Smoke test checkpoints**: After epochs 2, 4, 8 (automatic via best.pt updates)

---

## Success Gates (POC)

### Epoch 4 Gates (1k eval samples)
- ✅ Contain@50 ≥ 55%
- ✅ Eff@5 ≥ 0.65
- ✅ R@5 ≥ 35%
- ✅ P95 latency ≤ 2.0ms

### Final Gates (5.2k eval samples)
- 🎯 Contain@50 ≥ 60%
- 🎯 Eff@5 ≥ 0.68
- 🎯 R@5 ≥ 40%
- 🎯 P95 latency ≤ 1.45ms

---

## Monitoring Commands

### Check Training Progress
```bash
# Watch live training log
tail -f logs/mamba_s_poc_20251026_212057.log

# Check latest validation metrics
tail -50 logs/mamba_s_poc_20251026_212057.log | grep "val_cosine"

# Check if training is still running
ps aux | grep "train_mamba_unified" | grep mamba_s_poc
```

### Run Smoke Test (After Epoch 4)
```bash
# Wait for epoch 4 checkpoint
ls -lh artifacts/lvm/models/mamba_s_poc/best.pt

# Run 1k sample smoke test
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_smoke.py \
  --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 1000 \
  --epoch 4 \
  --out artifacts/lvm/smoke_test_epoch4.json
```

### Run Full Evaluation (After Training Complete)
```bash
# Full 5.2k sample evaluation
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/smoke_test_aligned_eval.py \
  --model artifacts/lvm/models/mamba_s_poc/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 5244 \
  --out artifacts/lvm/eval_mamba_s_poc_full.json
```

---

## Expected Results

### Training Metrics (Similar to Before)
- **Val cosine**: 0.54-0.58 (model quality)
- **Training loss**: Decreasing steadily
- **Early stopping**: Likely around epoch 10-12

### Retrieval Metrics (FIXED!)
Based on payload alignment, we expect:

| Metric | OLD (Misaligned) | NEW (Expected) | Why It Works |
|--------|------------------|----------------|--------------|
| **Validation cosine** | 0.54-0.58 | 0.54-0.58 | Same (model trains well) |
| **Payload cosine** | 0.25 ❌ | 0.54-0.58 ✅ | **Aligned!** |
| **Contain@50** | 0% | 60-75% | Truth vectors in payload |
| **R@5** | 0% | 40-55% | Predictions match space |
| **Eff@5** | 0% | 0.68-0.73 | Sequential bias works |

---

## What Changed (Why This Will Work)

### Before (0% Retrieval)
```
Training targets: From training_sequences_ctx5.npz
                  ↓ cos=0.14 ❌
Payload vectors: From wikipedia_584k_payload.npy

Result: Model predicts space A, retrieval searches space B → 0%
```

### Now (Expected 60-75% Retrieval)
```
Training targets: From wikipedia_584k_payload.npy
                  ↓ cos=1.00 ✅
Payload vectors: From wikipedia_584k_payload.npy

Result: Model predicts space A, retrieval searches space A → 60-75%!
```

---

## Decision Points

### After Epoch 4 Smoke Test
**IF** Contain@50 ≥ 55% AND Eff@5 ≥ 0.65:
- ✅ **Continue training** to 20 epochs (early stop will trigger around 10-12)
- ✅ **Gates passed** - alignment fix worked!

**IF** Contain@50 < 40%:
- ⚠️ **Investigate** - something unexpected
- Check model is loading correctly
- Verify inference produces normalized vectors
- Run diagnostic tools

### After Training Complete
**IF** R@5 ≥ 40% AND Contain@50 ≥ 60%:
- ✅ **POC SUCCESS** - Proceed with Sandwich/H/XL
- Launch parallel training for other models
- Expected total time: ~16-20 hours for all 4

**IF** R@5 ≥ 53.2% (AMN baseline):
- 🎉 **Production ready** - Mamba matches AMN!
- Consider A/B testing immediately

---

## Next Models (If POC Passes)

### Priority Order
1. **Mamba-Sandwich** (highest val cosine 0.5797 before) - Best performance
2. **Mamba-H** (hybrid 80/20) - Good balance
3. **Mamba-XL** (deeper/wider) - Stretch goal
4. ~~Mamba-GR~~ (skip - poor ROI)

### Parallel Training Strategy
```bash
# Terminal 1: Mamba-Sandwich
./scripts/train_mamba_sandwich_aligned.sh

# Terminal 2: Mamba-H
./scripts/train_mamba_hybrid_aligned.sh

# Terminal 3: Mamba-XL (optional, if resources allow)
./scripts/train_mamba_xl_aligned.sh
```

---

## Current Status

**Process**: Running (PID 29676, 6% CPU on MPS)
**Log**: `logs/mamba_s_poc_20251026_212057.log`
**Checkpoint**: Will appear at `artifacts/lvm/models/mamba_s_poc/best.pt`
**ETA**: ~3-4 hours total (check after 2 hours for epoch 4)

---

## Files Created

### Training Infrastructure
- `scripts/train_mamba_s_poc.sh` - POC training script
- `tools/eval_checkpoint_smoke.py` - Mid-training smoke test
- `tools/build_payload_aligned_training.py` - Data generator

### Training Data (Payload-Aligned!)
- `artifacts/lvm/train_payload_aligned.npz` (396k sequences)
- `artifacts/lvm/val_payload_aligned.npz` (99k sequences)
- `artifacts/lvm/eval_v2_payload_aligned.npz` (5.2k sequences)

### Documentation
- `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md` - Technical deep dive
- `docs/MAMBA_PHASE5_DATA_ALIGNMENT_COMPLETE.md` - Retraining guide
- `docs/MAMBA_POC_TRAINING_STATUS.md` (this file)

---

## Recommended Actions

**Now** (0-2 hours):
- Let training run undisturbed
- Check log occasionally: `tail -f logs/mamba_s_poc_20251026_212057.log`
- Training is loading data and running first epochs

**After 2 hours** (Epoch 4 should be done):
- Run smoke test on 1k samples
- Check gates: Contain@50 ≥ 55%, Eff@5 ≥ 0.65
- If passed, let training continue

**After 3-4 hours** (Training complete):
- Run full evaluation on 5.2k samples
- Check final gates: R@5 ≥ 40%, Contain@50 ≥ 60%
- If passed, start Sandwich/H/XL in parallel

**Within 24 hours** (All models trained):
- Compare all models on full eval set
- Pick best model (likely Sandwich)
- Run production comparison vs AMN baseline
- Generate Phase-5 completion report

---

## Confidence Level

**Data Alignment**: ✅ 100% verified (cos=1.000)
**Training Setup**: ✅ Correct (same hyperparameters as before)
**Expected Outcome**: ✅ High confidence (60-75% Contain@50)

The root cause (data mismatch) has been fixed. Training data now comes from the exact same source as the retrieval payload. Unless there's an unexpected issue, we should see 60-75% Contain@50 and 40-55% R@5.

---

**Last Updated**: 2025-10-26 21:27:00
**Training PID**: 29676
**Status**: RUNNING
