# P2 Smoke Test - Execution Plan

**Date**: 2025-10-30
**Status**: ✅ READY TO LAUNCH
**Est. Time**: 90 minutes (5 epochs)

---

## 🎯 Objective

Verify that the filtered 277k dataset's quality improvement translates to better OOD performance by:
1. Warm-starting from proven 584k baseline
2. Quick 5-epoch fine-tune on filtered data
3. Evaluating against strict gates

---

## 🚀 Step 1: Launch P2 Smoke Test

### Command (Run in External Terminal)

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

./scripts/train_amn_p2_smoke_test.sh
```

### Configuration

```
Data:           artifacts/lvm/training_sequences_ctx5_filtered.npz (277k sequences)
Warm-start:     artifacts/lvm/models/amn_584k_pure_mse_20251029_055838/best_model.pt
Epochs:         5
Batch size:     64
Learning rate:  0.0005
Device:         MPS
Loss:           MSE-only (lambda_mse=1.0, lambda_info=0.0)
```

### Expected Runtime

- ~18 minutes per epoch (277k sequences, batch=64)
- Total: ~90 minutes for 5 epochs

---

## 📊 Step 2: Monitor Training (Watch for Gates)

### Epoch 1 Gate: val_cosine >= 0.50

**Baseline comparison**:
- 790k unfiltered epoch 1: 0.4457
- Target: ≥ 0.50

**Interpretation**:
- ✅ `>= 0.50`: Quality filtering working! Continue.
- ❌ `< 0.50`: Abort. Filtered data not better than unfiltered.

### Epoch 3 Checkpoint

Watch for: val_cosine trend
- ✅ `>= 0.52`: Excellent progress
- ⚠️  `0.50-0.52`: Acceptable, but monitor
- ❌ `< 0.50`: Concerning, may need ctx=7

### Epoch 5 Gate: val_cosine >= 0.53

**Target**: Approaching 584k baseline (0.5605)
- ✅ `>= 0.53`: GREEN - proceed to OOD eval
- ⚠️  `0.50-0.53`: AMBER - OOD eval critical
- ❌ `< 0.50`: RED - filtering insufficient

---

## 🔍 Step 3: Post-Training OOD Evaluation

### Once training completes, run OOD eval:

```bash
MODEL_DIR=$(ls -td artifacts/lvm/models/amn_filtered_smoke_* | head -1)

./.venv/bin/python tools/eval_model_ood.py \
  --model $MODEL_DIR/best_model.pt \
  --model-type amn \
  --device mps
```

### OOD Gates

**Context**:
- 584k baseline OOD: 0.6375
- 790k unfiltered OOD: 0.0211 (catastrophic)

**Gates**:
- ✅ **GREEN** (>= 0.58): Filtered dataset works! Proceed to Step 4 (full training)
- ⚠️  **AMBER** (0.50-0.57): Partial recovery. Consider:
  - Option A: Proceed with ctx=7 in parallel
  - Option B: Full 20-epoch training on filtered ctx=5, evaluate
- ❌ **RED** (< 0.50): Filtering insufficient. Stick with 584k baseline.

---

## 📋 Step 4: Decision Tree

### If GREEN (OOD >= 0.58):

**Proceed to full 20-epoch production training**:
```bash
./scripts/train_amn_filtered_production.sh
```

**Expected results**:
- Final val: 0.56-0.58
- Final OOD: ≥0.62 (close to 584k baseline)

**Timeline**: ~6 hours (20 epochs on 277k)

---

### If AMBER (OOD 0.50-0.57):

**Two-track approach**:

**Track A**: Full training on filtered ctx=5 (proceed as GREEN)
**Track B**: Parallel ctx=7 salvage on low-coherence remainder

```bash
# Track B: Create ctx=7 sequences from filtered-out data
python tools/build_sequences_ctx7.py \
  --mask artifacts/lvm/quality_mask_790k.npy \
  --invert \
  --out artifacts/lvm/training_sequences_ctx7_lowcoh.npz

# Quick 4-epoch probe
./scripts/train_amn_ctx7_salvage.sh
```

**Decision**:
- If ctx=7 OOD >= 0.50 → hybrid fine-tune (70% ctx=5 + 30% ctx=7)
- If ctx=7 OOD < 0.50 → stick with filtered ctx=5 only

---

### If RED (OOD < 0.50):

**Stick with 584k baseline**:
- Proven performance: 0.5597 in-dist, 0.6375 OOD
- 543k high-quality sequences
- No risk of regression

**Alternative**: Investigate ctx=7 on full 790k (last resort)

---

## 📈 Success Criteria Summary

### P2 Smoke Test (This run)

| Metric       | RED      | AMBER    | GREEN    |
|--------------|----------|----------|----------|
| Ep1 val_cos  | < 0.50   | 0.50-0.52| >= 0.52  |
| Ep5 val_cos  | < 0.50   | 0.50-0.53| >= 0.53  |
| OOD cosine   | < 0.50   | 0.50-0.57| >= 0.58  |

### Full Production (If GREEN)

| Metric       | Target   | Baseline (584k) |
|--------------|----------|-----------------|
| Final val    | >= 0.56  | 0.5605          |
| Final OOD    | >= 0.62  | 0.6375          |
| OOD boost    | +0.05    | +0.08           |

---

## 🛠️ Troubleshooting

### If training crashes:

1. **Check macOS OpenMP**: Script sets `KMP_DUPLICATE_LIB_OK=TRUE`
2. **Check memory**: 277k @ batch=64 should be fine on MPS
3. **Check data**: Verify `artifacts/lvm/training_sequences_ctx5_filtered.npz` exists
4. **Check checkpoint**: Verify 584k checkpoint exists and loads

### If val_cosine doesn't improve:

- **Cause**: Warm-start may need lower LR
- **Fix**: Reduce LR to 0.00025 (half of current)
- **Command**: Edit `scripts/train_amn_p2_smoke_test.sh`, change `LEARNING_RATE=0.00025`

### If OOD evaluation fails:

- **Cause**: OOD test data may be wrong
- **Fix**: Use the 584k OOD test set initially
- **Command**:
  ```bash
  python tools/eval_model_ood.py \
    --model $MODEL_DIR/best_model.pt \
    --model-type amn \
    --ood-data artifacts/lvm/wikipedia_ood_test_ctx5.npz
  ```

---

## 🔬 What We're Testing

**Hypothesis**: Filtering 790k dataset to remove low-coherence sequences (0.3367 → 0.4250 mean coherence) will recover OOD generalization.

**Evidence So Far**:
- ✅ P0: Space alignment verified (encoder, normalization, FAISS all correct)
- ✅ P1: Root cause identified (low-coherence dilution, -30.5% vs 584k)
- ✅ P1: Filtering created high-quality subset (277k, coherence 0.4250, only -12.2% vs 584k)

**What This Test Proves**:
- If GREEN: Dataset quality was the issue, filtering solves it
- If AMBER: Partial solution, ctx=7 or hybrid may help
- If RED: More fundamental issue (architecture, OOD test data, etc.)

---

## 📁 Files Generated

- **Training log**: `artifacts/lvm/models/amn_filtered_smoke_TIMESTAMP/training.log`
- **Best checkpoint**: `artifacts/lvm/models/amn_filtered_smoke_TIMESTAMP/best_model.pt`
- **Training history**: `artifacts/lvm/models/amn_filtered_smoke_TIMESTAMP/training_history.json`
- **OOD eval results**: Will be printed to console

---

## ⏱️ Timeline

- **T+0**: Start training
- **T+18min**: Epoch 1 complete → Check gate (val_cosine >= 0.50)
- **T+54min**: Epoch 3 complete → Monitor trend
- **T+90min**: Epoch 5 complete → Check gate (val_cosine >= 0.53)
- **T+95min**: OOD evaluation → Final decision (GREEN/AMBER/RED)

---

**Status**: ✅ READY TO LAUNCH
**Command**: `./scripts/train_amn_p2_smoke_test.sh`
**Next**: Run in external terminal, monitor gates, report back results
