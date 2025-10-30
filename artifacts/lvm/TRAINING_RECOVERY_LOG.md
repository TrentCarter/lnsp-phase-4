# LVM Training Recovery Log - 790k Dataset

**Date**: 2025-10-30
**Objective**: Retrain all 4 LVM models on 790k Wikipedia dataset
**Previous Training**: 584k dataset (543k sequences)
**New Training**: 790k dataset (726k sequences)

---

## Current Status

### Database
- **Concepts**: 790,391 Wikipedia concepts
- **Articles**: 15,192 articles ingested
- **Source**: `wikipedia_500k.jsonl` articles 1-15023

### Data Files
- **Raw sequences**: `artifacts/lvm/training_sequences_ctx5.npz` (726,014 sequences)
- **Split file**: `artifacts/lvm/training_sequences_ctx5_790k_split.npz` (to be created)
- **OOD test**: `artifacts/lvm/wikipedia_ood_test_ctx5.npz` (7,140 sequences)

---

## Training Plan

### Order (Sequential - One at a Time)
1. ✅ **Data Prep**: Split 726k sequences → 90% train (653k) / 10% val (73k)
2. ⏳ **AMN**: Train first (fastest, smallest model - 6-8 hours)
3. ⏳ **LSTM**: Train second (8-10 hours)
4. ⏳ **GRU**: Train third (10-12 hours)
5. ⏳ **Transformer**: Train last (12-16 hours)

### Training Configuration (Per Model)
```bash
# Standard hyperparameters (matching 584k training)
epochs: 20
batch_size: 32
learning_rate: 0.0005
optimizer: AdamW
scheduler: CosineAnnealingLR
loss: MSE
device: mps (Apple Silicon)
```

---

## Progress Tracking

### Step 1: Data Preparation
- **Status**: ✅ COMPLETE
- **Command**: `./.venv/bin/python /tmp/prepare_train_val_split.py`
- **Output**: `artifacts/lvm/training_sequences_ctx5_790k_split.npz`
- **File size**: 11 GB
- **Train sequences**: 653,412 (90%)
- **Val sequences**: 72,602 (10%)
- **Started**: 2025-10-30 14:58:06
- **Completed**: 2025-10-30 15:00:18

### Step 2: AMN Training
- **Status**: PENDING
- **Model dir**: `artifacts/lvm/models/amn_790k_$(date +%Y%m%d_%H%M%S)/`
- **Training script**: `app/lvm/train_unified.py`
- **Command**: (to be determined)
- **Expected duration**: 6-8 hours
- **Started**: (pending)
- **Completed**: (pending)

### Step 3: LSTM Training
- **Status**: PENDING
- **Started**: (pending)
- **Completed**: (pending)

### Step 4: GRU Training
- **Status**: PENDING
- **Started**: (pending)
- **Completed**: (pending)

### Step 5: Transformer Training
- **Status**: PENDING
- **Started**: (pending)
- **Completed**: (pending)

---

## Recovery Instructions

If training crashes or is interrupted:

### 1. Check Last Completed Step
```bash
tail -100 artifacts/lvm/TRAINING_RECOVERY_LOG.md
```

### 2. Resume from Last Checkpoint
```bash
# For AMN (if crashed during training)
./.venv/bin/python app/lvm/train_unified.py \
  --model-type amn \
  --data-path artifacts/lvm/training_sequences_ctx5_790k_split.npz \
  --resume artifacts/lvm/models/amn_790k_*/last_checkpoint.pt \
  --epochs 20 \
  --device mps

# For LSTM, GRU, Transformer - similar pattern
```

### 3. Verify Data Integrity
```bash
# Check split file exists and is valid
python -c "import numpy as np; d = np.load('artifacts/lvm/training_sequences_ctx5_790k_split.npz'); print(f'Train: {len(d[\"train_context_sequences\"]):,}, Val: {len(d[\"val_context_sequences\"]):,}')"
```

---

## Comparison Metrics (584k vs 790k)

### Previous Results (584k Dataset - 543k sequences)

| Model       | In-Dist Cosine | OOD Cosine | Latency (ms) | Params |
|-------------|----------------|------------|--------------|--------|
| AMN         | 0.5597         | 0.6375     | 0.49         | 1.5M   |
| LSTM        | 0.5792         | 0.6046     | 0.56         | 5.1M   |
| GRU         | 0.5920         | 0.6295     | 2.08         | 7.1M   |
| Transformer | 0.5864         | 0.6257     | 2.65         | 17.9M  |

### Expected Improvements (790k Dataset)
- **+35% more training data** (543k → 726k sequences)
- **Expected accuracy gain**: +1-3% cosine similarity
- **Expected OOD improvement**: +2-5% (better generalization)

---

## Notes

- **macOS OpenMP Fix**: `export KMP_DUPLICATE_LIB_OK=TRUE` (prevents "Abort trap: 6" crashes)
- **Checkpoint frequency**: Every epoch + best model saved
- **Validation frequency**: Every epoch
- **Early stopping**: If val loss doesn't improve for 5 epochs
- **Monitor memory**: Keep eye on MPS memory (models share GPU with API servers)

---

**Last Updated**: 2025-10-30 (Initial creation)
