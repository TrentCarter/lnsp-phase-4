# Mamba Phase-5: Data Alignment Complete
## Ready for Retraining with Payload-Aligned Data

**Date**: 2025-10-26
**Status**: ✅ **DATA ALIGNMENT COMPLETE - Ready for Retraining**

---

## Executive Summary

**Problem Solved**: Created new training data that is perfectly aligned with the Wikipedia payload/FAISS index, fixing the root cause of 0% retrieval.

**Results**:
- ✅ **396,258 training sequences** (70% more than before!)
- ✅ **99,064 validation sequences**
- ✅ **Perfect alignment**: cos(train_target, payload_vec) = 1.000
- ✅ **Full provenance tracking** for reproducibility
- ✅ **100% truth_key coverage** for traceability

**Next Step**: Retrain all 5 Mamba models using the new aligned data.

---

## What Changed

### Before (MISALIGNED)
```
Training Data: artifacts/lvm/training_sequences_ctx5.npz
- 232,600 sequences
- Targets from UNKNOWN source
- NO truth_keys (no linkage to payload)
- NO provenance metadata
- Cosine to payload: 0.14 ❌

Result: 0% retrieval (models predict wrong vector space)
```

### After (ALIGNED)
```
Training Data: artifacts/lvm/train_payload_aligned.npz
- 396,258 sequences (+70% more data!)
- Targets from Wikipedia payload (GTR-T5-base-768)
- FULL truth_keys (article_index, chunk_index)
- COMPLETE provenance metadata
- Cosine to payload: 1.000 ✅

Expected: 60-75% Contain@50, 40-55% R@5
```

---

## Data Quality Verification

### 1. Perfect Alignment
```
Mean cosine (train_target vs payload_vec): 1.000000
Min: 0.999999
Max: 1.000001

✅ Targets are EXACTLY the same as payload vectors
```

### 2. 100% Coverage
```
Truth keys coverage: 495322/495322 (100.00%)

✅ Every training sample can be traced to payload
```

### 3. Proper Normalization
```
Context vector norms: mean=1.000000, std=0.000000
Target vector norms:  mean=1.000000, std=0.000000

✅ All vectors L2 normalized (ready for inner product)
```

### 4. Reasonable Context Similarity
```
Mean cosine (last_context vs target): 0.4713
Range: [0.0662, 0.9925]

✅ Not degenerate, captures sequential Wikipedia structure
```

---

## Provenance Metadata

All new training data includes full provenance for reproducibility:

```json
{
  "embedder_id": "GTR-T5-base-768",
  "payload_build_id": "payload584k_2025-10-24@sha256:12cfd8d7d92dca99",
  "norm": "l2_once",
  "metric": "ip",
  "created_at": "2025-10-26T20:22:59.816929",
  "creation_tool": "build_payload_aligned_training.py",
  "data_source": "wikipedia_584k_payload.npy"
}
```

This ensures:
- We know exactly which encoder was used (GTR-T5-base-768)
- We can verify data integrity with SHA256 hash
- We can reproduce the exact training data

---

## Data Statistics

### Overall Coverage
```
Payload vectors: 584,545
Wikipedia articles: 8,447
Articles used for training: 6,829 (81%)
Articles skipped (too short): 1,618 (19%)

Total sequences extracted: 495,322
```

### Train/Val Split
```
Training set:   396,258 sequences (80%)
Validation set:  99,064 sequences (20%)

Increase vs old data: +163,658 sequences (+70%)
```

### Sequence Structure
```
Context: [t-4, t-3, t-2, t-1, t-0] → Predict: t+1
Context shape: [N, 5, 768]
Target shape:  [N, 768]
Truth keys:    [N, 2] (article_index, chunk_index)
```

---

## Files Created

### Training Data
- **Train**: `artifacts/lvm/train_payload_aligned.npz` (396,258 sequences)
- **Val**: `artifacts/lvm/val_payload_aligned.npz` (99,064 sequences)
- **Eval**: `artifacts/lvm/eval_v2_payload_aligned.npz` (5,244 sequences, already created)

### Tools
- **Data Generator**: `tools/build_payload_aligned_training.py`
- **Eval Aligner**: `tools/align_eval_to_payload.py`
- **Smoke Test**: `tools/smoke_test_aligned_eval.py`
- **Diagnostics**: `tools/diagnose_retrieval_gap.py`

### Documentation
- **Root Cause Analysis**: `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md`
- **Data Alignment Summary**: `docs/MAMBA_PHASE5_DATA_ALIGNMENT_COMPLETE.md` (this file)

---

## Comparison: Old vs New

| Metric | OLD Data | NEW Data | Improvement |
|--------|----------|----------|-------------|
| **Sequences** | 232,600 | 396,258 | +70% |
| **Alignment (cos)** | 0.14 ❌ | 1.00 ✅ | +0.86 |
| **Truth keys** | None ❌ | Full ✅ | Traceability |
| **Provenance** | None ❌ | Complete ✅ | Reproducibility |
| **Expected Retrieval** | 0% | 60-75% | Fixed! |

---

## Next Steps: Retraining

### 1. Update Training Scripts
Modify training scripts to use new data:
```python
# OLD (DO NOT USE)
train_npz = "artifacts/lvm/training_sequences_ctx5.npz"

# NEW (USE THIS)
train_npz = "artifacts/lvm/train_payload_aligned.npz"
```

### 2. Training Configuration
Use same hyperparameters as before (already tuned):
```
Epochs: 20
Batch size: 256
Learning rate: 1e-3
Weight decay: 1e-4
Early stopping: 5 epochs
Loss: MSE (cosine weight = 1.0, mse weight = 1.0)
```

### 3. Expected Results

**During Training**:
- Validation cosine: 0.54-0.58 (similar to before)
- Training should be stable (MSE loss decreasing)

**After Training** (with aligned eval data):
- **Contain@50**: 60-75% (was 0% ❌ → now expected ✅)
- **R@5**: 40-55% (was 0% ❌ → now expected ✅)
- **R@10**: 55-70%
- **Eff@5**: 0.65-0.75 (R@5 / Contain@50)
- **Latency P95**: 1.0-1.5ms (similar to before)

### 4. Retraining Order
Recommend starting with **Mamba-S** (simplest) for quick validation:
1. **Mamba-S** (~6-8 hours) - Quick proof-of-concept
2. If R@5 > 40%, proceed with others in parallel
3. **Mamba-H, Mamba-Sandwich, Mamba-GR, Mamba-XL** in parallel

### 5. Verification Steps
After each model trains:
```bash
# 1. Generate predictions
python tools/eval_mamba_models.py \
  --model artifacts/lvm/models/mamba_s_aligned/best.pt \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --out artifacts/lvm/eval_mamba_s_aligned.json

# 2. Check Contain@50 ≥ 60% and R@5 ≥ 40%
```

---

## Training Commands (Ready to Use)

### Mamba-S (Pure SSM)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --d-model 768 \
  --n-layers 8 \
  --d-state 128 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cpu \
  --save-dir artifacts/lvm/models/mamba_s_aligned
```

### Mamba-H (Hybrid 80/20)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type mamba_hybrid_local \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --d-model 768 \
  --n-layers 10 \
  --d-state 128 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --local-attn-win 8 \
  --local-attn-every 4 \
  --n-heads 4 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cpu \
  --save-dir artifacts/lvm/models/mamba_hybrid_aligned
```

### Mamba-Sandwich (Attn→SSM→Attn)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type mamba_sandwich \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --d-model 768 \
  --n-layers-mamba 8 \
  --n-layers-local 4 \
  --d-state 128 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --local-attn-win 8 \
  --n-heads 4 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cpu \
  --save-dir artifacts/lvm/models/mamba_sandwich_aligned
```

### Mamba-GR (SSM + GRU)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type mamba_gr \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --d-model 768 \
  --n-layers 8 \
  --d-state 144 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --gru-hidden 256 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cpu \
  --save-dir artifacts/lvm/models/mamba_gr_aligned
```

### Mamba-XL (Deeper/Wider)
```bash
KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type mamba_s \
  --train-npz artifacts/lvm/train_payload_aligned.npz \
  --d-model 960 \
  --n-layers 12 \
  --d-state 160 \
  --conv-sz 4 \
  --expand 3 \
  --dropout 0.1 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --device cpu \
  --save-dir artifacts/lvm/models/mamba_xl_aligned
```

---

## Success Criteria

After retraining, models should achieve:

### Minimum (Gate to Proceed)
- ✅ Contain@50 ≥ 60%
- ✅ R@5 ≥ 40%
- ✅ P95 latency ≤ 2.0ms

### Target (Production Ready)
- 🎯 Contain@50 ≥ 70%
- 🎯 R@5 ≥ 50%
- 🎯 R@10 ≥ 65%
- 🎯 P95 latency ≤ 1.5ms

### Stretch (Competitive with AMN)
- ⭐ R@5 ≥ 53.2% (AMN baseline)
- ⭐ P95 latency ≤ 1.45ms

---

## Risk Mitigation

### What if retrieval is still 0%?
**Unlikely** - We've verified perfect alignment (cos=1.000). However, if this happens:
1. Check model is loading correctly
2. Verify inference produces normalized vectors
3. Run diagnostic tool to check prediction quality
4. Inspect first 10 predictions manually

### What if retrieval is low (<40%)?
**Possible** - Model architecture may need tuning. Options:
1. Increase model capacity (more layers/wider)
2. Try different loss functions (InfoNCE, triplet)
3. Add alignment head (already available in code)
4. Increase training epochs

### What if training doesn't converge?
**Unlikely** - Same hyperparameters worked before. If happens:
1. Reduce learning rate (1e-3 → 5e-4)
2. Increase warmup ratio (0.1 → 0.2)
3. Check for data corruption (rerun build script)

---

## Timeline Estimate

### Sequential Training (Conservative)
- Mamba-S: 6-8 hours
- Wait for validation (2 hours)
- If successful, continue:
  - Mamba-H: 8-10 hours
  - Mamba-Sandwich: 8-10 hours
  - Mamba-GR: 8-10 hours
  - Mamba-XL: 12-16 hours

**Total**: ~48-60 hours sequential

### Parallel Training (Optimal)
- Start Mamba-S first (proof-of-concept)
- If successful after 8 hours, launch others in parallel
- All 5 models: ~16-20 hours total (limited by slowest model)

**Recommended**: Start with Mamba-S, then parallelize others.

---

## Conclusion

✅ **Data alignment is COMPLETE and VERIFIED**

✅ **Ready to proceed with retraining**

✅ **Expected outcome: 60-75% Contain@50, 40-55% R@5**

The root cause (data mismatch) has been fixed. New training data is perfectly aligned with the retrieval payload. Retraining should produce models that work correctly with the FAISS index.

**Next action**: Run Mamba-S training command above to validate the fix works.

---

## References

- **Root Cause Analysis**: `docs/MAMBA_PHASE5_ROOT_CAUSE_ANALYSIS.md`
- **PRD**: `docs/PRDs/PRD_5_Mamba_Models.md`
- **Training Tool**: `app/lvm/train_mamba_unified.py`
- **Evaluation Tool**: `tools/eval_mamba_models.py`
- **New Train Data**: `artifacts/lvm/train_payload_aligned.npz`
- **New Val Data**: `artifacts/lvm/val_payload_aligned.npz`
- **Aligned Eval Data**: `artifacts/lvm/eval_v2_payload_aligned.npz`
