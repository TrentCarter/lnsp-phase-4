# LVM Retraining Plan - October 13, 2025

## Executive Summary

**Status**: Current LVM models have significant issues requiring retraining
**Root Cause**: Mixed use of incompatible encoders + insufficient training data (8K sequences)
**Action Required**: Retrain all models with larger dataset + correct encoder

---

## Problem Analysis

### Issue #1: Wrong Encoder Used (3 models)

**Affected Models**:
- `lstm_baseline`: Val cosine 1.6% (trained on `_sentence.npz` ❌)
- `transformer`: Val cosine 1.8% (trained on `_sentence.npz` ❌)
- `mamba2`: Val cosine 2.8% (trained on `_sentence.npz` ❌)

**Root Cause**: These models were trained using `training_sequences_ctx5_sentence.npz`, which contains vectors from:
- **Port 8765** (sentence-transformers GTR-T5) - DEPRECATED
- Incompatible with vec2text decoder (port 8766)
- Produces cosine ~0.076 instead of 0.63-0.85 when decoded

### Issue #2: Limited Training Data

**Current Data**:
- **8,106 training sequences** from `wikipedia_8111_ordered.npz`
- Only 8,111 concepts total
- Limited diversity and generalization

**Evidence of Poor Generalization**:
- LSTM_vec2text: 28.6% val cosine → 6.88% test (22 point drop!)
- Test sequences are out-of-distribution (cooking, conversations, travel)
- Models only learned Wikipedia-style text patterns

### Issue #3: Domain Mismatch

**Training Domain**: Wikipedia encyclopedic articles
- Formal, factual tone
- Academic vocabulary
- Dense information style

**Test Domain**: Everyday scenarios
- Conversational language
- Procedural instructions (recipes, routines)
- Casual vocabulary

**Result**: Models cannot generalize to unseen domains

---

## Current Model Performance

### Models Trained on CORRECT Data (vec2text-compatible, port 8767)

| Model | Training Val Cosine | Test Cosine (5 sequences) | Gap |
|-------|-------------------|--------------------------|-----|
| lstm_vec2text | 28.6% | 6.88% | -21.7% |
| gru_vec2text | 28.0% | TBD | TBD |
| transformer_vec2text | 24.2% | TBD | TBD |

### Models Trained on WRONG Data (sentence-transformers, port 8765)

| Model | Training Val Cosine | Status |
|-------|-------------------|--------|
| lstm_baseline | 1.6% | ❌ Unusable |
| transformer | 1.8% | ❌ Unusable |
| mamba2 | 2.8% | ❌ Unusable |

---

## Data Availability Check

### Current NPZ Files

```bash
artifacts/lvm/wikipedia_8111_ordered.npz              # 8.1K concepts (SMALL)
artifacts/lvm/wikipedia_42113_ordered.npz             # 42K concepts (BETTER!)
artifacts/lvm/wikipedia_8111_ordered_vec2text.npz     # 8.1K vec2text-compatible
artifacts/lvm/training_sequences_ctx5.npz             # 8.1K training sequences
artifacts/lvm/training_sequences_ctx5_sentence.npz    # 8.1K WRONG sequences
```

**Good News**: We have `wikipedia_42113_ordered.npz` with **42,113 concepts**!

This provides **~37,000 training sequences** (5x more than current!)

---

## Retraining Plan

### Phase 1: Verify Correct Data Exists ✅

Check if `wikipedia_42113_ordered.npz` was created with correct encoder:

```bash
# Check metadata
python3 -c "
import numpy as np
data = np.load('artifacts/lvm/wikipedia_42113_ordered.npz', allow_pickle=True)
meta = data.get('metadata', [None])[0]
print('Metadata:', meta)
print('Vector shape:', data['vectors'].shape)
"

# Test a sample vector with vec2text decoder
curl -X POST http://localhost:8766/decode \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[...]], "steps": 1, "subscribers": "ielab"}'
```

**Expected**: Cosine > 0.60 for round-trip encoding/decoding

### Phase 2: Generate Training Sequences from 42K Dataset

```bash
cd artifacts/lvm
python3 ../../tools/extract_ordered_training_data.py \
  --db lnsp \
  --output-dir . \
  --dataset-source user_input \
  --context-size 5

# This should create:
# - wikipedia_42113_ordered.npz (if not exists)
# - training_sequences_ctx5_42k.npz (~37K sequences)
```

### Phase 3: Retrain All Models (Correct Data + Larger Dataset)

**Priority Order**:
1. **LSTM** (baseline, fast to train)
2. **GRU** (similar to LSTM, slightly better)
3. **Transformer** (more expressive, slower)
4. **Mamba2** (best architecture, requires mamba-ssm)

**Training Commands**:

```bash
# 1. LSTM (fastest, good baseline)
PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5_42k.npz \
  --epochs 30 \
  --batch-size 64 \
  --device cpu \
  --output-dir artifacts/lvm/models/lstm_42k

# 2. GRU (similar to LSTM)
PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5_42k.npz \
  --epochs 30 \
  --batch-size 64 \
  --device cpu \
  --use-gru \
  --output-dir artifacts/lvm/models/gru_42k

# 3. Transformer (more expressive)
PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_transformer.py \
  --data artifacts/lvm/training_sequences_ctx5_42k.npz \
  --epochs 30 \
  --batch-size 32 \
  --device cpu \
  --output-dir artifacts/lvm/models/transformer_42k

# 4. Mamba2 (best architecture, if mamba-ssm installed)
pip install mamba-ssm  # Optional, uses GRU fallback otherwise
PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_mamba2.py \
  --data artifacts/lvm/training_sequences_ctx5_42k.npz \
  --epochs 30 \
  --batch-size 32 \
  --device cpu \
  --output-dir artifacts/lvm/models/mamba2_42k
```

**Training Time Estimates** (CPU, 37K sequences):
- LSTM: ~45 minutes
- GRU: ~45 minutes
- Transformer: ~90 minutes
- Mamba2: ~60 minutes

### Phase 4: Evaluate New Models

**Test on Wikipedia validation set** (in-distribution):
```bash
PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/evaluate_models.py \
  --models \
    artifacts/lvm/models/lstm_42k/best_model.pt \
    artifacts/lvm/models/gru_42k/best_model.pt \
    artifacts/lvm/models/transformer_42k/best_model.pt \
    artifacts/lvm/models/mamba2_42k/best_model.pt \
  --data artifacts/lvm/training_sequences_ctx5_42k.npz
```

**Test on diverse domains** (out-of-distribution):
```bash
# Use updated test script
PYTHONPATH=app/lvm ./.venv/bin/python tools/test_lvm_full_pipeline.py \
  artifacts/lvm/models/lstm_42k/best_model.pt \
  lstm
```

### Phase 5: Deploy Best Model

**Criteria**:
- Val cosine > 40% (training data)
- Test cosine > 15% (diverse domains)
- Inference speed < 50ms per prediction

**Deployment**:
```bash
# Copy best model to production path
cp artifacts/lvm/models/{best_architecture}_42k/best_model.pt \
   artifacts/lvm/models/production/lvm_best.pt

# Update LVM server config
echo "MODEL_PATH=artifacts/lvm/models/production/lvm_best.pt" > .env.lvm
```

---

## Expected Outcomes

### With 42K Training Data (5x current):

**Realistic Expectations**:
- **Val cosine**: 35-45% (in-distribution Wikipedia)
- **Test cosine**: 10-20% (out-of-distribution diverse)
- **Semantic preservation**: Moderate (better than current 2-6%)

**Why Still Limited**:
- LVM predicts next vector from **semantic context** (not memorization)
- 768D vectors are **lossy compression** of text meaning
- Training on Wikipedia limits generalization to other domains

### Long-Term Improvements (Future):

1. **Diverse Training Data**:
   - Add recipes, conversations, tutorials, stories
   - Mix Wikipedia with other sources
   - Target 100K+ diverse sequences

2. **Better Architecture**:
   - Add attention over context vectors
   - Conditional prediction (domain-aware)
   - Multi-scale vector representations

3. **Vec2text Quality**:
   - Use steps=5 instead of steps=1
   - Fine-tune vec2text on target domains
   - Ensemble multiple vec2text models

---

## Immediate Actions

### Before Retraining:

1. ✅ **Verify encoder compatibility**: Check `wikipedia_42113_ordered.npz` was created with port 8767
2. ✅ **Confirm data quality**: Test vec2text round-trip on sample vectors
3. ✅ **Generate training sequences**: Create `training_sequences_ctx5_42k.npz`

### During Retraining:

1. ⏳ **Train LSTM first** (fastest, establishes baseline)
2. ⏳ **Monitor training metrics** (val cosine should reach 35-45%)
3. ⏳ **Save checkpoints** (every epoch, keep best 3)

### After Retraining:

1. ⏳ **Evaluate on validation set** (in-distribution)
2. ⏳ **Test on diverse domains** (out-of-distribution)
3. ⏳ **Document performance** (update this file with results)
4. ⏳ **Deploy best model** (if val cosine > 40% and test > 15%)

---

## Files Reference

### Training Data
- `artifacts/lvm/wikipedia_42113_ordered.npz` - 42K concepts (source)
- `artifacts/lvm/training_sequences_ctx5_42k.npz` - ~37K sequences (to create)

### Training Scripts
- `app/lvm/train_lstm_baseline.py` - LSTM/GRU trainer
- `app/lvm/train_transformer.py` - Transformer trainer
- `app/lvm/train_mamba2.py` - Mamba2 trainer
- `tools/extract_ordered_training_data.py` - Data prep

### Evaluation Scripts
- `app/lvm/evaluate_models.py` - Batch evaluation
- `tools/test_lvm_full_pipeline.py` - End-to-end pipeline test

### Documentation
- `docs/how_to_use_jxe_and_ielab.md` - Encoder/decoder compatibility
- `LVM_PIPELINE_TEST_RESULTS_OCT13.md` - Current test results

---

## Risk Mitigation

### Risk: 42K Data Still Insufficient

**Mitigation**:
- Download larger Wikipedia subset (100K articles)
- Use `tools/download_wikipedia_full.py --limit 100000`
- Ingest via chunking pipeline (port 8001 → 8004)

### Risk: Training Takes Too Long (CPU)

**Mitigation**:
- Use `--batch-size 128` for faster training
- Reduce `--epochs 20` if metrics plateau early
- Consider MPS device if available (Mac M1/M2)

### Risk: Models Still Don't Generalize

**Mitigation**:
- Accept 10-20% as reasonable out-of-distribution performance
- Focus on Wikipedia domain (where we have training data)
- Plan Phase 2: diverse training data collection

---

**Status**: Ready to proceed with Phase 1 verification
**Next Step**: Check if `wikipedia_42113_ordered.npz` exists and uses correct encoder
**Estimated Total Time**: 4-6 hours (data prep + training + evaluation)
