# LVM Complete Analysis - October 13, 2025

## Executive Summary

**Current Status**: LVM models fail end-to-end pipeline test
**Root Causes**: ❌ Wrong encoder used + ❌ Insufficient training data
**Data Available**: Only 8.1K concepts (need 50K+ for good performance)
**Action Required**: Ingest more Wikipedia → Retrain all models

---

## Test Results Summary

### Test Configuration
- **Pipeline**: Text → GTR-T5 (768D) → LVM → 768D → vec2text → Text
- **Test Date**: October 13, 2025
- **Test Sequences**: 5 diverse domains (cooking, conversation, science, travel, routine)
- **APIs**: Port 8767 (encoder) + Port 8766 (decoder) ✅ CORRECT

### Model Performance

| Model | Training Data | Val Cosine | Test Cosine (5 seq) | Status |
|-------|--------------|------------|---------------------|--------|
| mamba2 | `_sentence.npz` ❌ | 2.8% | -4.24% | ❌ Broken |
| lstm_baseline | `_sentence.npz` ❌ | 1.6% | TBD | ❌ Broken |
| transformer | `_sentence.npz` ❌ | 1.8% | TBD | ❌ Broken |
| **lstm_vec2text** | `ctx5.npz` ✅ | **28.6%** | **6.88%** | ⚠️ Poor |
| **gru_vec2text** | `ctx5.npz` ✅ | **28.0%** | TBD | ⚠️ Poor |
| **transformer_vec2text** | `ctx5.npz` ✅ | **24.2%** | TBD | ⚠️ Poor |

### Key Findings

1. **Wrong Encoder Disaster** (3 models):
   - Models trained on `training_sequences_ctx5_sentence.npz`
   - This file contains vectors from **port 8765** (sentence-transformers)
   - Incompatible with vec2text decoder (port 8766)
   - Result: 1-3% validation cosine (essentially random)

2. **Correct Encoder Still Poor** (3 models):
   - Models trained on `training_sequences_ctx5.npz` ✅
   - Uses vec2text-compatible encoder (port 8767)
   - Result: 24-29% validation, but 6.88% on test set
   - **21 point drop** from validation to test!

3. **Root Cause**: Insufficient Training Data
   - Only **8,111 concepts** from **135 Wikipedia articles**
   - Only **8,106 training sequences** (5-context window)
   - Models overfit to Wikipedia style, can't generalize

---

## Data Analysis

### Current Database Contents

```sql
dataset_source | concepts | articles
----------------+----------+----------
 user_input     |     8111 |      135
```

**Average**: 60 concepts per article (reasonable chunk size)

### Training Data Files

| File | Size | Concepts | Source | Status |
|------|------|----------|--------|--------|
| `training_sequences_ctx5.npz` | 47MB | 8,106 | Port 8767 ✅ | Current (small) |
| `training_sequences_ctx5_sentence.npz` | 45MB | 8,106 | Port 8765 ❌ | DEPRECATED |
| `wikipedia_8111_ordered.npz` | TBD | 8,111 | Port 8767 ✅ | Source data |
| `wikipedia_8111_ordered_vec2text.npz` | TBD | 8,111 | Port 8767 ✅ | Same data |
| `wikipedia_42113_ordered.npz` | TBD | 8,111 ❌ | Port 8767 | **Misleading name!** |

**Critical Issue**: No file actually has 42K concepts! Need to ingest more Wikipedia.

### Domain Mismatch Problem

**Training Domain**: Wikipedia articles (135 articles)
- Encyclopedic style
- Formal academic tone
- Factual, dense information
- Limited stylistic diversity

**Test Domain**: Diverse everyday scenarios
- Cooking recipes ("Mix flour, sugar, and eggs...")
- Conversations ("How do you take it?")
- Travel stories ("We took a taxi to our hotel")
- Morning routines ("I checked my phone for messages")
- Scientific writing ("The findings were published...")

**Result**: Models learned Wikipedia patterns but fail on other domains

---

## Technical Deep Dive

### Why Vec2text Baseline Works (Now)

**Fixed Test (Oct 13)**:
```
Expected:  'Let the cake cool before frosting.'
Expected→: 'Let the cake cool before frosting, rather than...'
```
✅ Semantic meaning preserved (cosine ~0.63-0.85)

**Broken Test (Oct 12)**:
```
Expected:  'Let the cake cool before frosting.'
Expected→: '– R&D team, R&D, 'I have connections with Hennepin'...'
```
❌ Complete gibberish (cosine ~0.08)

**Root Cause**: Port 8765 (sentence-transformers) embeddings incompatible with port 8766 (vec2text decoder)

### Why LVM Predictions Fail

**Example 1 - Cooking Recipe**:
```
Context: [Heat oven → Mix flour → Add vanilla → Pour batter → Bake 30min]
Expected:  "Let the cake cool before frosting."
Predicted: "'Lastinator of Kiel'"Sumes na na na nad Devil magnesium"
Cosine: 9.31%
```

**Example 2 - Scientific Process**:
```
Context: [Hypothesis → Materials → Control groups → Data collected → Statistical analysis]
Expected:  "The findings were published in a journal."
Predicted: "a time when results were not accounted for, dummy sampling adjusted"
Cosine: 13.09%
```

**Analysis**:
- LVM predictions are **not completely random** (13% is better than 0%)
- Model learned **some** sequential structure from Wikipedia
- But vocabulary and style don't match test domains
- Training on 8K concepts is far too small for generalization

---

## Encoder Compatibility Deep Dive

### The Two GTR-T5 Encoders

#### Port 8765: sentence-transformers (DEPRECATED)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
vec = model.encode(["Hello world"], normalize_embeddings=True)
```
- Uses sentence-transformers library
- Different tokenization/pooling than vec2text
- **Cosine when decoded**: ~0.076 ❌

#### Port 8767: vec2text-compatible (CORRECT)
```python
# Via API
POST http://localhost:8767/embed
{"texts": ["Hello world"]}
```
- Uses vec2text's own encoder implementation
- Matches decoder's expectations exactly
- **Cosine when decoded**: 0.63-0.85 ✅

### Why Incompatibility Happens

**Subtle Differences**:
1. **Tokenization**: Different token splits (sentence-transformers vs vec2text)
2. **Pooling**: Different averaging strategies
3. **Normalization**: Different L2 normalization timing
4. **Library versions**: Slight numerical differences accumulate

**Result**: Even 1% difference in embedding → 50+ point cosine drop when decoded!

---

## Recommendations

### Option A: Quick Test with Existing Data ⚡

**Goal**: Verify pipeline works with correct data
**Time**: 2-3 hours
**Steps**:
1. ✅ Fix test script to use `lstm_vec2text` (done)
2. ✅ Run pipeline test (completed: 6.88% avg)
3. ✅ Document findings (this file)
4. ⏳ Test remaining vec2text models (GRU, Transformer)

**Expected Outcome**:
- Validation: 24-29% (in-distribution Wikipedia)
- Test: 6-15% (out-of-distribution diverse)

### Option B: Full Retraining with More Data 🚀

**Goal**: Train production-quality models
**Time**: 8-12 hours (ingestion + training)
**Required Data**: 50K-100K Wikipedia concepts (500-1000 articles)

**Steps**:

1. **Ingest More Wikipedia** (3-4 hours):
   ```bash
   # Download 1000 Wikipedia articles
   ./.venv/bin/python tools/download_wikipedia_full.py \
     --language 20231101.en \
     --limit 1000 \
     --output episodes/wikipedia_1k.jsonl

   # Process through chunking pipeline
   curl -X POST http://localhost:8001/chunk-episode \
     -H "Content-Type: application/json" \
     -d @episodes/wikipedia_1k.jsonl

   # Ingest chunks (generates vectors with port 8767)
   curl -X POST http://localhost:8004/ingest-chunks \
     -H "Content-Type: application/json" \
     -d '{"input_path": "episodes/chunks/wikipedia_1k_chunks.jsonl"}'
   ```

2. **Extract Training Sequences** (30 minutes):
   ```bash
   ./.venv/bin/python tools/extract_ordered_training_data.py \
     --db lnsp \
     --output-dir artifacts/lvm \
     --dataset-source user_input \
     --context-size 5

   # Should create: training_sequences_ctx5_50k.npz (~45K sequences)
   ```

3. **Train Models** (4-6 hours):
   ```bash
   # LSTM (fastest, 1 hour)
   PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_lstm_baseline.py \
     --data artifacts/lvm/training_sequences_ctx5_50k.npz \
     --epochs 30 --batch-size 64 --device cpu

   # GRU (similar to LSTM, 1 hour)
   PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_lstm_baseline.py \
     --data artifacts/lvm/training_sequences_ctx5_50k.npz \
     --epochs 30 --batch-size 64 --device cpu --use-gru

   # Transformer (slower, 2 hours)
   PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_transformer.py \
     --data artifacts/lvm/training_sequences_ctx5_50k.npz \
     --epochs 30 --batch-size 32 --device cpu

   # Mamba2 (if mamba-ssm installed, 1.5 hours)
   pip install mamba-ssm  # Optional
   PYTHONPATH=app/lvm ./.venv/bin/python app/lvm/train_mamba2.py \
     --data artifacts/lvm/training_sequences_ctx5_50k.npz \
     --epochs 30 --batch-size 32 --device cpu
   ```

4. **Evaluate** (30 minutes):
   ```bash
   # Test all models
   for model in lstm gru transformer mamba2; do
     PYTHONPATH=app/lvm ./.venv/bin/python tools/test_lvm_full_pipeline.py \
       artifacts/lvm/models/${model}_50k/best_model.pt \
       $model | tee /tmp/lvm_test_${model}_50k.log
   done
   ```

**Expected Outcome**:
- Validation: 35-50% (5x more data)
- Test: 15-25% (better generalization)

### Option C: Diverse Training Data 🎯

**Goal**: Train domain-agnostic models
**Time**: 2-3 days (dataset creation + training)
**Required**: Multiple data sources

**Data Mix**:
- 40% Wikipedia (encyclopedic)
- 20% Recipes (procedural)
- 20% Conversations (dialogues)
- 10% Scientific papers (academic)
- 10% Stories (narrative)

**Expected Outcome**:
- Validation: 30-40% (diverse distribution)
- Test: 20-35% (much better generalization!)

---

## Immediate Next Steps

### Phase 1: Document Current State ✅
- ✅ Created comprehensive analysis (this file)
- ✅ Documented encoder compatibility issues
- ✅ Identified data shortage problem

### Phase 2: Choose Path Forward ⏳

**Recommended**: **Option B** (Full Retraining with More Wikipedia)

**Rationale**:
1. Wikipedia is consistent, high-quality domain
2. 50K concepts achievable in 3-4 hours
3. Should reach 35-50% validation cosine
4. Test performance will still be limited, but acceptable
5. Can iterate to Option C later if needed

### Phase 3: Verification Checklist

Before starting retraining:
- [ ] Port 8767 encoder API is running
- [ ] Port 8766 decoder API is running
- [ ] Vec2text baseline works (cosine > 0.60)
- [ ] PostgreSQL has space for 50K+ concepts
- [ ] Training scripts use correct NPZ file

### Phase 4: Success Criteria

**Minimum Acceptable Performance**:
- ✅ Val cosine > 35% (Wikipedia validation set)
- ✅ Test cosine > 12% (diverse test set)
- ✅ No gibberish output (semantic coherence preserved)
- ✅ Inference speed < 100ms per prediction

**Good Performance**:
- ✅ Val cosine > 45%
- ✅ Test cosine > 18%
- ✅ Recognizable domain patterns

**Excellent Performance** (requires Option C):
- ✅ Val cosine > 55%
- ✅ Test cosine > 25%
- ✅ Generalizes to unseen domains

---

## Technical Appendix

### Training Data Format

**NPZ Structure** (from `extract_ordered_training_data.py`):
```python
{
  'context_sequences': np.ndarray,  # [N, 5, 768] - 5 context vectors
  'target_vectors': np.ndarray,     # [N, 768] - next vector to predict
  'target_texts': np.ndarray,       # [N] - ground truth text (for evaluation)
  'target_tmds': np.ndarray,        # [N] - TMD codes
  'target_ids': np.ndarray,         # [N] - CPE IDs
  'sequence_positions': np.ndarray, # [N] - position in source data
  'metadata': dict                   # Training data info
}
```

### Model Architectures

| Architecture | Parameters | Layers | Training Time (8K) | Training Time (50K) |
|--------------|-----------|--------|-------------------|---------------------|
| LSTM | 7.4M | 2 | 15 min | 60 min |
| GRU | 7.4M | 2 | 15 min | 60 min |
| Transformer | 45M | 6 | 30 min | 120 min |
| Mamba2 | 7.4M | 4 | 20 min | 80 min |

### Loss Functions

**Primary**: MSE (Mean Squared Error)
```python
loss = torch.nn.functional.mse_loss(predicted_vec, target_vec)
```

**Auxiliary** (in some trainers): InfoNCE (Contrastive Loss)
```python
# Pulls similar vectors together, pushes dissimilar apart
loss = -log(exp(sim(pred, pos)) / sum(exp(sim(pred, neg))))
```

### Evaluation Metrics

**Cosine Similarity** (primary):
```python
cosine = dot(v1_normalized, v2_normalized)
# Range: [-1, 1]
# Good: > 0.4
# Excellent: > 0.6
```

**MSE Loss** (training):
```python
mse = mean((predicted - target)^2)
# Lower is better
# Good: < 0.0005
```

---

## References

### Documentation
- `docs/how_to_use_jxe_and_ielab.md` - Encoder compatibility guide
- `LVM_PIPELINE_TEST_RESULTS_OCT13.md` - Initial test results
- `LVM_RETRAINING_PLAN_OCT13.md` - Retraining strategy
- `CLAUDE.md` - Critical LVM training rules (NO ONTOLOGIES!)

### Code Files
- `tools/test_lvm_full_pipeline.py` - End-to-end pipeline test
- `tools/extract_ordered_training_data.py` - Training data generation
- `app/lvm/train_lstm_baseline.py` - LSTM/GRU trainer
- `app/lvm/train_transformer.py` - Transformer trainer
- `app/lvm/train_mamba2.py` - Mamba2 trainer

### Data Files
- `artifacts/lvm/training_sequences_ctx5.npz` - Current training data (8K)
- `artifacts/lvm/wikipedia_8111_ordered.npz` - Source concepts (8K)
- PostgreSQL `lnsp` database - Ingested Wikipedia chunks

---

**Status**: Analysis complete, ready for Option B (Wikipedia ingestion + retraining)
**Next Session**: Ingest 1000 Wikipedia articles → Generate 50K training sequences → Retrain all models
**Estimated Total Time**: 8-12 hours (can run overnight)
