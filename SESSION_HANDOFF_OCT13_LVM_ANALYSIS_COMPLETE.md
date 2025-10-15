# Session Handoff - October 13, 2025: LVM Analysis Complete

## What Was Accomplished

### ✅ Ran Full Pipeline Test (5 sequences)
- **Test File**: `tools/test_lvm_full_pipeline.py`
- **Pipeline**: Text → GTR-T5 (768D) → LVM → 768D → vec2text → Text
- **Models Tested**: mamba2 (wrong data), lstm_vec2text (correct data)
- **Results**:
  - mamba2: -4.24% avg (trained on wrong encoder)
  - lstm_vec2text: 6.88% avg (correct encoder, but small dataset)

### ✅ Fixed Critical Encoder Issue
- **Problem**: Port 8765 (sentence-transformers) incompatible with vec2text
- **Solution**: Updated test script to use port 8767 (vec2text-compatible)
- **Verification**: Vec2text baseline now works (semantic reconstruction preserved)

### ✅ Discovered Training Data Problems
1. **3 models trained on WRONG encoder** (port 8765):
   - lstm_baseline, transformer, mamba2
   - Val cosine: 1.6-2.8% (unusable)

2. **3 models trained on CORRECT encoder** (port 8767):
   - lstm_vec2text, gru_vec2text, transformer_vec2text
   - Val cosine: 24-29% (ok, but limited)

3. **Insufficient training data**:
   - Only 8,111 concepts from 135 Wikipedia articles
   - Only 8,106 training sequences
   - Models overfit to Wikipedia style, can't generalize

### ✅ Created Comprehensive Documentation
- `LVM_COMPLETE_ANALYSIS_OCT13.md` - Full analysis + recommendations
- `LVM_RETRAINING_PLAN_OCT13.md` - Detailed retraining strategy
- `LVM_PIPELINE_TEST_RESULTS_OCT13.md` - Initial test results
- `/tmp/lvm_pipeline_test_FIXED.log` - Fixed test log (port 8767)
- `/tmp/lvm_pipeline_test_lstm_vec2text.log` - LSTM test log

---

## Current Status

### Model Performance Summary

| Model | Encoder | Val Cosine | Test Cosine | Verdict |
|-------|---------|-----------|-------------|---------|
| mamba2 | Port 8765 ❌ | 2.8% | -4.24% | DELETE |
| lstm_baseline | Port 8765 ❌ | 1.6% | TBD | DELETE |
| transformer | Port 8765 ❌ | 1.8% | TBD | DELETE |
| lstm_vec2text | Port 8767 ✅ | 28.6% | 6.88% | KEEP |
| gru_vec2text | Port 8767 ✅ | 28.0% | TBD | KEEP |
| transformer_vec2text | Port 8767 ✅ | 24.2% | TBD | KEEP |

### APIs Status
- ✅ Port 8767: vec2text-compatible encoder (RUNNING)
- ✅ Port 8766: vec2text decoder (RUNNING)
- ⚠️ Port 8765: DEPRECATED encoder (should be stopped)

### Data Available
- **PostgreSQL**: 8,111 concepts from 135 articles
- **Training File**: `artifacts/lvm/training_sequences_ctx5.npz` (8,106 sequences)
- **Need**: 50K-100K concepts for good performance

---

## Next Steps (Recommended)

### Option 1: Delete Bad Models ⚡ (5 minutes)
```bash
# Remove models trained on wrong encoder
rm -rf artifacts/lvm/models/mamba2
rm -rf artifacts/lvm/models/lstm_baseline
rm -rf artifacts/lvm/models/transformer

# Keep only vec2text models
ls -d artifacts/lvm/models/*vec2text
```

### Option 2: Quick Test Remaining Models ⚡ (30 minutes)
```bash
# Test GRU (trained on correct data)
PYTHONPATH=app/lvm ./.venv/bin/python tools/test_lvm_full_pipeline.py \
  artifacts/lvm/models/gru_vec2text/best_model.pt \
  gru | tee /tmp/lvm_test_gru_vec2text.log

# Test Transformer (trained on correct data)
PYTHONPATH=app/lvm ./.venv/bin/python tools/test_lvm_full_pipeline.py \
  artifacts/lvm/models/transformer_vec2text/best_model.pt \
  transformer | tee /tmp/lvm_test_transformer_vec2text.log
```

**Expected**: Similar to LSTM (6-15% test cosine)

### Option 3: Ingest More Wikipedia 🚀 (3-4 hours)
```bash
# Download 1000 Wikipedia articles (~50MB)
./.venv/bin/python tools/download_wikipedia_full.py \
  --language 20231101.en \
  --limit 1000 \
  --output episodes/wikipedia_1k.jsonl

# Chunk articles (creates ~60K chunks from 1000 articles)
curl -X POST http://localhost:8001/chunk-episode \
  -H "Content-Type: application/json" \
  -d @episodes/wikipedia_1k.jsonl

# Ingest chunks (generates vectors with port 8767)
# This takes ~2-3 hours for 60K chunks
curl -X POST http://localhost:8004/ingest-chunks \
  -H "Content-Type: application/json" \
  -d '{"input_path": "episodes/chunks/wikipedia_1k_chunks.jsonl"}'

# Verify ingestion
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source='user_input';"
# Should show ~60K+ concepts
```

### Option 4: Generate New Training Data ⚡ (30 minutes)
**Prerequisite**: Option 3 complete (50K+ concepts in database)

```bash
# Extract ordered sequences from PostgreSQL
./.venv/bin/python tools/extract_ordered_training_data.py \
  --db lnsp \
  --output-dir artifacts/lvm \
  --dataset-source user_input \
  --context-size 5

# Should create: training_sequences_ctx5.npz with ~55K sequences
```

### Option 5: Retrain All Models 🚀 (4-6 hours)
**Prerequisite**: Option 4 complete (50K training sequences)

```bash
# LSTM (fastest, 1 hour)
PYTHONPATH=app/lvm nohup ./.venv/bin/python app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5.npz \
  --epochs 30 --batch-size 64 --device cpu \
  --output-dir artifacts/lvm/models/lstm_50k \
  > /tmp/lstm_50k_training.log 2>&1 &

# GRU (1 hour)
PYTHONPATH=app/lvm nohup ./.venv/bin/python app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5.npz \
  --epochs 30 --batch-size 64 --device cpu --use-gru \
  --output-dir artifacts/lvm/models/gru_50k \
  > /tmp/gru_50k_training.log 2>&1 &

# Transformer (2 hours)
PYTHONPATH=app/lvm nohup ./.venv/bin/python app/lvm/train_transformer.py \
  --data artifacts/lvm/training_sequences_ctx5.npz \
  --epochs 30 --batch-size 32 --device cpu \
  --output-dir artifacts/lvm/models/transformer_50k \
  > /tmp/transformer_50k_training.log 2>&1 &

# Monitor training
tail -f /tmp/lstm_50k_training.log
```

**Expected Results**:
- Val cosine: 35-50% (5x better than current 28%)
- Test cosine: 15-25% (2-3x better than current 6.88%)

---

## Key Files Reference

### Documentation (Read These!)
- `LVM_COMPLETE_ANALYSIS_OCT13.md` ⭐ - **START HERE**
- `LVM_RETRAINING_PLAN_OCT13.md` - Detailed retraining steps
- `docs/how_to_use_jxe_and_ielab.md` - Encoder compatibility guide

### Test Results
- `/tmp/lvm_pipeline_test_FIXED.log` - Mamba2 test (wrong encoder)
- `/tmp/lvm_pipeline_test_lstm_vec2text.log` - LSTM test (correct encoder)

### Test Scripts
- `tools/test_lvm_full_pipeline.py` - End-to-end pipeline test
- `tools/extract_ordered_training_data.py` - Generate training sequences

### Training Scripts
- `app/lvm/train_lstm_baseline.py` - LSTM/GRU trainer
- `app/lvm/train_transformer.py` - Transformer trainer
- `app/lvm/train_mamba2.py` - Mamba2 trainer

### Current Models
- `artifacts/lvm/models/lstm_vec2text/` ✅ GOOD (28.6% val)
- `artifacts/lvm/models/gru_vec2text/` ✅ GOOD (28.0% val)
- `artifacts/lvm/models/transformer_vec2text/` ✅ GOOD (24.2% val)
- `artifacts/lvm/models/mamba2/` ❌ BAD (2.8% val, wrong encoder)
- `artifacts/lvm/models/lstm_baseline/` ❌ BAD (1.6% val, wrong encoder)
- `artifacts/lvm/models/transformer/` ❌ BAD (1.8% val, wrong encoder)

### Training Data
- `artifacts/lvm/training_sequences_ctx5.npz` - 8K sequences (CURRENT)
- `artifacts/lvm/training_sequences_ctx5_sentence.npz` - DEPRECATED (DELETE)

---

## Quick Commands

### Start APIs (if not running)
```bash
# Start vec2text-compatible encoder (port 8767)
VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn \
  app.api.vec2text_embedding_server:app \
  --host 127.0.0.1 --port 8767 &

# Start vec2text decoder (port 8766)
VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn \
  app.api.vec2text_server:app \
  --host 127.0.0.1 --port 8766 &

# Verify both APIs
curl http://localhost:8767/health
curl http://localhost:8766/health
```

### Stop Deprecated Encoder
```bash
# Kill port 8765 (sentence-transformers - deprecated)
pkill -f "8765"
```

### Check Training Progress
```bash
# Monitor LSTM training
tail -f /tmp/lstm_50k_training.log | grep "Epoch"

# Check validation metrics
tail -f /tmp/lstm_50k_training.log | grep "Val Loss"

# Check if training completed
ls -lh artifacts/lvm/models/lstm_50k/best_model.pt
```

### Quick Database Check
```bash
# How many concepts do we have?
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"

# How many articles?
psql lnsp -c "SELECT COUNT(DISTINCT batch_id) FROM cpe_entry;"

# Check latest ingestion
psql lnsp -c "SELECT MAX(created_at) FROM cpe_entry;"
```

---

## Success Criteria

### Minimum Acceptable ✅
- Val cosine > 35% (5x current)
- Test cosine > 12% (2x current)
- No gibberish output
- Models trained on port 8767 data only

### Good Performance ⭐
- Val cosine > 45%
- Test cosine > 18%
- Semantic coherence preserved

### Excellent Performance 🏆
- Val cosine > 55%
- Test cosine > 25%
- Generalizes to diverse domains
- (Requires diverse training data, not just Wikipedia)

---

## Critical Rules (from CLAUDE.md)

### ❌ NEVER Use Ontology Data for LVM Training
- Ontologies are TAXONOMIC (classification hierarchies)
- LVMs need SEQUENTIAL data (temporal/causal flow)
- ✅ Use: Wikipedia, recipes, tutorials, stories
- ❌ Never: WordNet, SWO, GO, DBpedia ontologies

### ✅ Always Use Correct Encoder
- ✅ Port 8767: vec2text-compatible GTR-T5
- ❌ Port 8765: DEPRECATED sentence-transformers
- Verify: `curl http://localhost:8767/health`

### ✅ Training Data Must Be Sequential
- Order matters! (batch_id ASC, created_at ASC)
- Concepts must follow document flow
- Use `extract_ordered_training_data.py` to preserve order

---

## Recommended Flow for Next Session

1. **Read Documentation** (10 min):
   - `LVM_COMPLETE_ANALYSIS_OCT13.md`
   - Understand encoder issue and data shortage

2. **Quick Cleanup** (5 min):
   - Delete models trained on wrong encoder (mamba2, lstm_baseline, transformer)
   - Stop port 8765 API

3. **Choose Path**:
   - **Path A**: Test remaining models (GRU, Transformer) → Accept current performance
   - **Path B**: Ingest more Wikipedia (3-4h) → Retrain all models (4-6h) → Better performance

4. **If Path B** (Recommended):
   - Start Wikipedia download (1000 articles)
   - Monitor ingestion progress
   - Generate new training sequences
   - Train all models overnight
   - Test next session

---

**Session Date**: October 13, 2025
**Status**: ✅ Analysis complete, ready for data ingestion + retraining
**Recommendation**: Option 3 (Ingest more Wikipedia) + Option 5 (Retrain all)
**Estimated Time**: 8-12 hours (can run overnight)
**Expected Improvement**: 2-3x test performance (6.88% → 15-25%)
