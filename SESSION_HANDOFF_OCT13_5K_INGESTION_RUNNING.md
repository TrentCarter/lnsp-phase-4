# Session Handoff - October 13, 2025: 5K Wikipedia Ingestion Running

## ✅ Current Status

**5,000 Article Wikipedia Ingestion IN PROGRESS**
- **Started**: October 13, 2025, 7:24 AM
- **Process ID**: 5972
- **Expected completion**: ~11:15 AM (IMPROVED from 1:00 PM!)
- **Progress**: 108 articles, 1,247 concepts (2.1% complete)
- **Performance**: 21.6 articles/min (faster than expected!)

---

## 📊 Performance Test Results (10 articles)

| Metric | Value |
|--------|-------|
| Articles | 10 |
| Chunks | 135 |
| Concepts | 135 |
| Time | 40 seconds |
| Rate | 4 seconds/article |

**Extrapolation to 5,000 articles**:
- Concepts: ~67,500 new
- Total database: ~75,600 (existing 8.1K + 67.5K)
- Training sequences: ~70,000 (vs current 8,106)
- **9x data improvement**

---

## 🔍 Monitor Ingestion

### ⭐ Best: Live Status (bypasses Python buffering)
```bash
/tmp/live_ingestion_status.sh
```

### Auto-refresh every 30 seconds
```bash
watch -n 30 /tmp/live_ingestion_status.sh
```

### Database Query
```bash
psql lnsp -c "SELECT COUNT(DISTINCT batch_id) as articles, COUNT(*) as concepts FROM cpe_entry WHERE dataset_source='wikipedia_5k';"
```

### Check Process
```bash
ps aux | grep 5972
```

### Original Progress Log (buffered, may be empty)
```bash
tail -f /tmp/ingest_5000_progress.log
```

**Note**: Python buffers stdout, so `/tmp/ingest_5000_progress.log` may appear empty until buffer flushes. Use `/tmp/live_ingestion_status.sh` for real-time monitoring via database queries.

---

## 📁 Key Files

### Ingestion
- **Script**: `/tmp/ingest_5000_articles.py`
- **Input**: `/tmp/wiki_5000_articles.jsonl` (5,000 articles)
- **Log**: `/tmp/ingest_5000_progress.log`
- **Monitor**: `/tmp/monitor_ingestion.sh`

### Source Data
- **Full Wikipedia**: `data/datasets/wikipedia/full_wikipedia.jsonl` (100K articles, 506MB)
- **Metadata**: `data/datasets/wikipedia/full_wikipedia_metadata.json`

### Documentation
- **LVM Analysis**: `LVM_COMPLETE_ANALYSIS_OCT13.md`
- **Retraining Plan**: `LVM_RETRAINING_PLAN_OCT13.md`
- **Performance Test**: `/tmp/perf_test_10articles.log`

---

## ⏭️ Next Steps (After Ingestion)

### Step 1: Verify Completion (~11:15 AM)

```bash
# Check final log
tail -50 /tmp/ingest_5000_progress.log

# Verify count
psql lnsp -c "
SELECT
    dataset_source,
    COUNT(*) as concepts,
    COUNT(DISTINCT batch_id) as articles
FROM cpe_entry
GROUP BY dataset_source;
"

# Expected output:
# dataset_source  | concepts | articles
# ----------------+----------+----------
# user_input      |     8111 |      135  (original)
# wikipedia_5k    |   ~67500 |     5000  (new)
```

### Step 2: Extract Training Sequences (~30 min)

```bash
# Generate new training data from ALL concepts
python3 tools/extract_ordered_training_data.py \
  --db lnsp \
  --output-dir artifacts/lvm \
  --dataset-source ALL \
  --context-size 5

# Should create:
# artifacts/lvm/training_sequences_ctx5_75k.npz (~70K sequences)
```

### Step 3: Retrain Models (~4-6 hours)

**LSTM** (fastest, 1 hour):
```bash
PYTHONPATH=app/lvm nohup python3 app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5_75k.npz \
  --epochs 30 \
  --batch-size 64 \
  --device cpu \
  --output-dir artifacts/lvm/models/lstm_75k \
  > /tmp/lstm_75k_training.log 2>&1 &
```

**GRU** (1 hour):
```bash
PYTHONPATH=app/lvm nohup python3 app/lvm/train_lstm_baseline.py \
  --data artifacts/lvm/training_sequences_ctx5_75k.npz \
  --epochs 30 \
  --batch-size 64 \
  --device cpu \
  --use-gru \
  --output-dir artifacts/lvm/models/gru_75k \
  > /tmp/gru_75k_training.log 2>&1 &
```

**Transformer** (2 hours):
```bash
PYTHONPATH=app/lvm nohup python3 app/lvm/train_transformer.py \
  --data artifacts/lvm/training_sequences_ctx5_75k.npz \
  --epochs 30 \
  --batch-size 32 \
  --device cpu \
  --output-dir artifacts/lvm/models/transformer_75k \
  > /tmp/transformer_75k_training.log 2>&1 &
```

### Step 4: Test New Models

```bash
# Test best model (probably LSTM or GRU)
PYTHONPATH=app/lvm python3 tools/test_lvm_full_pipeline.py \
  artifacts/lvm/models/lstm_75k/best_model.pt \
  lstm | tee /tmp/lvm_test_lstm_75k.log
```

**Expected Results**:
- Val cosine: 45-55% (vs current 28.6%)
- Test cosine: 18-25% (vs current 6.88%)
- 3x improvement on diverse test set

---

## 🚨 If Ingestion Fails

### Check Logs
```bash
tail -100 /tmp/ingest_5000_progress.log
```

### Check APIs
```bash
curl http://localhost:8001/health  # Chunker
curl http://localhost:8004/health  # Ingest
curl http://localhost:8767/health  # Encoder
```

### Restart APIs if Needed
```bash
# Chunker
pkill -f "8001"
python3 -m uvicorn app.api.chunking:app --host 127.0.0.1 --port 8001 &

# Ingest
pkill -f "8004"
python3 -m uvicorn app.api.ingest_chunks:app --host 127.0.0.1 --port 8004 &

# Encoder (vec2text-compatible)
pkill -f "8767"
VEC2TEXT_FORCE_CPU=1 python3 -m uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767 &
```

### Resume Ingestion
```bash
# Check how many were completed
COMPLETED=$(psql lnsp -t -c "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source='wikipedia_5k';" | tr -d ' ')
echo "Completed: $COMPLETED"

# If needed, modify script to skip completed articles and restart
# (Manual intervention required)
```

---

## 📈 Expected Outcomes

### Database Growth
| Metric | Before | After | Growth |
|--------|--------|-------|--------|
| Total concepts | 8,111 | ~75,600 | 9.3x |
| Articles | 135 | 5,135 | 38x |
| Training sequences | 8,106 | ~70,000 | 8.6x |

### LVM Performance Predictions
| Model | Current Val | Expected Val | Current Test | Expected Test |
|-------|-------------|--------------|--------------|---------------|
| LSTM | 28.6% | 50-55% | 6.88% | 20-25% |
| GRU | 28.0% | 48-53% | ~7% | 18-23% |
| Transformer | 24.2% | 45-50% | ~6% | 15-20% |

---

## ⚠️ Critical Reminders

### Data Quality
- ✅ **Keep existing 8K**: High-quality vec2text-compatible (port 8767)
- ✅ **New 67K**: Also vec2text-compatible (port 8767)
- ✅ **Sequential order**: Preserved via batch_id + created_at
- ❌ **NEVER factoid-wiki**: Not used, only Wikipedia

### Encoder Usage
- ✅ **Port 8767**: Vec2text-compatible GTR-T5 (CORRECT)
- ❌ **Port 8765**: DEPRECATED sentence-transformers
- Verify: `curl http://localhost:8767/health`

### Training Data Rules (from CLAUDE.md)
- ✅ **Use**: Wikipedia, sequential documents
- ❌ **Never**: Ontologies (WordNet, SWO, GO) for LVM training
- Ontologies are TAXONOMIC, LVMs need SEQUENTIAL data

---

## 🎯 Success Criteria

### Minimum Acceptable
- ✅ Ingestion completes successfully (67K+ concepts)
- ✅ Val cosine > 45% (9x data → 60% improvement)
- ✅ Test cosine > 15% (2-3x improvement)
- ✅ No gibberish output

### Good Performance
- ✅ Val cosine > 50%
- ✅ Test cosine > 20%
- ✅ Semantic coherence preserved

### Excellent Performance
- ✅ Val cosine > 55%
- ✅ Test cosine > 25%
- ✅ Generalizes to diverse domains

---

## 📞 Quick Commands Reference

```bash
# Monitor ingestion
tail -f /tmp/ingest_5000_progress.log

# Check status
/tmp/monitor_ingestion.sh

# Verify completion
tail -50 /tmp/ingest_5000_progress.log | grep "COMPLETE"

# Count concepts
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"

# Extract training data (after ingestion)
python3 tools/extract_ordered_training_data.py --db lnsp --output-dir artifacts/lvm --context-size 5

# Train LSTM (after extraction)
PYTHONPATH=app/lvm python3 app/lvm/train_lstm_baseline.py --data artifacts/lvm/training_sequences_ctx5_75k.npz --epochs 30

# Test model (after training)
PYTHONPATH=app/lvm python3 tools/test_lvm_full_pipeline.py artifacts/lvm/models/lstm_75k/best_model.pt lstm
```

---

**Session Date**: October 13, 2025
**Status**: ✅ 5K Wikipedia ingestion running (PID 5972)
**ETA**: ~11:15 AM (IMPROVED! 21.6 articles/min vs expected 15)
**Progress**: 108/5,000 articles (2.1%), 1,247 concepts
**Monitoring**: `/tmp/live_ingestion_status.sh` (real-time via database)
**Next Action**: Monitor progress, verify completion, extract training data
**Expected Outcome**: 9x dataset improvement → 2-3x LVM test performance
