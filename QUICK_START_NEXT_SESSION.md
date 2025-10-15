# Quick Start - Next Session
## October 12, 2025

## 🎯 TL;DR

**ROOT CAUSE**: Sentence-transformers GTR-T5 embeddings are INCOMPATIBLE with vec2text (cosine 0.076 instead of 0.63+)

**SOLUTION**: Use vec2text's own encoder for all embeddings

## ⚡ Quick Commands

### Start Services
```bash
# Vec2text server
VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766 &

# Vec2Text-compatible GTR-T5 encoder
./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767 &
```

### Test Vec2Text Compatibility
```bash
# Test vec2text roundtrip (should be 0.63+)
python3 -c "
import requests
r = requests.post('http://localhost:8766/encode-decode', 
                  json={'texts': ['The Earth is round'], 'steps': 1})
print('Cosine:', r.json()['results'][0]['subscribers']['gtr → jxe']['cosine'])
"
```

### Check Training Data
```bash
# Verify training data exists and check dimensions
python3 -c "
import numpy as np
data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
print('Target vectors:', data['target_vectors'].shape)
print('Sample norm:', np.linalg.norm(data['target_vectors'][0]))
"
```

## 📋 Implementation Checklist

### Phase 1: Fix Embedder ⏳
- [ ] Modify `app/api/gtr_embedding_server.py` to use vec2text encoder
- [ ] Test: GTR-T5 API → vec2text (should get 0.63+ cosine)

### Phase 2: Regenerate Data ⏳  
- [ ] Clear all databases: `python tools/nuclear_clear_all_databases.py`
- [ ] Re-ingest 100 test chunks
- [ ] Verify: database vectors → vec2text (should get 0.63+ cosine)
- [ ] Full data ingestion

### Phase 3: Retrain Models ⏳
- [ ] Extract new training sequences
- [ ] Train LSTM: `python app/lvm/train_lstm_baseline.py --epochs 20`
- [ ] Train GRU: `python app/lvm/train_mamba2.py --epochs 20`
- [ ] Train Transformer: `python app/lvm/train_transformer.py --epochs 20`

### Phase 4: Validate ⏳
- [ ] Test LVM → vec2text pipeline
- [ ] Expected: cosine 0.60-0.70, sensible decoded text

## 📁 Key Files

**Documentation**:
- `LVM_TEST_SUMMARY_FINAL.md` - Executive summary
- `LVM_TRAINING_RESULTS_OCT12.md` - Detailed implementation plan

**Tools Created**:
- `tools/nuclear_clear_all_databases.py` - Clear all data
- `tools/test_vec2text_roundtrip.py` - Test roundtrip
- `tools/test_lvm_vec2text_pipeline.py` - Test full pipeline

**Fixed Training Scripts**:
- `app/lvm/train_lstm_baseline.py` - Now outputs norm=1.0 ✅
- `app/lvm/train_mamba2.py` - Now outputs norm=1.0 ✅  
- `app/lvm/train_transformer.py` - Now outputs norm=1.0 ✅

## 🔬 What We Learned

1. **Library compatibility ≠ Model compatibility**
   - Same model (`gtr-t5-base`) in different libraries = incompatible embeddings
   
2. **Always test the full pipeline early**
   - We trained models before discovering embedding incompatibility
   
3. **Library consistency > Model consistency**
   - Use vec2text encoder for: ingestion → training → inference → decoding

## 🚨 Critical Insight

**Vec2text roundtrip (cosine 0.63)** ✅
```
Text → vec2text.encode() → vec → vec2text.decode() → Text
```

**Sentence-transformers (cosine 0.076)** ❌
```
Text → SentenceTransformer.encode() → vec → vec2text.decode() → Nonsense
```

**Both claim to use GTR-T5 with mean pooling and L2 norm, but produce different embeddings!**

## ⚡ Next Action

**Start with Phase 1**: Update GTR-T5 API to use vec2text encoder (see `LVM_TRAINING_RESULTS_OCT12.md` for code)
