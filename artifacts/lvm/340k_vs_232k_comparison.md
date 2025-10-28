# 340k vs 232k Training Comparison

**Date:** October 18, 2025
**Training Time:** 1h 15m (all 4 models)
**Device:** Apple M1 Max (MPS)

---

## 📊 Results Comparison

| Model       | 232k Baseline | 340k New | Difference | Change   |
|-------------|---------------|----------|------------|----------|
| Transformer | **0.5820**    | 0.5658   | -0.0162    | **-2.8%** ⬇️ |
| LSTM        | **0.5758**    | 0.5526   | -0.0232    | **-4.0%** ⬇️ |
| GRU         | **0.5754**    | 0.5546   | -0.0208    | **-3.6%** ⬇️ |
| AMN         | **0.5664**    | 0.5405   | -0.0259    | **-4.6%** ⬇️ |

**Average degradation: -3.75%**

---

## 🤔 Unexpected Finding

**The 340k models performed WORSE than the 232k baseline**, despite having:
- ✅ 46% more training data (232k → 340k sequences)
- ✅ Same architecture
- ✅ Same hyperparameters (20 epochs, batch 64, lr 0.0005)
- ✅ Same training procedure

---

## 🔍 Possible Causes

### 1. **Data Quality Issue** (Most Likely)
The 340k dataset includes newly ingested Wikipedia data (Oct 15-18) that may have:
- Lower quality vectors (different encoder settings?)
- Missing CPESH metadata (confirmed - see CLAUDE.md)
- Data corruption during the crash/resume cycles
- Mixed data sources (old 232k + new 107k)

**Evidence:**
- Recent ingestion crashed multiple times at ~article 477
- 107k new concepts added without full CPESH generation
- NPZ file rebuilt from PostgreSQL (potential corruption point)

### 2. **Insufficient Training Epochs**
- 340k dataset is 46% larger → may need more epochs to converge
- 232k baseline might have used more epochs (need to verify)
- Training stopped at 20 epochs (hardcoded limit)

### 3. **Data Distribution Shift**
- New Wikipedia articles (3,432-3,431) may have different characteristics
- Potential domain mismatch between old and new data
- Sequence ordering might be suboptimal

### 4. **Overfitting to Old Data**
- Models might have been trained on 232k data previously
- Loading pretrained weights? (unlikely but possible)

---

## ✅ What Went Right

1. **Training Speed**: Only 1h 15m for all 4 models (faster than expected!)
2. **No Errors**: All models trained successfully to completion
3. **Convergence**: All models showed improvement across epochs
4. **Infrastructure**: Service scripts and checkpoints working perfectly

---

## 🎯 Next Steps

### Option 1: Investigate Data Quality ⭐ Recommended
```bash
# Check vector quality
python tools/verify_vector_quality.py \
  --npz artifacts/wikipedia_500k_corrected_vectors.npz \
  --compare-with <old_232k_vectors.npz>

# Verify data distribution
python tools/analyze_data_distribution.py \
  --old artifacts/old_232k_vectors.npz \
  --new artifacts/wikipedia_500k_corrected_vectors.npz
```

### Option 2: Retrain with More Epochs
```bash
# Try 40 epochs instead of 20
./tools/train_all_lvms_340k.sh --epochs 40
```

### Option 3: Use Only High-Quality Subset
```bash
# Train on first 232k vectors only (verified quality)
python tools/export_lvm_training_data.py \
  --input artifacts/wikipedia_500k_corrected_vectors.npz \
  --max-samples 232600 \
  --output-dir artifacts/lvm/data/

./tools/train_all_lvms_340k.sh
```

### Option 4: Fresh Ingestion with Proper CPESH
```bash
# Restart services
./scripts/stop_all_fastapi_services.sh
sleep 5
./scripts/start_all_fastapi_services.sh

# Delete corrupt data and re-ingest
psql lnsp -c "DELETE FROM cpe_entry WHERE created_at > '2025-10-15';"

# Fresh ingestion with checkpoint system
LNSP_TMD_MODE=hybrid \
LNSP_LLM_ENDPOINT="http://localhost:11434" \
LNSP_LLM_MODEL="llama3.1:8b" \
./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3432 \
  --limit 3000
```

---

## 📈 Training Curves (Need to Extract)

To diagnose the issue, we should plot:
- Training loss over epochs (all models)
- Validation loss over epochs
- Cosine similarity trends
- Compare 232k vs 340k learning curves

---

## 💡 Lessons Learned

1. **More data ≠ always better** - Quality > Quantity
2. **Data validation is critical** - Should verify before training
3. **Keep baseline data pristine** - Don't mix old + new without validation
4. **Monitor ingestion quality** - The crashes were a red flag

---

## 🔬 Recommended Action

**Investigate data quality FIRST** before re-training:

1. Compare vector distributions (old vs new)
2. Check for NaN/Inf values
3. Verify normalization (should all be L2=1.0)
4. Sample check: decode random vectors to text
5. Check FAISS index quality (search precision)

If data is good → retrain with more epochs
If data is bad → fresh ingestion with proper quality checks

---

**Models Saved:**
- `artifacts/lvm/models_340k/amn/`
- `artifacts/lvm/models_340k/lstm/`
- `artifacts/lvm/models_340k/gru/`
- `artifacts/lvm/models_340k/transformer/`

**Baseline Models (Better Performance):**
- Need to locate 232k baseline models for comparison
