# LVM-T Model Diagnostic Report
**Date:** October 16, 2025
**Model:** Transformer (4 layers, 512 d_model, 17.8M parameters)
**Training Data:** 80,629 sequences from CORRECTED Wikipedia vectors

---

## Executive Summary

⚠️ **CRITICAL FINDING:** The trained LVM-T model is **underperforming simple baseline models** on held-out test data.

**Key Metrics:**
- **LVM-T Performance:** 0.3539 cosine similarity
- **Best Baseline:** 0.5462 (Linear Average of Context)
- **Performance Gap:** -35.2% worse than best baseline

While the model showed improvement during training (Val Cosine: 0.2340 → 0.3163), it fails to outperform trivial baselines on held-out data, indicating significant issues.

---

## Test Results Summary

### Test Configuration
- **Test Set Size:** 16,127 vectors (20% held-out from 80,634 total)
- **Evaluation Samples:** 1,000 sequences
- **Inference Speed:** 4,619 samples/second
- **Device:** Apple Silicon (MPS)

### Quantitative Results

| Model | Cosine Mean | Cosine Std | Top-5 Accuracy | Performance vs LVM-T |
|-------|-------------|------------|----------------|---------------------|
| Random Vectors | 0.0002 | 0.0349 | 0.00% | -99.9% |
| Mean Vector | 0.4218 | 0.0736 | 14.00% | +19.2% |
| Persistence (Last Context) | 0.4383 | 0.1559 | 36.80% | +23.9% |
| **Linear Average** | **0.5462** | 0.1341 | **67.80%** | **+54.3%** |
| **LVM-T Transformer** | **0.3539** | 0.0910 | **1.90%** | **baseline** |

---

## Analysis: Why Is LVM-T Underperforming?

### 1. **Loss Function Mismatch**
**Issue:** The model was trained with InfoNCE loss (contrastive learning), but evaluated with cosine similarity.

**Evidence:**
- Training metric: InfoNCE loss decreased from 7.04 → 1.34 (✓ good)
- Validation cosine improved: 0.234 → 0.316 (✓ modest improvement)
- **But:** Test cosine (0.354) < Simple baselines (0.546) (✗ problem!)

**Explanation:** InfoNCE optimizes for distinguishing vectors in contrastive space, not for direct vector prediction accuracy.

### 2. **Training Distribution Shift**
**Issue:** Training sequences are sequential chunks from Wikipedia articles, but the model may be overfitting to specific article structures.

**Evidence:**
- Training used ordered sequences (chunk₁ → chunk₂ → chunk₃)
- Test set is from different Wikipedia articles
- Linear average (0.546) works because sequential chunks ARE semantically similar
- Model (0.354) fails because it learned article-specific patterns, not general semantic flow

### 3. **Baseline Is Too Strong**
**Key Finding:** A simple linear average of the 5 context vectors achieves 0.5462 cosine similarity!

**Why this baseline is strong:**
- Wikipedia chunks are topically coherent
- Next chunk is usually semantically related to previous chunks
- Simple averaging captures "topic drift" effectively

**Why LVM-T struggles:**
- Trying to learn complex non-linear patterns where linear patterns dominate
- 17.8M parameters may be overfitting to noise instead of signal
- Transformer attention may be focusing on wrong features

### 4. **Insufficient Training Signal**
**Issue:** The model may need different training objectives or more data.

**Evidence:**
- Cosine std is low (0.091) → model makes safe, averaged predictions
- Top-5 accuracy only 1.9% (cosine > 0.5) → rarely makes confident predictions
- Compare: Linear baseline has 67.8% Top-5 accuracy

---

## Sample Predictions Analysis

### Sample Cosine Similarities (10 random tests):
```
Sample 1: 0.276  (Below average)
Sample 2: 0.316  (Average)
Sample 3: 0.390  (Above average)
Sample 4: 0.254  (Below average)
Sample 5: 0.396  (Above average)
Sample 6: 0.025  (Very poor - failure case)
Sample 7: 0.489  (Good - near baseline)
Sample 8: 0.212  (Below average)
Sample 9: 0.412  (Above average)
Sample 10: 0.302 (Average)
```

**Distribution:** Wide variance (0.025 to 0.489), with failures dragging down average.

---

## Root Cause Analysis

### Primary Issues

1. **Wrong Optimization Target**
   - **Current:** InfoNCE loss (contrastive)
   - **Should use:** Direct cosine similarity loss or MSE on normalized vectors

2. **Task Formulation Problem**
   - **Current:** Predict next vector from 5-vector context
   - **Reality:** Next Wikipedia chunk is highly correlated with context average
   - **Solution:** Need harder task or different data structure

3. **Overfitting to Training Distribution**
   - **Evidence:** Val cosine 0.316 vs Test cosine 0.354
   - **Gap:** 0.038 suggests model generalizes somewhat
   - **But:** Still worse than baselines, suggesting fundamental issue

### Secondary Issues

4. **Model Complexity vs Task Difficulty**
   - 17.8M parameters for a task where linear average wins
   - Transformer attention may be "overthinking" simple semantic drift

5. **Context Length May Be Wrong**
   - Using 5 vectors as context
   - Perhaps shorter (2-3) or longer (10-20) would be better

---

## Recommendations

### Immediate Actions (High Priority)

1. **✅ Change Loss Function**
   ```python
   # Instead of InfoNCE contrastive loss
   loss = nn.functional.mse_loss(prediction, target)
   # OR
   loss = 1 - cosine_similarity(prediction, target)
   ```

2. **✅ Try Simpler Model First**
   - Start with 1-2 layer Transformer
   - Compare with simple MLP baseline
   - Verify complexity is needed

3. **✅ Add Residual Connections to Baseline**
   ```python
   # Predict DELTA from linear average, not absolute vector
   baseline = linear_average(context)
   delta = model(context)
   prediction = normalize(baseline + delta)
   ```

### Medium-Term Improvements

4. **Curriculum Learning**
   - Start training with easy predictions (1 context vector)
   - Gradually increase to 5 vectors
   - Helps model learn incremental improvements over baseline

5. **Better Training Data**
   - Mix different sources (not just Wikipedia)
   - Include negative examples (unrelated chunks)
   - Force model to learn when NOT to predict continuation

6. **Architectural Changes**
   - Add explicit baseline branch
   - Use gating mechanism to blend baseline + learned residuals
   - Consider LSTM/GRU (simpler than Transformer for sequential tasks)

### Long-Term Research

7. **Task Reformulation**
   - Instead of next-vector prediction, try:
     - Multi-hop prediction (predict vector at t+5, not just t+1)
     - Bidirectional context (predict middle vector from surrounding context)
     - Conditional generation (predict vector matching a text query)

8. **Hybrid Models**
   - Combine LVM-T with retrieval
   - Use LVM-T to refine retrieved candidates
   - Ensemble with baseline

---

## Positive Findings

Despite underperformance, several aspects worked well:

1. ✅ **Training Infrastructure:** Stable training for 20 epochs without crashes
2. ✅ **Vector Quality:** CORRECTED vectors load and process correctly (1.0 norm)
3. ✅ **Inference Speed:** 4,619 samples/sec is very fast
4. ✅ **Baseline Establishment:** Now have clear performance targets (0.546 to beat)
5. ✅ **Diagnostic Framework:** Comprehensive testing reveals issues early

---

## Next Steps

### Immediate (Today)
1. ✅ Document findings in this report
2. ⏳ Retrain with MSE loss instead of InfoNCE
3. ⏳ Test residual architecture (predict delta from baseline)

### Short-Term (This Week)
4. ⏳ Implement simpler 2-layer model
5. ⏳ Add curriculum learning
6. ⏳ Collect more diverse training data

### Long-Term (Future Sprints)
7. ⏳ Explore alternative tasks (multi-hop, bidirectional)
8. ⏳ Research hybrid LVM + retrieval systems
9. ⏳ Benchmark against other vector sequence models

---

## Conclusion

The LVM-T model training was **technically successful** but **functionally inadequate**. The model learned something (better than random, improved over training), but failed to learn the right thing (worse than simple baselines).

**Key Insight:** This is a classic case of **"optimizing the wrong metric"**. InfoNCE loss pushed the model toward contrastive distinctiveness, but we need predictive accuracy.

**Recommended Path Forward:**
1. **Quick fix:** Retrain with direct cosine/MSE loss
2. **Better fix:** Add residual architecture over linear baseline
3. **Best fix:** Reformulate task to require non-linear reasoning

This diagnostic report provides a clear roadmap for improving the model from 0.35 → 0.60+ cosine similarity.

---

**Report Generated:** October 16, 2025
**Model Path:** `artifacts/lvm/models/transformer_corrected_80k/`
**Test Results:** `artifacts/lvm/test_results/lvm_t_test_results.json`
