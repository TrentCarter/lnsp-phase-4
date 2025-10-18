# GraphMERT-LVM Benchmark Summary (2025-10-17)

## Executive Summary

GraphMERT-LVM (neurosymbolic vector language model) was successfully trained and benchmarked against 4 standard LVM models (AMN, LSTM, GRU, Transformer). **Surprisingly, the standard models outperformed GraphMERT-LVM** in the current test scenario.

## Results Overview

### 🏆 Overall Rankings (by Text Quality)

| Rank | Model | Text Cosine | Train Val | Latency (ms) | Params | Memory (MB) |
|------|-------|-------------|-----------|--------------|--------|-------------|
| 🥇 | **AMN** | **0.8046** | 0.5664 | 0.62 | 1.5M | 5.8 |
| 🥈 | **GRU** | **0.6451** | 0.5754 | 2.17 | 7.1M | 27.1 |
| 🥉 | **Transformer** | **0.4823** | 0.5820 | 2.74 | 17.9M | 68.2 |
| 4. | **LSTM** | **0.4189** | 0.5758 | 0.70 | 5.1M | 19.5 |
| 5. | **GraphMERT-LVM** | **0.4119** | 0.5783 | 6.67 | 67.4M | 256.9 |

**Key Metrics:**
- **Text Cosine**: Real-time text→vec→LVM→vec cosine similarity
- **Train Val**: Validation cosine from training
- **Latency**: Mean inference time per query

## Key Findings

### 1. AMN is the Clear Winner 🥇

- **Best text quality**: 0.8046 cosine (95% higher than GraphMERT)
- **Fastest inference**: 0.62 ms/query (10.7x faster than GraphMERT)
- **Smallest model**: 5.8 MB (44x smaller than GraphMERT)
- **Most efficient**: 1.5M parameters vs 67M for GraphMERT

**Why AMN performed so well:**
- Simplest architecture (Additive Memory Network)
- Well-suited for the test scenario (repeating input vectors)
- No overfitting issues
- Excellent generalization

### 2. GraphMERT-LVM Underperformed

**Issues Identified:**
- **Text quality**: 0.4119 cosine (lowest among all models)
- **Latency**: 6.67 ms/query (slowest by 2.4x)
- **Size**: 256.9 MB (largest model)
- **Overfitting**: Best validation cosine at epoch 8 (0.5783), degraded to 0.5499 by epoch 25

**Possible Reasons:**
1. **Test methodology limitation**: Using `.repeat(1, 5, 1)` creates unrealistic context (identical vectors)
2. **Architecture complexity**: 67M parameters may be overkill for this task
3. **Overfitting**: Continued training past optimal point
4. **Neurosymbolic benefits not realized**: Leafy chain graphs may not help for simple vector prediction

### 3. Training vs. Inference Performance Gap

All models show a **significant gap** between training validation cosine and real-time text quality:

| Model | Train Val Cosine | Text Cosine | Gap |
|-------|------------------|-------------|-----|
| AMN | 0.5664 | **0.8046** | **+42%** ✅ |
| GRU | 0.5754 | 0.6451 | +12% |
| Transformer | 0.5820 | 0.4823 | -17% ❌ |
| LSTM | 0.5758 | 0.4189 | -27% ❌ |
| GraphMERT | 0.5783 | 0.4119 | -29% ❌ |

**Observation**: AMN actually **improves** in real-world scenario, while others (including GraphMERT) degrade significantly.

## GraphMERT-LVM Training Details

### Architecture
- **Model**: GraphMERTLVM768D (neurosymbolic)
- **Layers**: 12 transformer layers
- **Heads**: 8 attention heads
- **Hidden dim**: 768D (d_model)
- **FFN dim**: 2048D
- **Parameters**: 67,352,833
- **Training data**: 80,629 Wikipedia sequences

### Training Results
- **Best validation cosine**: 0.5783 (epoch 8)
- **Final validation cosine**: 0.5499 (epoch 25)
- **Training time**: ~31 minutes (25 epochs × 75s/epoch)
- **Overfitting detected**: Performance degraded after epoch 8

### Neurosymbolic Components
- **Leafy chain graphs**: 80,629 graphs built (456 MB)
- **Entity pool**: 80,634 Wikipedia concepts
- **Entity linking**: α-filtering with 0.55 threshold
- **Average leaves**: 13.15 entities per sequence
- **Relation types**: topical, contextual, related_to, sequential

## Recommendations

### For Production Use: **AMN** ⭐
- Highest quality (0.8046 cosine)
- Fastest inference (0.62 ms)
- Smallest footprint (5.8 MB)
- Best efficiency score

### For GraphMERT-LVM Improvement

1. **Stop training at epoch 8** (before overfitting)
2. **Use realistic test context**:
   - Replace `.repeat(1, 5, 1)` with actual sequential vectors from Wikipedia
   - Test with real multi-sentence sequences
3. **Simplify architecture**:
   - Reduce from 12 to 6-8 layers
   - Reduce parameters to ~20M
4. **Investigate leafy chain graph utility**:
   - May not help for pure vector prediction
   - Better suited for knowledge-intensive tasks?
5. **Train with more data**:
   - Currently 80k sequences
   - Could benefit from 200k+ sequences (overnight ingestion planned)

## Test Methodology Concerns

**Current approach:**
```python
# Creates IDENTICAL context vectors (unrealistic)
context = torch.FloatTensor(input_vector).unsqueeze(0).repeat(1, 5, 1)
```

**Better approach:**
```python
# Use actual sequential context from Wikipedia
# context[0] = sentence 1 vector
# context[1] = sentence 2 vector
# ...
# context[4] = sentence 5 vector
# Predict: sentence 6 vector
```

This would better reflect real-world usage and might favor GraphMERT's neurosymbolic architecture.

## Next Steps

1. ✅ **Complete comparison benchmark** (DONE)
2. ✅ **Identify overfitting in GraphMERT** (DONE - epoch 8 optimal)
3. 🔄 **Resume Wikipedia ingestion** (tonight - 7k more articles)
4. ⏭️ **Retrain GraphMERT with**:
   - Larger dataset (240k+ sequences)
   - Early stopping at optimal epoch
   - Simplified architecture
5. ⏭️ **Improve test methodology** with realistic sequential context

## Files Generated

- **Comparison report**: `artifacts/lvm/GRAPHMERT_COMPARISON.md`
- **Benchmark script**: `tools/compare_graphmert_with_other_lvms.py`
- **Leafy chain graphs**: `artifacts/graphmert_lvm/leafy_chain_graphs_80k.npz` (456 MB)
- **Trained model**: `artifacts/lvm/models/graphmert_lvm_80k_full/benchmark_model.pt` (771 MB)

## Conclusion

While GraphMERT-LVM's neurosymbolic architecture is theoretically promising, **AMN currently provides the best performance** for vector language modeling tasks. GraphMERT-LVM needs:
1. Better training (stop at epoch 8)
2. More realistic evaluation (sequential context)
3. Architectural simplification
4. Larger training dataset

The surprising dominance of AMN suggests that **simpler architectures can be more effective** for this task, at least with the current dataset size and test methodology.

---

**Generated**: 2025-10-17
**Benchmark Tool**: `tools/compare_graphmert_with_other_lvms.py`
**Training Data**: 80k Wikipedia sequences
**Test Device**: Apple M1 Max (MPS)
