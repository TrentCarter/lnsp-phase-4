# GraphMERT-LVM vs Other LVM Models - Comparison

**Date:** 2025-10-17
**Models Tested:** 5
**Test Data:** 80k Wikipedia training sequences

---

## 📊 Performance Comparison

| Rank | Model | Text Cosine | Train Val Cosine | Latency (ms) | Params | Memory (MB) |
|------|-------|-------------|------------------|--------------|--------|-------------|
| 🥇 | **AMN** | 0.8046 | 0.5664 | 0.62 | 1.5M | 5.8 |
| 🥈 | **GRU** | 0.6451 | 0.5754 | 2.17 | 7.1M | 27.1 |
| 🥉 | **Transformer** | 0.4823 | 0.5820 | 2.74 | 17.9M | 68.2 |
| 4. | **LSTM** | 0.4189 | 0.5758 | 0.70 | 5.1M | 19.5 |
| 5. | **GraphMERT-LVM-80k** | 0.4119 | 0.5783 | 6.67 | 67.4M | 256.9 |

**Key Metrics:**
- **Text Cosine**: Real-time text→vec→LVM→vec cosine similarity (higher = better)
- **Train Val Cosine**: Validation cosine from training (reference)
- **Latency**: Mean inference time per query

---

## 🔍 Detailed Analysis

### AMN

**Performance:**
- Text quality (avg cosine): 0.8046
- Validation cosine (training): 0.5664
- Inference latency: 0.62 ms (±0.54)
- P95 latency: 0.68 ms

**Resources:**
- Parameters: 1,510,912
- Memory footprint: 5.8 MB

**Sample Outputs** (first 3 examples):
1. Input: "Artificial intelligence is transforming modern technology...."
   Cosine: 0.8725
2. Input: "The quick brown fox jumps over the lazy dog...."
   Cosine: 0.7732
3. Input: "Machine learning models learn patterns from data...."
   Cosine: 0.8077

---

### GRU

**Performance:**
- Text quality (avg cosine): 0.6451
- Validation cosine (training): 0.5754
- Inference latency: 2.17 ms (±0.56)
- P95 latency: 2.65 ms

**Resources:**
- Parameters: 7,095,552
- Memory footprint: 27.1 MB

**Sample Outputs** (first 3 examples):
1. Input: "Artificial intelligence is transforming modern technology...."
   Cosine: 0.7119
2. Input: "The quick brown fox jumps over the lazy dog...."
   Cosine: 0.5801
3. Input: "Machine learning models learn patterns from data...."
   Cosine: 0.6929

---

### Transformer

**Performance:**
- Text quality (avg cosine): 0.4823
- Validation cosine (training): 0.5820
- Inference latency: 2.74 ms (±0.73)
- P95 latency: 3.46 ms

**Resources:**
- Parameters: 17,867,520
- Memory footprint: 68.2 MB

**Sample Outputs** (first 3 examples):
1. Input: "Artificial intelligence is transforming modern technology...."
   Cosine: 0.5531
2. Input: "The quick brown fox jumps over the lazy dog...."
   Cosine: 0.4754
3. Input: "Machine learning models learn patterns from data...."
   Cosine: 0.4570

---

### LSTM

**Performance:**
- Text quality (avg cosine): 0.4189
- Validation cosine (training): 0.5758
- Inference latency: 0.70 ms (±0.22)
- P95 latency: 0.95 ms

**Resources:**
- Parameters: 5,120,768
- Memory footprint: 19.5 MB

**Sample Outputs** (first 3 examples):
1. Input: "Artificial intelligence is transforming modern technology...."
   Cosine: 0.4258
2. Input: "The quick brown fox jumps over the lazy dog...."
   Cosine: 0.3572
3. Input: "Machine learning models learn patterns from data...."
   Cosine: 0.4251

---

### GraphMERT-LVM-80k

**Performance:**
- Text quality (avg cosine): 0.4119
- Validation cosine (training): 0.5783 (best at epoch 8, overfitting after)
- Inference latency: 6.67 ms (±1.65)
- P95 latency: 9.47 ms

**Resources:**
- Parameters: 67,352,833
- Memory footprint: 256.9 MB

**Sample Outputs** (first 3 examples):
1. Input: "Artificial intelligence is transforming modern technology...."
   Cosine: 0.5751
2. Input: "The quick brown fox jumps over the lazy dog...."
   Cosine: 0.3499
3. Input: "Machine learning models learn patterns from data...."
   Cosine: 0.4438

---

## 💡 Key Insights

1. **Highest Accuracy**: AMN (0.8046 cosine)
2. **Fastest Inference**: AMN (0.62 ms/query)
3. **Smallest Model**: AMN (5.8 MB)

### GraphMERT-LVM Neurosymbolic Features

GraphMERT-LVM combines:
- **Neural**: Autoregressive vector prediction (like other LVMs)
- **Symbolic**: Knowledge graph entity relationships via leafy chain graphs
- **Architecture**: 12 layers, 8 heads, 67M parameters
- **Performance**: 0.4119 cosine, 6.67 ms latency

**Standard LVM Advantage**: AMN leads by 0.3928 cosine

---

**Generated:** 2025-10-17
**Tool:** `tools/compare_graphmert_with_other_lvms.py`
