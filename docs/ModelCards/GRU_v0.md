# Model Card — GRU_v0

**Model Type:** Gated Recurrent Unit (Single-Tower)
**Version:** v0
**Date:** Oct 28, 2025
**Owner:** Retrieval Platform

---

## Overview

GRU_v0 is the **fallback production model** for Release v0, selected for tasks requiring maximum accuracy. It achieves the highest in-distribution performance while maintaining excellent OOD generalization, making it ideal for batch processing and accuracy-critical applications.

---

## Architecture

- **Type:** Single-tower encoder (Gated Recurrent Unit)
- **Input:** 5 × 768D context vectors (sequence of embeddings)
- **Output:** 768D prediction vector
- **Parameters:** 7.1M (2nd largest in fleet)
- **Model Size:** 28.3 MB (disk)
- **Layers:** 2-layer bidirectional GRU with residual connections

---

## Performance Metrics

### Validation Cosine Similarity
- **In-Distribution:** 0.5920 ± 0.1235 **(best accuracy)**
- **OOD (Zero-Shot Article):** 0.6295 ± 0.1071 (excellent generalization)
- **Notes:** Highest in-domain accuracy, very close to AMN on OOD (-0.8pp)

### Latency (Apple M1 Pro, CPU)
- **Mean:** 2.11 ms
- **P95:** 2.87 ms
- **P99:** 3.45 ms
- **Throughput:** ~475 queries/sec (single-threaded)

### Memory
- **Model Size:** 28.3 MB
- **Runtime Peak:** ~45 MB (inference, batch=1)
- **VRAM:** 0 MB (CPU-only deployment)

---

## Training Details

- **Dataset:** Wikipedia 80k sequences (5 context → 1 target)
- **Loss:** MSE (Mean Squared Error)
- **Epochs:** 30
- **Batch Size:** 512
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.01)
- **Device:** MPS (Apple Silicon)
- **Training Time:** ~65 minutes
- **Checkpoint:** `artifacts/lvm/models/gru_v0.pt`

---

## Use Cases

### ✅ Primary Use Case (Fallback Production)
- Batch processing (latency not critical)
- Maximum accuracy requirements
- In-domain tasks where training data is representative
- Offline analytics and reporting

### ⚠️ Not Recommended For
- High-throughput serving (use AMN_v0 instead)
- Memory-constrained environments (28.3 MB vs AMN's 5.8 MB)
- Sub-millisecond latency requirements

---

## Deployment

### Python Inference
```python
import torch
from app.lvm.model import GRUEncoder

# Load model
device = torch.device('cpu')  # or 'mps', 'cuda'
model = GRUEncoder(input_dim=768, hidden_dim=1024, output_dim=768, num_layers=2)
model.load_state_dict(torch.load('artifacts/lvm/models/gru_v0.pt', map_location=device))
model.eval()

# Predict next vector
context = torch.randn(1, 5, 768)  # (batch, sequence, dims)
with torch.no_grad():
    prediction = model(context)  # (batch, 768)
```

### Batch Processing Pipeline
```python
import torch
import numpy as np
from app.lvm.model import GRUEncoder

def batch_predict(contexts: np.ndarray, model: GRUEncoder, batch_size: int = 128):
    """
    Efficient batch prediction with GRU_v0

    Args:
        contexts: (N, 5, 768) array of context sequences
        model: Loaded GRU model
        batch_size: Batch size for processing

    Returns:
        predictions: (N, 768) array of predicted vectors
    """
    model.eval()
    predictions = []

    for i in range(0, len(contexts), batch_size):
        batch = torch.tensor(contexts[i:i+batch_size], dtype=torch.float32)
        with torch.no_grad():
            pred = model(batch)
        predictions.append(pred.numpy())

    return np.vstack(predictions)
```

---

## Comparison with Alternatives

| Model        | OOD Cosine | Latency (ms) | Size (MB) | Params  | Recommendation         |
|--------------|------------|--------------|-----------|---------|------------------------|
| AMN_v0       | 0.6375     | 0.62         | 5.8       | 1.5M    | Primary (default)      |
| **GRU_v0**   | **0.6295** | 2.11         | 28.3      | 7.1M    | **Fallback (accuracy)**|
| LSTM_v0      | 0.6281     | 0.56         | 14.1      | 3.5M    | Archived (redundant)   |
| Transformer  | 0.6042     | 2.68         | 23.7      | 5.9M    | Archived (slower)      |

**Decision:** GRU_v0 is fallback due to:
- Best in-distribution accuracy (0.5920)
- Excellent OOD generalization (0.6295)
- Acceptable latency for batch use (2.11 ms)

---

## Known Limitations

1. **Latency:** 3.4x slower than AMN_v0 (2.11 ms vs 0.62 ms)
   - **Mitigation:** Use for batch processing only, not real-time serving

2. **Memory:** 4.9x larger than AMN_v0 (28.3 MB vs 5.8 MB)
   - **Mitigation:** Acceptable for server deployment, avoid edge/mobile

3. **OOD Performance:** 0.8pp lower than AMN_v0 (0.6295 vs 0.6375)
   - **Mitigation:** Still excellent generalization, minimal practical impact

4. **Sequence Length:** Fixed K=5 context window
   - **Mitigation:** Pad/truncate sequences to exactly 5 vectors

---

## When to Use GRU_v0 vs AMN_v0

### Use GRU_v0 If:
- ✅ Accuracy is paramount (in-domain tasks)
- ✅ Latency budget > 2ms per query
- ✅ Batch processing (throughput not critical)
- ✅ Server deployment (memory available)

### Use AMN_v0 If:
- ✅ Latency budget < 1ms per query
- ✅ High throughput required (>1000 QPS)
- ✅ Memory constrained (edge/mobile)
- ✅ OOD generalization critical

### Performance Trade-off
```
AMN_v0: Fast (0.62ms) + Small (5.8MB) + Best OOD (0.6375)
GRU_v0: Slower (2.11ms) + Larger (28.3MB) + Best Accuracy (0.5920)
```

---

## Versioning

- **v0 (Current):** Initial production release (Oct 28, 2025)
- **Future:** v1 may incorporate dynamic sequence length + attention mechanisms

---

## Monitoring

### Key Metrics to Track
- **Latency:** P95 should stay ≤ 3.0 ms
- **Throughput:** ≥ 400 QPS (single-threaded, batch processing)
- **Cosine Similarity:** Mean ≥ 0.62 on OOD eval
- **Memory:** Peak ≤ 50 MB per worker

### Alerts
- P95 latency > 4.0 ms → investigate load/hardware
- Mean cosine < 0.60 → model degradation check
- Memory > 60 MB → memory leak investigation
- Batch processing time > expected → check data loading

---

## Benchmarks

### Batch Processing Performance
| Batch Size | Time (sec) | Throughput (seq/sec) | Memory (MB) |
|------------|------------|----------------------|-------------|
| 1          | 0.00211    | 474                  | 45          |
| 32         | 0.0421     | 760                  | 87          |
| 128        | 0.1523     | 841                  | 189         |
| 512        | 0.5892     | 869                  | 512         |

**Optimal Batch Size:** 128 (best throughput/memory trade-off)

---

## References

- **Training Code:** `app/lvm/train_model.py`
- **Model Definition:** `app/lvm/model.py`
- **Benchmarks:** `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`
- **Release Notes:** `docs/PROD/Release_v0_Retriever.md`
- **Primary Model:** `docs/ModelCards/AMN_v0.md`

---

## Contact

For issues or questions, contact Retrieval Platform team.
