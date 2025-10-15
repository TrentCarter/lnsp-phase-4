# PRD: LVM Testing & Evaluation Plan
**Created**: October 12, 2025
**Status**: Ready to Execute
**Models**: LSTM (5.1M), GRU (7.1M), Transformer (17.6M)

---

## 🎯 Test Objectives

### Primary Goals
1. **Validate Performance** - Confirm models achieve >75% cosine similarity
2. **Compare Architectures** - Identify best model for production
3. **Measure Retrieval Quality** - Test Top-K accuracy with FAISS
4. **Benchmark Speed** - Measure inference latency and throughput
5. **Assess Production Readiness** - Check deployment requirements

### Secondary Goals
6. **Understand Failure Modes** - Analyze when predictions fail
7. **Test Generalization** - Evaluate on out-of-domain data
8. **Optimize Inference** - Quantize/prune for deployment
9. **Integration Testing** - Connect with vec2text pipeline

---

## 📋 Phase 1: Basic Validation (COMPLETE FOR ALL MODELS)

### Test 1.1: Model Loading ✅
**Status**: PASSED for LSTM, GRU, Transformer

```bash
# Load each checkpoint
for model in lstm_baseline mamba2 transformer; do
    python -c "
import torch
checkpoint = torch.load('artifacts/lvm/models/${model}/best_model.pt')
print(f'${model}: Epoch {checkpoint[\"epoch\"]}, Loss {checkpoint[\"val_loss\"]:.6f}')
"
done
```

**Expected Results**:
- ✅ All models load without errors
- ✅ Checkpoints contain required keys (model_state_dict, val_loss, etc.)
- ✅ Model architectures match saved configs

---

### Test 1.2: Inference on Validation Set ✅
**Status**: PASSED

**Results**:
| Model | Val Loss | Val Cosine | Pass/Fail |
|-------|----------|------------|-----------|
| LSTM | 0.000504 | 78.30% | ✅ PASS (>75%) |
| GRU | 0.000503 | 78.33% | ✅ PASS (>75%) |
| Transformer | 0.000498 | 78.60% | ✅ PASS (>75%) |

---

### Test 1.3: Inference Speed Benchmark
**Status**: PENDING

**Test Script**:
```python
import time
import torch
from train_lstm_baseline import LSTMVectorPredictor

model = LSTMVectorPredictor().to('mps')
model.load_state_dict(torch.load('artifacts/lvm/models/lstm_baseline/best_model.pt')['model_state_dict'])
model.eval()

# Warmup
dummy_input = torch.randn(32, 5, 768).to('mps')
for _ in range(10):
    _ = model(dummy_input)

# Benchmark
batch_sizes = [1, 8, 16, 32, 64]
for bs in batch_sizes:
    inputs = torch.randn(bs, 5, 768).to('mps')
    torch.mps.synchronize()

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(inputs)
    torch.mps.synchronize()
    elapsed = time.time() - start

    print(f"Batch size {bs}: {elapsed/100*1000:.2f}ms/batch ({bs/(elapsed/100):.1f} samples/sec)")
```

**Expected Results**:
- Batch size 1: <5ms/sample (real-time capable)
- Batch size 32: <50ms/batch (>600 samples/sec)
- Transformer should be 2-3x slower than LSTM

---

### Test 1.4: Memory Profiling
**Status**: PENDING

**Test Script**:
```python
import torch
import tracemalloc

for model_name in ['lstm_baseline', 'mamba2', 'transformer']:
    tracemalloc.start()

    # Load model
    if model_name == 'lstm_baseline':
        from train_lstm_baseline import LSTMVectorPredictor
        model = LSTMVectorPredictor().to('mps')
    # ... load model ...

    # Inference
    dummy_input = torch.randn(32, 5, 768).to('mps')
    _ = model(dummy_input)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{model_name}: Peak memory: {peak / 1024**2:.2f} MB")
```

**Expected Results**:
- LSTM: <200MB peak memory
- GRU: <300MB peak memory
- Transformer: <500MB peak memory

---

## 📋 Phase 2: Retrieval Evaluation (CRITICAL)

### Test 2.1: Top-K Accuracy with FAISS
**Status**: PENDING - **HIGHEST PRIORITY**

**Purpose**: Measure how often the target vector is in top-K nearest neighbors of prediction

**Test Script**:
```python
import numpy as np
import torch
import faiss

# Load all vectors
all_vectors_data = np.load('artifacts/lvm/wikipedia_42113_ordered.npz')
all_vectors = all_vectors_data['vectors']  # [42113, 768]
concept_texts = all_vectors_data['concept_texts']

# Build FAISS index
index = faiss.IndexFlatIP(768)  # Inner product (cosine after normalization)
# Normalize vectors
faiss.normalize_L2(all_vectors)
index.add(all_vectors.astype('float32'))

# Load model
model = ...  # Load trained model
model.eval()

# Test on validation set
val_data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
contexts = val_data['context_sequences'][:100]  # Test on 100 samples
targets = val_data['target_vectors'][:100]

results = {k: 0 for k in [1, 5, 10, 20, 50]}

for context, target in zip(contexts, targets):
    # Predict
    with torch.no_grad():
        pred = model(torch.FloatTensor(context).unsqueeze(0)).numpy()

    # Normalize prediction
    faiss.normalize_L2(pred)

    # Find K nearest neighbors
    D, I = index.search(pred.astype('float32'), 50)

    # Normalize target and find its index in database
    target_norm = target / np.linalg.norm(target)
    _, target_idx = index.search(target_norm.reshape(1, -1).astype('float32'), 1)
    target_idx = target_idx[0][0]

    # Check if target is in top-K
    for k in results.keys():
        if target_idx in I[0][:k]:
            results[k] += 1

# Compute percentages
for k in results.keys():
    results[k] = (results[k] / 100) * 100
    print(f"Top-{k} Accuracy: {results[k]:.1f}%")
```

**Expected Results**:
| Model | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 |
|-------|-------|-------|--------|--------|--------|
| LSTM | 15-25% | 40-50% | 55-65% | 70-80% | 85-95% |
| GRU | 15-25% | 40-50% | 55-65% | 70-80% | 85-95% |
| Transformer | 20-30% | 45-55% | 60-70% | 75-85% | 90-95% |

**Success Criteria**:
- ✅ Top-20 accuracy > 70%
- ✅ Top-50 accuracy > 85%

---

### Test 2.2: Semantic Coherence Analysis
**Status**: PENDING

**Purpose**: Visualize predictions vs targets in vector space

**Test Script**:
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Collect predictions and targets
predictions = []
targets_list = []

for context, target in zip(val_contexts[:500], val_targets[:500]):
    pred = model(torch.FloatTensor(context).unsqueeze(0)).numpy()[0]
    predictions.append(pred)
    targets_list.append(target)

predictions = np.array(predictions)
targets_list = np.array(targets_list)

# Combine and reduce dimensions
combined = np.vstack([predictions, targets_list])
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(combined)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(reduced[:500, 0], reduced[:500, 1], c='blue', alpha=0.5, label='Predictions')
plt.scatter(reduced[500:, 0], reduced[500:, 1], c='red', alpha=0.5, label='Targets')
plt.legend()
plt.title('Predictions vs Targets (t-SNE)')
plt.savefig('artifacts/lvm/evaluation/tsne_pred_vs_target.png')
```

**Expected Results**:
- Predictions and targets should overlap significantly
- Clusters should align (not offset)
- Few outlier predictions

---

### Test 2.3: Error Pattern Analysis
**Status**: PENDING

**Purpose**: Understand when and why predictions fail

**Test Script**:
```python
import pandas as pd

errors = []

for i, (context, target) in enumerate(zip(val_contexts, val_targets)):
    pred = model(torch.FloatTensor(context).unsqueeze(0)).numpy()[0]

    # Compute error
    mse = ((pred - target) ** 2).mean()
    cosine = (pred @ target) / (np.linalg.norm(pred) * np.linalg.norm(target))

    errors.append({
        'sample_id': i,
        'mse': mse,
        'cosine': cosine,
        'error': 1 - cosine,
        'context_similarity': np.mean([
            (context[j] @ context[j+1]) /
            (np.linalg.norm(context[j]) * np.linalg.norm(context[j+1]))
            for j in range(len(context)-1)
        ])
    })

df = pd.DataFrame(errors)

# Analyze
print("Error Distribution:")
print(df['error'].describe())

print("\nWorst 10 predictions:")
print(df.nlargest(10, 'error')[['sample_id', 'error', 'context_similarity']])

# Hypothesis: Does context similarity predict error?
import scipy.stats
corr, p_value = scipy.stats.pearsonr(df['context_similarity'], df['error'])
print(f"\nContext similarity vs Error correlation: {corr:.3f} (p={p_value:.4f})")
```

**Expected Insights**:
- High error when context vectors are dissimilar (topic shifts)
- Low error when context is coherent (smooth narrative)
- Errors correlate with rare/technical concepts

---

## 📋 Phase 3: Integration Testing

### Test 3.1: Vec2Text Pipeline
**Status**: PENDING

**Purpose**: Full vector → prediction → text generation

**Test Script**:
```python
from app.vect_text_vect.vec_text_vect_isolated import vec2text_decode

# Pick a test example
context = val_contexts[0]  # 5 vectors of 768D
target_text = concept_texts[some_index]  # Ground truth

# Predict next vector
pred_vector = model(torch.FloatTensor(context).unsqueeze(0)).numpy()[0]

# Decode to text (JXE and IELab)
jxe_text = vec2text_decode(pred_vector, backend='jxe', steps=1)
ielab_text = vec2text_decode(pred_vector, backend='ielab', steps=1)

print(f"Target: {target_text}")
print(f"JXE:    {jxe_text}")
print(f"IELab:  {ielab_text}")

# Compute BLEU/ROUGE if applicable
```

**Expected Results**:
- Vec2text should produce sensible text
- JXE typically faster, IELab more accurate
- Some semantic overlap with target (not exact match expected)

---

### Test 3.2: Autoregressive Generation
**Status**: PENDING

**Purpose**: Generate multi-step sequences (not just one vector)

**Test Script**:
```python
def generate_sequence(model, seed_context, num_steps=10):
    """Generate num_steps vectors autoregressively"""
    context = seed_context.copy()  # [5, 768]
    generated = []

    for step in range(num_steps):
        # Predict next
        pred = model(torch.FloatTensor(context).unsqueeze(0)).numpy()[0]
        generated.append(pred)

        # Update context (slide window)
        context = np.vstack([context[1:], pred])

    return np.array(generated)

# Test
seed = val_contexts[0]
sequence = generate_sequence(model, seed, num_steps=20)

# Check for degeneration (all vectors become similar)
similarities = []
for i in range(len(sequence)-1):
    sim = (sequence[i] @ sequence[i+1]) / (
        np.linalg.norm(sequence[i]) * np.linalg.norm(sequence[i+1])
    )
    similarities.append(sim)

print("Autoregressive generation quality:")
print(f"  Mean cosine (consecutive): {np.mean(similarities):.3f}")
print(f"  Std dev: {np.std(similarities):.3f}")
print(f"  Degeneration check: {'PASS' if np.mean(similarities) < 0.95 else 'FAIL'}")
```

**Expected Results**:
- Mean similarity 0.7-0.9 (coherent but diverse)
- No degeneration (all vectors converging to same value)
- Generated sequences should follow semantic flow

---

### Test 3.3: FastAPI Integration
**Status**: PENDING

**Purpose**: Deploy model behind HTTP endpoint

**Test Script**:
```python
# In app/api/lvm_inference.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI()

# Load model at startup
model = ...  # Load best model

class PredictionRequest(BaseModel):
    context_vectors: list[list[float]]  # [5, 768]

class PredictionResponse(BaseModel):
    predicted_vector: list[float]  # [768]
    cosine_confidence: float

@app.post("/predict")
def predict(request: PredictionRequest):
    context = np.array(request.context_vectors)
    pred = model(torch.FloatTensor(context).unsqueeze(0)).numpy()[0]

    # Compute confidence (how similar to training data)
    # ... (simplified)

    return PredictionResponse(
        predicted_vector=pred.tolist(),
        cosine_confidence=0.85
    )

# Test with curl
# curl -X POST http://localhost:8080/predict \
#   -H "Content-Type: application/json" \
#   -d '{"context_vectors": [[...], ..., [...]]}'
```

**Expected Results**:
- API responds in <100ms
- Predictions match offline evaluation
- Can handle concurrent requests

---

## 📋 Phase 4: Production Optimization

### Test 4.1: Model Quantization
**Status**: PENDING

**Purpose**: Reduce model size and increase speed

**Test Script**:
```python
import torch.quantization

# Dynamic quantization (easy, good for LSTMs)
model_fp32 = LSTMVectorPredictor()
model_fp32.load_state_dict(...)
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# Test accuracy loss
# ... (run validation set through both models)

# Measure size reduction
torch.save(model_fp32.state_dict(), 'model_fp32.pt')
torch.save(model_int8.state_dict(), 'model_int8.pt')

import os
size_fp32 = os.path.getsize('model_fp32.pt') / 1024**2  # MB
size_int8 = os.path.getsize('model_int8.pt') / 1024**2  # MB

print(f"FP32: {size_fp32:.2f} MB")
print(f"INT8: {size_int8:.2f} MB ({100*(1-size_int8/size_fp32):.1f}% reduction)")
```

**Expected Results**:
- 60-75% size reduction
- <1% accuracy loss
- 2-3x inference speedup (on CPU)

---

### Test 4.2: ONNX Export
**Status**: PENDING

**Purpose**: Deploy to production inference engines (TensorRT, etc.)

**Test Script**:
```python
import torch.onnx

dummy_input = torch.randn(1, 5, 768)
torch.onnx.export(
    model,
    dummy_input,
    "artifacts/lvm/models/lstm_baseline/model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['context'],
    output_names=['prediction'],
    dynamic_axes={'context': {0: 'batch_size'}}
)

# Validate ONNX
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX model is valid")

# Test with ONNX Runtime
import onnxruntime as ort
ort_session = ort.InferenceSession("model.onnx")
outputs = ort_session.run(None, {"context": dummy_input.numpy()})
print(f"ONNX output shape: {outputs[0].shape}")
```

**Expected Results**:
- ONNX export succeeds
- Inference matches PyTorch (< 0.1% difference)
- 1.5-2x speedup with ONNX Runtime

---

## 📊 Test Execution Schedule

### Week 1 (Today - Oct 12)
- [x] Phase 1.1: Model Loading
- [x] Phase 1.2: Validation Set Inference
- [ ] Phase 1.3: Inference Speed Benchmark
- [ ] Phase 1.4: Memory Profiling

### Week 1 (Oct 13-14)
- [ ] Phase 2.1: Top-K Retrieval Accuracy ⭐ **HIGHEST PRIORITY**
- [ ] Phase 2.2: Semantic Coherence Visualization
- [ ] Phase 2.3: Error Pattern Analysis

### Week 2 (Oct 15-18)
- [ ] Phase 3.1: Vec2Text Integration
- [ ] Phase 3.2: Autoregressive Generation
- [ ] Phase 3.3: FastAPI Deployment

### Week 3 (Oct 19+)
- [ ] Phase 4.1: Quantization Optimization
- [ ] Phase 4.2: ONNX Export
- [ ] Final model selection for production

---

## 🎯 Success Criteria Summary

### Must-Have (Required)
- ✅ Val cosine similarity > 75%
- ✅ All models load and inference correctly
- ⏳ Top-20 retrieval accuracy > 70%
- ⏳ Inference < 100ms per sample (batch=32)
- ⏳ No degeneration in autoregressive mode

### Should-Have (Target)
- ⏳ Top-10 retrieval accuracy > 60%
- ⏳ Vec2text integration produces coherent text
- ⏳ FastAPI endpoint deployed and tested
- ⏳ Quantized model with <1% accuracy loss

### Nice-to-Have (Stretch)
- ⏳ Top-5 retrieval accuracy > 50%
- ⏳ ONNX export for production deployment
- ⏳ Comprehensive error analysis with insights
- ⏳ Multi-step generation demo (10+ steps)

---

## 📝 Test Report Template

For each test phase, document:

```markdown
### Test [Phase.Number]: [Name]
**Date**: YYYY-MM-DD
**Tester**: [Name]
**Models Tested**: LSTM, GRU, Transformer

**Setup**:
- Data: [dataset used]
- Environment: [device, Python version, libraries]
- Parameters: [any test-specific settings]

**Results**:
| Model | Metric 1 | Metric 2 | ... | Pass/Fail |
|-------|----------|----------|-----|-----------|
| LSTM  | 78.30%   | ...      | ... | ✅         |
| GRU   | 78.33%   | ...      | ... | ✅         |
| Trans | 78.60%   | ...      | ... | ✅         |

**Analysis**:
- Observation 1
- Observation 2
- Recommendation

**Next Steps**:
- Action item 1
- Action item 2
```

---

**Last Updated**: October 12, 2025
**Next Review**: After Phase 2.1 completion
