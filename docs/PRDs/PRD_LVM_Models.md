# PRD: LVM Model Training & Evaluation Matrix
**Created**: October 12, 2025
**Status**: In Progress
**Goal**: Train and evaluate 12+ LVM architectures for vector-native next-vector prediction

---

## 📊 Model Training & Evaluation Matrix

| # | Architecture | Target Params | Status | Val Loss | Val Cosine | Train Time | Model File | Success Score | Notes |
|---|--------------|---------------|--------|----------|------------|------------|------------|---------------|-------|
| 1 | **LSTM Baseline** | 5.1M | ✅ **COMPLETE** | 0.000504 | 78.30% | 10 min | `artifacts/lvm/models/lstm_baseline/best_model.pt` | 70% | Simple recurrent, 2 layers, 512 hidden |
| 2 | **GRU Stacked** | 7.1M | ✅ **COMPLETE** | 0.000503 | 78.33% | 10 min | `artifacts/lvm/models/mamba2/best_model.pt` | 72% | Gated recurrent, 4 layers, 512 d_model |
| 3 | **Transformer** | 17.6M | ✅ **COMPLETE** | **0.000498** | **78.60%** | 12 min | `artifacts/lvm/models/transformer/best_model.pt` | 85% | 4 layers, 8 heads, 512 d_model - **CURRENT BEST** |
| 4 | **DistilGPT-2** | 82M | 🔄 QUEUED | - | - | - | - | 82% | Lightweight GPT-2 variant |
| 5 | **Performer** | 100M | 🔄 QUEUED | - | - | - | - | 78% | Linear attention (FAVOR+) |
| 6 | **Linformer** | 100M | 🔄 QUEUED | - | - | - | - | 77% | Linear complexity attention |
| 7 | **S4** | 100M | 🔄 QUEUED | - | - | - | - | 75% | Structured state space |
| 8 | **Hyena** | 125M | 🔄 QUEUED | - | - | - | - | 80% | Long convolution operator |
| 9 | **RetNet** | 125M | 🔄 QUEUED | - | - | - | - | 88% | Retention mechanism, O(1) inference |
| 10 | **Mamba-2** | 130M | ❌ BLOCKED | - | - | - | - | 95% | Requires Python <3.13 (current: 3.13) |
| 11 | **Hybrid Mamba-Attn** | 150M | 🔄 QUEUED | - | - | - | - | 92% | Interleaved SSM + Attention |
| 12 | **RWKV** | 169M | 🔄 QUEUED | - | - | - | - | 85% | Receptance Weighted Key Value |
| 13 | **Meta LCM** | 1.6B | 🔄 QUEUED | - | - | - | - | 90% | Large Concept Model (sentence embeddings) |

---

## 🎯 Success Criteria

### Minimum Viable Performance
- **Val Loss**: < 0.001000 (MSE)
- **Cosine Similarity**: > 70%
- **Training Time**: < 4 hours per model
- **Model Size**: < 2GB on disk

### Target Performance
- **Val Loss**: < 0.000600 (beat LSTM baseline)
- **Cosine Similarity**: > 75%
- **Training Time**: < 2 hours per model

### Stretch Goals
- **Val Loss**: < 0.000500 (beat Transformer)
- **Cosine Similarity**: > 78.6% (beat Transformer)

---

## 📈 Training Configuration

### Dataset
- **Source**: Wikipedia articles (42,113 concepts)
- **Training pairs**: 42,108 sequences
- **Train/Val split**: 90/10 (37,897 / 4,211)
- **Context length**: 5 vectors
- **Vector dimension**: 768D (GTR-T5-base)

### Hyperparameters (Default)
- **Batch size**: 32
- **Epochs**: 20
- **Learning rate**: 0.0005 (AdamW)
- **Weight decay**: 0.01
- **LR schedule**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Gradient clipping**: 1.0
- **Loss**: MSE
- **Device**: MPS (Apple Silicon)

---

## 🏗️ Architecture Details

### 1. LSTM Baseline ✅
```python
LSTMVectorPredictor(
    input_dim=768,
    hidden_dim=512,
    num_layers=2,
    dropout=0.2
)
```
**Complexity**: O(n)
**Speed**: ⚡⚡⚡ Fast
**Best for**: Resource-constrained deployment

### 2. GRU Stacked ✅
```python
Mamba2VectorPredictor(  # Using GRU fallback
    input_dim=768,
    d_model=512,
    num_layers=4
)
```
**Complexity**: O(n)
**Speed**: ⚡⚡⚡ Fast
**Best for**: Balanced performance/efficiency

### 3. Transformer ✅
```python
TransformerVectorPredictor(
    input_dim=768,
    d_model=512,
    nhead=8,
    num_layers=4,
    dim_feedforward=2048,
    dropout=0.1
)
```
**Complexity**: O(n²)
**Speed**: ⚡⚡ Moderate
**Best for**: Maximum accuracy

### 4. DistilGPT-2
```python
# Using Hugging Face transformers
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config(
    vocab_size=1,  # Not using tokens
    n_embd=768,
    n_layer=6,
    n_head=12
)
```
**Complexity**: O(n²)
**Speed**: ⚡ Slow
**Best for**: Transfer learning from pretrained

### 5. Performer
```python
# Linear attention using FAVOR+
from performer_pytorch import PerformerLM

model = PerformerLM(
    dim=768,
    depth=6,
    heads=8,
    dim_head=64
)
```
**Complexity**: O(n)
**Speed**: ⚡⚡ Moderate
**Best for**: Long sequences with linear complexity

### 6-13. (To be detailed as implemented)

---

## 🧪 Evaluation Metrics

### Primary Metrics
1. **MSE Loss** - Mean squared error between predicted and target vectors
2. **Cosine Similarity** - Semantic alignment (0-100%)

### Secondary Metrics
3. **Top-K Retrieval Accuracy** - How often target is in top K neighbors
   - Top-1: Exact match
   - Top-5: Target in top 5
   - Top-10: Target in top 10
   - Top-20: Target in top 20
4. **Inference Speed**
   - Latency (ms/sample)
   - Throughput (samples/sec)
5. **Memory Usage**
   - Peak GPU RAM (MB)
   - Model size on disk (MB)

### Tertiary Metrics
6. **Training Efficiency**
   - Epochs to convergence
   - Total training time
   - GPU-hours consumed
7. **Generalization**
   - Train/Val gap
   - Performance on held-out test set

---

## 📋 Test Plan

### Phase 1: Basic Validation (All Models)
- [ ] Load trained model checkpoint
- [ ] Run inference on validation set (4,211 samples)
- [ ] Compute MSE loss and cosine similarity
- [ ] Measure inference latency
- [ ] Record model size

### Phase 2: Retrieval Evaluation (Top Models)
- [ ] Load full vector database (42,113 concepts)
- [ ] For each prediction, find K nearest neighbors
- [ ] Compute Top-1, Top-5, Top-10, Top-20 accuracy
- [ ] Analyze failure cases (where target not in Top-20)

### Phase 3: Qualitative Analysis (Best 3 Models)
- [ ] Visualize predictions vs targets (t-SNE/UMAP)
- [ ] Analyze error patterns by article type
- [ ] Test on out-of-domain data (if available)
- [ ] Generate text via vec2text decoding

### Phase 4: Production Readiness (Winner)
- [ ] Optimize for inference (quantization, pruning)
- [ ] Benchmark on target hardware (CPU, MPS, CUDA)
- [ ] Implement FastAPI endpoint
- [ ] Load testing (concurrent requests)
- [ ] Deploy to staging environment

---

## 🎨 Visualization Plan

### Training Curves
- Loss curves (train vs val) for all models
- Cosine similarity over epochs
- Learning rate schedule
- Gradient norms

### Model Comparison
- Bar chart: Val loss by model
- Bar chart: Cosine similarity by model
- Scatter: Params vs Performance
- Scatter: Training time vs Performance

### Retrieval Analysis
- Heatmap: Top-K accuracy by model
- Confusion matrix: Predicted vs Target categories
- t-SNE plot: Predictions vs Targets (colored by error)

### Error Analysis
- Histogram: Error distribution (MSE per sample)
- Box plot: Error by article length
- Scatter: Error vs similarity to training data

---

## 📁 File Organization

```
artifacts/lvm/models/
├── lstm_baseline/
│   ├── best_model.pt           # Best validation checkpoint
│   ├── final_model.pt          # Final epoch checkpoint
│   ├── training_history.json   # Loss/cosine per epoch
│   └── config.json             # Hyperparameters
├── gru_stacked/
│   └── ...
├── transformer/
│   └── ...
├── distilgpt2/
│   └── ...
├── performer/
│   └── ...
└── [model_name]/
    └── ...

artifacts/lvm/evaluation/
├── validation_results.json     # All models on validation set
├── retrieval_results.json      # Top-K accuracy results
├── inference_benchmarks.json   # Speed/memory metrics
└── comparison_plots/
    ├── loss_comparison.png
    ├── cosine_comparison.png
    ├── topk_accuracy.png
    └── params_vs_performance.png
```

---

## 🚀 Next Steps

### Immediate (Today)
1. ✅ Complete LSTM, GRU, Transformer baselines
2. ⏳ Start DistilGPT-2 training
3. ⏳ Start Performer training
4. ⏳ Start RWKV training (if available)

### Short-term (This Week)
5. Complete all queued models
6. Run full evaluation pipeline
7. Generate comparison visualizations
8. Select top 3 models for phase 3 testing

### Medium-term (Next Week)
9. Implement retrieval evaluation (Top-K)
10. Qualitative analysis on best models
11. Begin production optimization
12. Start integration with vec2text

### Long-term (Month 1)
13. Deploy best model to FastAPI
14. Production benchmarking
15. A/B testing with baseline retrieval
16. Begin 100-iteration experiment plan

---

## 📊 Success Metrics Dashboard

### Current Best: Transformer
- **Val Loss**: 0.000498
- **Cosine**: 78.60%
- **Params**: 17.6M
- **Speed**: 12 min training

### Models to Beat Baseline (Val Loss < 0.000504)
- [ ] DistilGPT-2
- [ ] Performer
- [ ] Linformer
- [ ] S4
- [ ] Hyena
- [ ] RetNet
- [ ] Hybrid Mamba-Attn
- [ ] RWKV
- [ ] Meta LCM

### Stretch Goal (Val Loss < 0.000498)
**Beat Transformer!** 🏆

---

## 🔗 Related Documents
- [LVM Training Results - Oct 12](../../LVM_TRAINING_RESULTS_OCT12.md)
- [Training Data Handoff](../../SESSION_HANDOFF_OCT12_READY_FOR_LVM_TRAINING.md)
- [LVM Architecture Options](../LVM_ARCHITECTURE_OPTIONS.md)
- [100-Iteration Training Plan](../../SESSION_HANDOFF_OCT12_READY_FOR_LVM_TRAINING.md#100-iteration-training-plan)

---

**Last Updated**: October 12, 2025
**Status**: 3/13 models trained, 10 queued
