# GraphMERT-LVM 10k Benchmark Guide

**Created:** 2025-10-16
**Status:** Ready to run

## Overview

This benchmark tests the **768-d native GraphMERT-LVM architecture** on 10k Wikipedia sequences to measure training time before scaling to the full 80k dataset.

## Key Design Decisions

### 1. **768-d Native Architecture** (No Vec2Text During Training!)
- ❌ **Original plan**: Decode vectors to text → entity linking → train
- ✅ **Simplified approach**: Train directly on 768-d vectors (GTR-T5 space)
- **Why**: Vec2Text is slow (~10s/sequence) and not needed for training
- **When to use Vec2Text**: Only for inference/evaluation (debugging, FActScore*)

### 2. **No Projection Layer**
- Standard GraphMERT: 768-d word embeddings → project to 512-d
- **Our approach**: Stay in 768-d GTR-T5 space (truly vector-native!)
- **Benefits**:
  - Simpler architecture
  - Directly compatible with Vec2Text for inference
  - Fewer parameters to train

### 3. **Simplified Version (No KG Yet)**
- This benchmark: Pure autoregressive vector prediction (5 context → 1 target)
- Full version (later): Add KG leaves for neurosymbolic reasoning
- **Rationale**: Get training time baseline first, then add complexity

## Architecture Specs

```
Model: GraphMERT-LVM-768D
Parameters: ~67M

Input: 5×768-d context vectors
  ↓
Position embeddings (learned)
  ↓
12 Transformer layers (RoBERTa-style)
  - 8 attention heads
  - 2048 feed-forward hidden
  - Attention decay mask (λ=0.6)
  ↓
Mean pooling
  ↓
Output head
  ↓
Output: 768-d prediction (normalized)

Loss: MSE (mean squared error)
Metric: Cosine similarity
```

## Files Created

### 1. **Data**
- `tools/create_10k_training_subset.py` - Extract 10k from 80k
- `artifacts/lvm/training_sequences_ctx5_10k.npz` - 10k subset (175.8 MB)

### 2. **Model**
- `app/lvm/graphmert_lvm_768d.py` - 768-d native encoder
  - `GraphMERTLVM768D` - Main model class (~67M params)
  - `AttentionDecayMask` - Exponential decay (λ=0.6, learnable threshold)
  - `GraphMERTTransformerLayer` - RoBERTa-style layer

### 3. **Training**
- `app/lvm/train_graphmert_lvm_benchmark.py` - Training script
  - Supports single GPU or multi-GPU (DDP)
  - MSE loss + cosine similarity metric
  - 90/10 train/val split

### 4. **Launcher**
- `tools/run_graphmert_benchmark.sh` - Easy launcher script

## How to Run

### Option 1: Single GPU (Quick Test)
```bash
./tools/run_graphmert_benchmark.sh 1
```

### Option 2: 8 GPUs (Recommended for Benchmark)
```bash
./tools/run_graphmert_benchmark.sh 8
```

### Option 3: 40 GPUs (Maximum Parallelization)
```bash
./tools/run_graphmert_benchmark.sh 40
```

### Manual Run (Custom Parameters)
```bash
# Single GPU
python app/lvm/train_graphmert_lvm_benchmark.py \
    --data artifacts/lvm/training_sequences_ctx5_10k.npz \
    --epochs 3 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda:0

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 \
    app/lvm/train_graphmert_lvm_benchmark.py \
    --epochs 3 \
    --batch-size 32 \
    --lr 1e-4
```

## Expected Results

### Training Time Estimates

**10k dataset, 3 epochs:**
- **1 GPU**: ~10-15 min (rough estimate)
- **8 GPUs**: ~2-3 min (8x speedup)
- **40 GPUs**: <1 min? (depends on communication overhead)

**Scaling to 80k dataset, 25 epochs:**
- Based on 10k results × 8 (data) × 8.33 (epochs)
- **1 GPU**: ~10-15 hours
- **8 GPUs**: ~1-2 hours ✅
- **40 GPUs**: <30 min? (if communication overhead is low)

### Validation Metrics
- **Target cosine similarity**: ≥0.55 (LSTM baseline: 0.5758)
- **MSE loss**: Should decrease over epochs

## Output Files

After running, check:
```
artifacts/lvm/models/graphmert_lvm_benchmark/
├── benchmark_model.pt           # Trained model checkpoint
└── benchmark_results.json       # Training metrics + timing
```

**benchmark_results.json** contains:
```json
{
  "model_type": "GraphMERT-LVM-768D",
  "dataset_size": 10000,
  "train_size": 9000,
  "val_size": 1000,
  "epochs": 3,
  "batch_size": 32,
  "world_size": 8,  // Number of GPUs used
  "total_training_time": 180.5,  // seconds
  "avg_epoch_time": 60.2,
  "parameters": 67352833,
  "history": [
    {"epoch": 1, "train_loss": 0.12, "train_cosine": 0.45, ...},
    {"epoch": 2, "train_loss": 0.08, "train_cosine": 0.52, ...},
    {"epoch": 3, "train_loss": 0.06, "train_cosine": 0.56, ...}
  ]
}
```

## Next Steps After Benchmark

### 1. Analyze Timing
```bash
# Extract timing from results
cat artifacts/lvm/models/graphmert_lvm_benchmark/benchmark_results.json | jq '.total_training_time'
```

### 2. Scale to Full 80k Dataset
```bash
# If timing looks good, run full training
python app/lvm/train_graphmert_lvm_benchmark.py \
    --data artifacts/lvm/training_sequences_ctx5.npz \  # Full 80k
    --epochs 25 \
    --batch-size 32 \
    --output-dir artifacts/lvm/models/graphmert_lvm_full
```

### 3. Add KG Leaves (Phase 2)
- Vector-based entity linking (no Vec2Text!)
- Build leafy chain graphs
- Modify encoder to handle KG structure
- Add MNM loss (masked node modeling)

## Comparison: Original Plan vs Simplified Approach

| Aspect | Original Plan (PRD) | Simplified Approach (This Benchmark) |
|--------|---------------------|--------------------------------------|
| **Vec2Text decoding** | 80k sequences (3-7 days) | ❌ Skip for training |
| **Entity linking** | Text-based NER | Vector-based cosine similarity (Phase 2) |
| **Seed KG** | Wikidata (1M entities) | Not needed for benchmark |
| **Leafy chain graphs** | Full structure | Simplified (5 vectors only) |
| **Training data prep** | Weeks | Hours |
| **Architecture** | 512-d (with projection) | 768-d native (no projection) |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python app/lvm/train_graphmert_lvm_benchmark.py --batch-size 16
```

### DDP Errors
```bash
# Check GPU availability
nvidia-smi

# Use fewer GPUs
./tools/run_graphmert_benchmark.sh 4
```

### Model Not Loading
```python
# Load checkpoint
import torch
checkpoint = torch.load('artifacts/lvm/models/graphmert_lvm_benchmark/benchmark_model.pt')
print(checkpoint.keys())
```

## Resources

- **PRD**: `docs/PRDs/PRD_GraphMERT_LVM_Integration.md`
- **GraphMERT Paper**: `docs/papers/GraphMERT_Princeton_University.pdf`
- **Existing LVM training**: `app/lvm/train_unified.py` (reference)

---

**Ready to run?**
```bash
./tools/run_graphmert_benchmark.sh 8  # Start with 8 GPUs
```
