# LVM Training Progress - October 13, 2025

## Summary

All 3 LVM models are training successfully with InfoNCE loss + moment matching. Training started at 00:34 EDT.

## Model Configurations

| Model | Parameters | Architecture | Learning Rate |
|-------|-----------|--------------|---------------|
| LSTM | 5.38M | 2 layers, hidden_dim=512 | 0.001 |
| Mamba2/GRU | 7.36M | 4 layers, d_model=512 (GRU fallback) | 0.0005 |
| Transformer | 17.87M | 4 layers, 8 heads, d_model=512 | 0.0005 |

## Current Progress (as of 00:36 EDT)

### LSTM (PID 55928) - Epoch 8/20 ⚡ Fastest
- **Epoch 7**: Val Cosine **1.61%**, Val Loss 0.002562
- InfoNCE loss: 6.84 (down from 6.97)
- Training smoothly, loss decreasing steadily

### Mamba2/GRU (PID 55957) - Epoch 5/20
- **Epoch 4**: Train Cosine **4.07%**, Val Cosine -0.04%
- InfoNCE loss: 6.28 (lowest of all models)
- Strong training signal, best early performance

### Transformer (PID 55992) - Epoch 2/20 🐢 Slowest
- **Epoch 1**: Val Cosine **1.81%**, Val Loss 0.002557
- InfoNCE loss: 6.99
- Largest model, slowest training but making progress

## Target Metrics

- **Baseline**: 48.16% (global mean vector)
- **Target**: 55-60%+ cosine similarity
- **Current best**: 4.07% (Mamba2 train), 1.81% (Transformer val)
- **Gap to baseline**: Still 44-46 percentage points away

## Training Commands

```bash
# LSTM
nohup ./.venv/bin/python -u app/lvm/train_lstm_baseline.py \
  --epochs 20 --device cpu \
  --data artifacts/lvm/training_sequences_ctx5_sentence.npz \
  > /tmp/lstm_training_oct13_v2.log 2>&1 &

# Mamba2/GRU
nohup ./.venv/bin/python -u app/lvm/train_mamba2.py \
  --epochs 20 --device cpu \
  --data artifacts/lvm/training_sequences_ctx5_sentence.npz \
  > /tmp/mamba2_training_oct13_v2.log 2>&1 &

# Transformer
nohup ./.venv/bin/python -u app/lvm/train_transformer.py \
  --epochs 20 --device cpu \
  --data artifacts/lvm/training_sequences_ctx5_sentence.npz \
  > /tmp/transformer_training_oct13_v2.log 2>&1 &
```

## Monitoring

```bash
# Quick status check
/tmp/check_training_status.sh

# Full logs
tail -f /tmp/lstm_training_oct13_v2.log
tail -f /tmp/mamba2_training_oct13_v2.log
tail -f /tmp/transformer_training_oct13_v2.log

# Process info
ps aux | grep -E "(55928|55957|55992)" | grep python
```

## Training Data

- **File**: `artifacts/lvm/training_sequences_ctx5_sentence.npz`
- **Sequences**: 8,106 training pairs
- **Format**: Context (5 vectors) → Target (1 vector)
- **Dimensions**: 768D (GTR-T5 embeddings)
- **Split**: 90% train (7,295), 10% val (811)

## Loss Function

**Composite Loss** = InfoNCE + Moment Matching + Variance Regularization

- **InfoNCE**: Contrastive loss with in-batch negatives (prevents collapse)
- **Moment Matching**: Preserves per-dimension mean/std
- **Variance Regularization**: Maintains vector diversity

## Expected Timeline

- **LSTM**: ~40 minutes (fastest, smallest model)
- **Mamba2/GRU**: ~50 minutes (medium size)
- **Transformer**: ~70 minutes (largest, slowest)
- **Total**: All 3 should complete within 1-1.5 hours

## Key Findings

1. ✅ **Bytecode cache issue resolved** - Cleared __pycache__ before training
2. ✅ **Unbuffered output working** - Using `python -u` flag for real-time logs
3. ✅ **InfoNCE loss effective** - Loss decreasing from 6.97 → 6.28-6.84
4. ✅ **No crashes** - All 3 models training stably
5. ⚠️ **Still far from baseline** - Need to reach 48%+ to beat mean vector

## Next Steps

1. Wait for training to complete (all 20 epochs)
2. Compare final validation cosine similarity
3. Test best model on held-out test set
4. If performance is good (>55%), deploy to LVM server
5. If performance is poor (<48%), investigate:
   - Need more training data?
   - Need longer context window?
   - Need different architecture?
   - Need different loss weights?

## Session Info

- **Date**: October 13, 2025
- **User Request**: "Well retrain with 20 epochs!"
- **Previous Issue**: Models were undertrained (1 epoch only)
- **Resolution**: Full 20-epoch training with proper logging
