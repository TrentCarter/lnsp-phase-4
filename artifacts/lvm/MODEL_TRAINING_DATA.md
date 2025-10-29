# LVM Model Training Data Documentation

**Always list what models were trained on!** This file documents the training data sources for all LVM models to prevent regression and ensure reproducibility.

---

## ✅ Production Models (Port Assignments)

### Port 9001: AMN (Attention Mixture Network)
**Model File**: `artifacts/lvm/models/amn_v0.pt` → `amn_584k_pure_mse_20251029_055838/best_model.pt`

**Training Data**:
- **Source**: Wikipedia 500k articles (first 339,615 concepts)
- **Dataset**: `artifacts/lvm/training_sequences_ctx5_584k_fresh.npz`
- **Sequences**: 543,773 training pairs (5-vector context → next vector)
- **Format**: Sequential chunks from Wikipedia articles (narrative flow)
- **Data Type**: Sequential document data (NOT ontological/taxonomic)

**Training Configuration**:
- **Loss**: Pure MSE (λ=1.0), InfoNCE disabled (λ=0.0)
- **Optimizer**: Adam, LR=0.0005, cosine schedule
- **Epochs**: 20
- **Batch Size**: 32
- **Device**: MPS (Apple Silicon GPU)
- **Val Cosine**: 0.5605 (target: ≥0.56)
- **Parameters**: 1,510,912

**Training Date**: October 29, 2025
**Critical Fix**: InfoNCE completely disabled to prevent gradient dominance

---

### Port 9002: Transformer (Baseline)
**Model File**: `artifacts/lvm/models/transformer_v0.pt`

**Training Data**: Same as AMN (Wikipedia 584k sequential chunks)
**Val Cosine**: ~0.577
**Loss**: MSE primary (baseline config)

---

### Port 9003: GRU
**Model File**: `artifacts/lvm/models/gru_v0.pt`

**Training Data**: Same as AMN (Wikipedia 584k sequential chunks)
**Val Cosine**: ~0.573
**Loss**: MSE primary

---

### Port 9004: LSTM ⭐ (Best Balance)
**Model File**: `artifacts/lvm/models/lstm_v0.pt`

**Training Data**: Same as AMN (Wikipedia 584k sequential chunks)
**Val Cosine**: ~0.576
**Loss**: MSE primary
**Note**: Best balance of accuracy and speed (0.56ms/query)

---

### Port 9005: Vec2Text Direct (Passthrough)
**Model File**: N/A (no LVM - direct encoder→decoder)

**No Training**: Passthrough mode skips LVM entirely
**Purpose**: Baseline comparison for vec2text quality

---

### Port 9006: Transformer (Optimized)
**Model File**: `artifacts/lvm/models/transformer_optimized_v0.pt` → `transformer_optimized_20251024_072726/best_model.pt`

**Training Data**: Same as AMN (Wikipedia 584k sequential chunks)
**Val Cosine**: ~0.586 (highest accuracy)
**Loss**: MSE primary with optimizations

---

## 🚨 Critical Rules for Training Data

### ✅ MUST USE (Sequential Data)
- **Wikipedia articles** - Narrative progression ✅
- **Textbooks** - Sequential instruction ("First... → Next... → Finally...") ✅
- **Scientific papers** - Temporal flow ("Methods → Results → Conclusions") ✅
- **Programming tutorials** - Step-by-step procedures ✅
- **Stories/narratives** - Causal/temporal relationships ✅

### ❌ NEVER USE (Taxonomic/Ontological Data)
- **WordNet** - Taxonomic hierarchies ("dog → mammal → animal") ❌
- **SWO/GO** - Ontological categories ❌
- **DBpedia ontology chains** - Classification structures ❌
- **FactoidWiki** - Short isolated facts ❌

**Why**: Autoregressive LVMs predict next vector from context. They need temporal/causal relationships (sequential flow), NOT IS-A hierarchies (classification trees).

**Validation**: Use `tools/test_sequential_coherence.py` before training on new datasets

---

## 📊 Training Dataset Validation

### Wikipedia 584k Dataset
**Location**: `artifacts/lvm/training_sequences_ctx5_584k_fresh.npz`

**Validation Checks** (Passed ✅):
- ✅ Sequential chunks from same document (preserves narrative flow)
- ✅ 5-vector context windows (matches inference distribution)
- ✅ No CPESH metadata required (pure vector prediction)
- ✅ Vectors: 768D GTR-T5 embeddings (vec2text-compatible)
- ✅ No ontology contamination (dataset_source = "wikipedia")

**NPZ Structure**:
```python
{
    'context_vectors': np.ndarray([N, 5, 768]),  # Context windows
    'target_vectors': np.ndarray([N, 768]),      # Next vector to predict
    'doc_ids': np.ndarray([N], dtype=int32),     # Document IDs (for filtering)
    'positions': np.ndarray([N], dtype=int32),   # Position in document
}
```

---

## 🔴 Known Training Failures (Documented for Prevention)

### ❌ AMN Attempt 1: InfoNCE Dominant (FAILED)
- **Date**: October 28, 2025
- **Config**: MSE λ=0.05, InfoNCE λ=1.0
- **Val Cosine**: 0.405 (poor)
- **Issue**: InfoNCE loss magnitude (0.92) >> MSE magnitude (0.0014)
- **Result**: InfoNCE gradient dominated → poor convergence
- **Output Quality**: Gibberish

### ❌ AMN Attempt 2: InfoNCE Still Dominant (FAILED)
- **Date**: October 29, 2025
- **Config**: MSE λ=1.0, InfoNCE λ=0.05
- **Val Cosine**: 0.405 (no improvement)
- **Issue**: Even at λ=0.05, InfoNCE contribution (0.046) >> MSE (0.001427) - 32x larger!
- **Result**: InfoNCE still dominated gradients
- **Output Quality**: Gibberish

### ✅ AMN Attempt 3: Pure MSE (SUCCESS)
- **Date**: October 29, 2025
- **Config**: MSE λ=1.0, InfoNCE λ=0.0 (completely disabled)
- **Val Cosine**: 0.5605 (success!)
- **Result**: Pure regression training worked
- **Output Quality**: Coherent text with manifold snap

**Lesson**: For autoregressive LVM training, **disable InfoNCE** unless you implement gradient-norm balancing to prevent dominance.

---

## 🎯 Production Pipeline Enhancements

### Manifold Snap + Topic Anchor (All Ports)
**Purpose**: Pull LVM predictions onto valid vec2text decoding manifold
**Method**: kNN barycentric projection (K=16) + weighted blend
**Weights**:
- Standard: 75% snap + 15% query + 10% context
- Drift clamp (cos<0.20 or cos<0.30): 60% snap + 25% query + 15% context

**Impact**: Transformed AMN gibberish → coherent output (no retraining required!)

### Adaptive Decode with Quality Checks (All Ports)
**Escalation Ladder**: steps=1 → steps=3 → steps=5 → extractive fallback
**Quality Checks**:
- Bigram repetition ≤25%
- Entropy ≥2.8
- Keyword overlap with source

**Result**:
- 70%+ requests finish at steps=1 (~840ms decode)
- Cache hits: 0.1ms decode (instant!)
- Total latency: 1.0s (down from 2.7s)

---

## 📝 Update Protocol

**When adding new models**:
1. Document training data source here
2. Record val cosine and final metrics
3. Note any special training configurations
4. Document known issues/fixes

**When retraining existing models**:
1. Update model file symlink
2. Record new val cosine
3. Note configuration changes
4. Archive old model with timestamp

---

**Last Updated**: October 29, 2025
**Maintainer**: Claude Code (Anthropic)
**Status**: All 6 ports operational with production-grade inference pipeline
