# LVM Data Map: Complete Training & Inference Pipeline

**Last Updated**: October 30, 2025 (OOD Evaluation Fix)
**Purpose**: Comprehensive reference for all LVM (Latent Vector Model) training data, models, and inference pipeline

---

## 📍 Quick Navigation

- [Training Data](#training-data) - Where LVM training sequences come from
- [Trained Models](#trained-models) - 4 production-ready models
- [Inference Pipeline](#inference-pipeline) - How to use models for prediction
- [Evaluation & Benchmarks](#evaluation--benchmarks) - Performance metrics
- [Data Flow](#lvm-data-flow) - End-to-end pipeline visualization

---

## 🎯 LVM Overview

**What is LVM?**
- **Latent Vector Model**: Predicts the next semantic vector from context vectors
- **Tokenless**: Operates in 768D semantic space (no text tokens)
- **Sequential**: Learns temporal/causal relationships in narrative text
- **Vec2Text Compatible**: Output vectors decode back to text via vec2text

**Current Production Status**:
- ✅ 4 trained models (AMN, LSTM, GRU, Transformer)
- ✅ 80k training sequences from Wikipedia
- ✅ Full text→vec→LVM→vec→text pipeline operational
- ✅ Average latency: ~10 seconds (99% from vec2text decoding, 0.4ms from LVM)

---

## Training Data

### ✅ ACTIVE Training Dataset

#### 🆕 NEW: Clean Splits (Article-Based Holdout) - **RECOMMENDED**
```bash
./artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz     # 663 MB - 438k sequences
./artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz   # 27 MB - 18k sequences
./artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz          # 15 MB - 10k OOD test
```

**Why This Is Better** (Oct 30, 2025 Fix):
- ✅ **Article-based splits**: No article appears in both train and val
- ✅ **Representative coherence**: All splits ~0.47 (realistic Wikipedia)
- ✅ **True OOD generalization**: Test articles never seen during training
- ✅ **Proven results**: OOD=0.5622 matches Val=0.5546 (Δ=+0.0076)

**Split Design**:
- Training: Articles 1-1499, 2000-3999, 4500-7671 (438k sequences)
- Validation: Articles 4000-4499 (18k sequences)
- OOD Test: Articles 1500-1999 (10k sequences)
- Removed: Articles 7672-8470 (high-coherence anomaly)

**See**: `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md` for full details

#### Legacy: Wikipedia Training Sequences (80k)
```bash
./artifacts/lvm/training_sequences_ctx5.npz         # 449 MB - 80k sequences (⚠️ random splits)
```

**⚠️ DEPRECATED**: Uses `random_split()` which mixes articles across train/val.
Results in inflated validation scores that don't reflect true generalization.
**Use clean_splits version above instead.**

**Structure**:
```python
import numpy as np
data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)

# Keys and shapes:
data['context_vectors']   # [80000, 5, 768] - 5 previous chunks as context
data['target_vectors']    # [80000, 768]    - Next chunk to predict
data['context_texts']     # [80000, 5]      - Original text (for debugging)
data['target_texts']      # [80000]         - Target text
data['sequence_ids']      # [80000]         - Unique sequence IDs
data['article_ids']       # [80000]         - Source article IDs
```

**Training Statistics**:
- Total sequences: 80,000
- Context window: 5 chunks
- Vector dimension: 768 (GTR-T5 embeddings)
- Source: Wikipedia articles (sequential narrative text)
- Split: 80% train (64k), 20% val (16k)

#### Source: Wikipedia Vectors
```bash
./artifacts/wikipedia_500k_corrected_vectors.npz    # 230 MB - Raw corpus
```

**How Training Data Was Created**:
1. Start with 500k Wikipedia chunks (sequential from articles)
2. Encode each chunk to 768D vector using GTR-T5
3. Create sliding windows: [chunk_i-5, ..., chunk_i-1] → chunk_i
4. Filter for quality (min cosine similarity, max length, etc.)
5. Result: 80k high-quality training sequences

---

### 🗑️ DEPRECATED Training Data (Do NOT Use)

#### Ontology Data (Wrong for LVM Training)
```bash
./artifacts/archive/ontological_DEPRECATED_20251011/wordnet_training_sequences.npz

# ❌ DO NOT USE FOR LVM TRAINING
# Reason: Taxonomic hierarchies (is-a, part-of), not sequential narrative
# See: CLAUDE.md rules and docs/LVM_TRAINING_CRITICAL_FACTS.md
```

**Why Ontologies Are Wrong**:
- Ontologies teach classification ("dog → mammal → animal")
- LVMs need temporal/causal flow ("First... → Then... → Finally...")
- Ontologies: taxonomic structure
- LVMs: sequential prediction

**Correct Data Sources for LVM**:
- ✅ Wikipedia articles (narrative progression)
- ✅ Textbooks (sequential instruction)
- ✅ Scientific papers (methods → results → conclusions)
- ✅ Programming tutorials (step-by-step)
- ❌ WordNet, SWO, GO, DBpedia ontologies

---

## Trained Models

### ✅ ACTIVE Production Models

#### 🆕 NEW: Clean Splits Models (October 30, 2025)

**AMN Clean Splits** - **RECOMMENDED FOR SEQUENTIAL PREDICTION**
```bash
Location: ./artifacts/lvm/models/amn_clean_splits_20251030_204541/best_model_fixed.pt
```

**Performance**:
- **Val Cosine**: 0.5546 (honest validation on held-out articles)
- **OOD Cosine**: 0.5622 (true generalization!)
- **Delta**: +0.0076 (essentially zero - proves generalization)
- Latency: ~0.5 ms/query
- Parameters: 1.5M

**Training**:
- **Data**: 438k sequences with article-based splits
- **Training**: Articles 1-1499, 2000-3999, 4500-7671
- **Validation**: Articles 4000-4499 (never seen in training)
- **OOD Test**: Articles 1500-1999 (truly held-out)
- **Loss**: MSE
- **Epochs**: 20
- **Device**: MPS

**Why This Is Better**:
- ✅ True OOD generalization (OOD matches Val)
- ✅ Article-based splits (no data contamination)
- ✅ Representative coherence (~0.47 across all splits)
- ❌ **Not compatible with chat repeat-pad mode** (use LSTM/GRU/Transformer for chat)

**When to Use**:
- Wikipedia sequence prediction (original use case)
- Document completion with sequential context
- Any application with proper sequential context (not repeat-pad)

**See**: `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md` for full details

---

#### Legacy Models (October 16, 2025) - ⚠️ Random Splits

All legacy models trained with:
- **Loss**: MSE (Mean Squared Error)
- **Data**: 80k Wikipedia sequences (**⚠️ random splits, data contamination**)
- **Epochs**: 20
- **Device**: Apple M1 Max (MPS)
- **Note**: Validation scores inflated due to article mixing

---

### 1. AMN (Attention Mixture Network) - Legacy 790k

**✅ BEST FOR**: Ultra-low latency, batch processing, interpretability

```bash
Location: ./artifacts/lvm/models/amn_20251016_133427/
```

**Performance**:
- Val Cosine: **0.5664**
- Latency: **0.49 ms/query** ⚡ (fastest)
- Parameters: **1.5M** (smallest)
- Memory: **5.8 MB**
- Batch speedup: **138x** (best scaling)

**Architecture**:
- Residual learning over linear baseline
- Attention-weighted context mixing
- Learned residual correction
- Output: normalize(baseline + residual)

**Usage**:
```python
import torch
from app.lvm.models import create_model

checkpoint = torch.load('artifacts/lvm/models/amn_20251016_133427/best_model.pt')
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
ctx = torch.randn(1, 5, 768)  # [batch, 5 context, 768D]
pred = model(ctx)              # [batch, 768D]
```

**When to Use**:
- Real-time inference (<1ms required)
- Batch processing (138x speedup)
- Resource-constrained environments
- Interpretable predictions (attention weights)

---

### 2. LSTM (Baseline)

**✅ BEST FOR**: Production deployment, best balance

```bash
Location: ./artifacts/lvm/models/lstm_20251016_133934/
```

**Performance**:
- Val Cosine: **0.5758** (2nd best accuracy)
- Latency: **0.56 ms/query** ⚡
- Parameters: **5.1M**
- Memory: **19.5 MB**
- Batch speedup: 63x

**Architecture**:
- 2-layer bidirectional LSTM
- 512 hidden dimensions
- Layer normalization
- Dropout: 0.2

**Usage**:
```python
checkpoint = torch.load('artifacts/lvm/models/lstm_20251016_133934/best_model.pt')
model = create_model('lstm', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**When to Use**:
- ⭐ **Recommended for production** (best balance)
- Good accuracy + low latency
- Proven architecture, easy to deploy
- Stable training, good generalization

---

### 3. GRU (Stack)

**✅ BEST FOR**: Middle ground between LSTM and Transformer

```bash
Location: ./artifacts/lvm/models/gru_20251016_134451/
```

**Performance**:
- Val Cosine: **0.5754**
- Latency: **2.08 ms/query**
- Parameters: **7.1M**
- Memory: **27.1 MB**
- Batch speedup: 79x

**Architecture**:
- 4-layer GRU with residual connections
- 512 hidden dimensions per layer
- Layer normalization
- Dropout: 0.2

**Usage**:
```python
checkpoint = torch.load('artifacts/lvm/models/gru_20251016_134451/best_model.pt')
model = create_model('gru', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**When to Use**:
- Need more capacity than LSTM
- Still want relatively fast inference
- Residual connections for deeper networks

---

### 4. Transformer

**✅ BEST FOR**: Maximum accuracy

```bash
Location: ./artifacts/lvm/models/transformer_20251016_135606/
```

**Performance**:
- Val Cosine: **0.5820** 🏆 (best accuracy)
- Latency: **2.68 ms/query**
- Parameters: **17.9M** (largest)
- Memory: **68.4 MB**
- Batch speedup: 75x

**Architecture**:
- 4-layer transformer encoder
- 8 attention heads
- 2048 feedforward dimension
- Dropout: 0.1

**Usage**:
```python
checkpoint = torch.load('artifacts/lvm/models/transformer_20251016_135606/best_model.pt')
model = create_model('transformer', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**When to Use**:
- Accuracy is critical
- Latency <3ms acceptable
- Worth +1.6% accuracy vs AMN
- In full pipeline: 2.68ms LVM vs 10,000ms total (0.03% difference)

---

## Model Performance Comparison

| Model | Val Cosine | ms/Query | Params | Memory | Efficiency* | Recommended For |
|-------|-----------|----------|--------|--------|-------------|-----------------|
| **Transformer** | **0.5820** 🏆 | 2.68 | 17.9M | 68.4 MB | 217 | Maximum accuracy |
| **LSTM** | **0.5758** ⭐ | 0.56 | 5.1M | 19.5 MB | **1035** | **Production** |
| **GRU** | **0.5754** | 2.08 | 7.1M | 27.1 MB | 277 | Middle ground |
| **AMN** | **0.5664** ⚡ | **0.49** | **1.5M** | **5.8 MB** | **1146** 🏆 | Ultra-low latency |

*Efficiency = (Val Cosine / ms/Q) × 1000

### Baseline Comparison
- **Linear Average Baseline**: 0.5462 cosine
- **All models beat baseline**: +2-7% improvement
- **Best improvement**: Transformer (+6.5%)

---

## Inference Pipeline

### Full Text→Text Pipeline

```
Input Text (5 chunks)
    ↓
┌────────────────────────┐
│ 1. Text Encoding       │  Vec2Text Encoder (GTR-T5)
│    5 texts → 5 vectors │  ~100ms per chunk = ~300ms total
└────────────────────────┘
    ↓
Context Vectors [5, 768]
    ↓
┌────────────────────────┐
│ 2. LVM Inference       │  AMN/LSTM/GRU/Transformer
│    [5,768] → [768]     │  0.49-2.68ms ⚡
└────────────────────────┘
    ↓
Predicted Vector [768]
    ↓
┌────────────────────────┐
│ 3. Vector Decoding     │  Vec2Text Decoder (JXE/IELab)
│    [768] → Text        │  ~10,000ms (bottleneck) ⏱️
└────────────────────────┘
    ↓
Output Text (predicted chunk)
```

**Pipeline Latency Breakdown**:
- Encoding: ~300 ms (3%)
- LVM: **0.5-2.7 ms** (0.03%)
- Decoding: **~10,000 ms** (97%)
- **TOTAL: ~10,300 ms per prediction**

**Key Insight**: Vec2text decoding is the bottleneck, not LVM!

---

### Python API

#### Option 1: Direct LVM Inference (Vectors Only)
```python
import torch
import numpy as np
from app.lvm.models import create_model

# Load model
checkpoint = torch.load('artifacts/lvm/models/lstm_20251016_133934/best_model.pt')
model = create_model('lstm', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference (assuming you already have vectors)
context_vectors = torch.randn(1, 5, 768)  # Your 5 context vectors
with torch.no_grad():
    predicted_vector = model(context_vectors)  # [1, 768]

print(f"Predicted vector shape: {predicted_vector.shape}")
```

#### Option 2: Full Text→Text Pipeline
```python
import sys
sys.path.insert(0, 'app/lvm')
sys.path.insert(0, 'app/vect_text_vect')

from models import create_model
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Load LVM
checkpoint = torch.load('artifacts/lvm/models/lstm_20251016_133934/best_model.pt')
lvm_model = create_model('lstm', input_dim=768, d_model=256, hidden_dim=512)
lvm_model.load_state_dict(checkpoint['model_state_dict'])
lvm_model.eval()

# Load vec2text orchestrator
orch = IsolatedVecTextVectOrchestrator(steps=1, debug=False)

# Encode context text
context_texts = [
    "First chunk of text.",
    "Second chunk of text.",
    "Third chunk of text.",
    "Fourth chunk of text.",
    "Fifth chunk of text."
]
context_vectors = orch.encode_texts(context_texts)  # [5, 768]

# LVM prediction
if isinstance(context_vectors, torch.Tensor):
    context_vectors = context_vectors.cpu().numpy()
ctx_tensor = torch.from_numpy(context_vectors).float().unsqueeze(0)  # [1, 5, 768]
with torch.no_grad():
    pred_vector = lvm_model(ctx_tensor).cpu().numpy()[0]  # [768]

# Decode to text
result = orch._run_subscriber_subprocess(
    'jxe',
    torch.from_numpy(pred_vector).unsqueeze(0).cpu(),
    metadata={'original_texts': [' ']},
    device_override='cpu'
)
predicted_text = result['result'][0]
print(f"Predicted next chunk: {predicted_text}")
```

---

## Evaluation & Benchmarks

### Performance Files

```bash
# Comprehensive benchmark results
./artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md       # All 4 models compared
./artifacts/lvm/PIPELINE_ARCHITECTURE_AND_PERFORMANCE.md  # Pipeline analysis
./artifacts/lvm/benchmark_results.json             # Raw benchmark data
./artifacts/lvm/FINAL_RESULTS_Oct16_2025.md        # Training summary
```

### Evaluation Metrics

#### Vector Similarity (Primary)
- **Metric**: Cosine similarity between predicted and target vectors
- **Range**: -1 to 1 (higher is better)
- **Baseline**: 0.5462 (linear average)
- **Best model**: 0.5820 (Transformer)

#### Text Quality (Secondary)
- **ROUGE-1**: Unigram overlap (avg: 0.112)
- **ROUGE-2**: Bigram overlap (avg: 0.020)
- **ROUGE-L**: Longest common subsequence (avg: 0.089)
- **BLEU**: N-gram precision (avg: 0.007)

**Note**: Low ROUGE/BLEU scores are expected because:
- LVM predicts semantic vectors, not exact words
- Vec2text reconstruction introduces variation
- This is a semantic prediction task, not text generation

### Example Test Script

```bash
# Run full pipeline test with scoring
./venv/bin/python3 tools/lvm_text_output_examples.py

# Outputs:
# - 10 sample predictions with context
# - Vector cosine scores
# - ROUGE-1/2/L scores
# - BLEU scores
# - Latency breakdown
```

---

## LVM Data Flow

### Complete End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ RAW DATA: Wikipedia Articles                                │
│ Location: ./data/datasets/wikipedia/                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────┐
          │ Chunking & Preprocessing   │
          │ Split into semantic chunks │
          └───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STRUCTURED DATA: PostgreSQL                                 │
│ 80,636 concepts with metadata                               │
│ Location: /opt/homebrew/var/postgresql@17/                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────┐
          │ GTR-T5 Encoding (768D)     │
          │ Vec2text-compatible method │
          └───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ VECTOR STORE: Wikipedia 500k Vectors (NPZ)                  │
│ 230 MB, [500k, 768] embeddings + text + IDs                │
│ Location: ./artifacts/wikipedia_500k_corrected_vectors.npz  │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────┐
          │ Create Training Sequences  │
          │ Sliding window: [5 ctx→1] │
          └───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING DATA: LVM Sequences (NPZ)                          │
│ 449 MB, 80k sequences [context_5x768 → target_768]         │
│ Location: ./artifacts/lvm/training_sequences_ctx5.npz       │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────┐
          │ Train LVM Models           │
          │ MSE Loss, 20 epochs        │
          └───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINED MODELS: 4 LVM Architectures                        │
│ ~200 MB total (AMN, LSTM, GRU, Transformer)                │
│ Location: ./artifacts/lvm/models/                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
          ┌───────────────────────────┐
          │ Inference Pipeline         │
          │ Text→Vec→LVM→Vec→Text      │
          └───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Predicted Next Chunk                                │
│ Semantic continuation of input context                      │
└─────────────────────────────────────────────────────────────┘
```

### FAISS Index (Parallel Path for Retrieval)

```
Wikipedia 500k Vectors (NPZ)
            ↓
┌────────────────────────────┐
│ Build FAISS IVF_FLAT Index │
│ nlist=512, nprobe=16       │
└────────────────────────────┘
            ↓
┌─────────────────────────────────────────────────────────────┐
│ FAISS INDEX: Wikipedia 500k Search                          │
│ 238 MB, 500k vectors, IP similarity                         │
│ Location: ./artifacts/wikipedia_500k_corrected_ivf_flat_ip.index │
└─────────────────────────────────────────────────────────────┘
            ↓
        (Used for)
            ↓
    ┌───────────────┐
    │ vecRAG Search │  Query → Top-K nearest neighbors
    └───────────────┘
```

---

## Critical Rules & Best Practices

### ✅ DO:
- Use Wikipedia or other **sequential narrative data** for LVM training
- Use MSE loss for regression (predicting vectors)
- Normalize vectors before and after LVM
- Use vec2text-compatible GTR-T5 encoding
- Validate on held-out sequential data
- Monitor cosine similarity as primary metric

### ❌ DON'T:
- Use ontology data for LVM training (taxonomic, not sequential)
- Use InfoNCE loss (for contrastive learning, not regression)
- Use sentence-transformers encoding directly (incompatible with vec2text)
- Mix training data from different domains without testing
- Expect high ROUGE/BLEU scores (this is semantic prediction, not text generation)

### 🔍 Data Quality Checks:
```bash
# Check training data integrity
python3 -c "
import numpy as np
data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
print(f'Context shape: {data[\"context_vectors\"].shape}')
print(f'Target shape: {data[\"target_vectors\"].shape}')
print(f'Total sequences: {len(data[\"target_vectors\"])}')
assert data['context_vectors'].shape[1] == 5, 'Context window must be 5'
assert data['context_vectors'].shape[2] == 768, 'Vector dim must be 768'
print('✓ Training data valid')
"
```

---

## Training Scripts

### Train All Models
```bash
# Train all 4 models sequentially
./tools/train_all_lvms.sh

# Individual model training
./.venv/bin/python app/lvm/train_unified.py \
    --model-type lstm \
    --data artifacts/lvm/training_sequences_ctx5.npz \
    --epochs 20 \
    --batch-size 32 \
    --lambda-mse 1.0
```

### Benchmark Models
```bash
# Comprehensive benchmark (all 4 models)
./.venv/bin/python tools/benchmark_lvm_comprehensive.py

# Full pipeline test with text examples
VEC2TEXT_FORCE_PROJECT_VENV=1 TOKENIZERS_PARALLELISM=false \
    ./venv/bin/python3 tools/lvm_text_output_examples.py
```

---

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```python
# Ensure you're using the correct model type
checkpoint = torch.load('path/to/model/best_model.pt')
model = create_model('lstm', input_dim=768, d_model=256, hidden_dim=512)  # Match type!
model.load_state_dict(checkpoint['model_state_dict'])
```

**2. Vec2text Encoding Incompatibility**
```python
# ❌ WRONG: Don't use sentence-transformers directly
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('gtr-t5-base')  # Produces incompatible vectors!

# ✅ CORRECT: Use vec2text orchestrator
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
orch = IsolatedVecTextVectOrchestrator()
vectors = orch.encode_texts(texts)  # Vec2text-compatible!
```

**3. Tensor Device Mismatches**
```python
# Ensure all tensors on same device
context_vectors = context_vectors.to(model.device)
# Or convert to CPU before numpy
pred_vector = model(ctx).cpu().numpy()
```

---

## Summary

### Active LVM Production Stack:
✅ **Training Data**: 80k Wikipedia sequences (449 MB)
✅ **Models**: 4 trained architectures (AMN, LSTM ⭐, GRU, Transformer)
✅ **Inference**: Full text→vec→LVM→vec→text pipeline operational
✅ **Performance**: 0.49-2.68ms LVM latency, 0.5664-0.5820 val cosine
✅ **Bottleneck**: Vec2text decoding (~10s), NOT LVM (0.5ms)

### Recommended Production Setup:
- **Model**: LSTM (best balance of accuracy + speed)
- **Checkpoint**: `artifacts/lvm/models/lstm_20251016_133934/best_model.pt`
- **Val Cosine**: 0.5758
- **Latency**: 0.56 ms/query
- **Full Pipeline**: ~10.3 seconds total

### Next Optimization Targets:
1. 🎯 **Vec2text decoding** (10s → 2-3s via caching/batching)
2. Text encoding (300ms → 100ms via batching)
3. LVM inference (0.5ms already excellent)

---

**See Also**:
- `docs/DATABASE_LOCATIONS.md` - All databases and vector stores
- `docs/DATA_FLOW_DIAGRAM.md` - Visual system architecture
- `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` - Detailed benchmarks
- `CLAUDE.md` - Training rules and critical facts

---

**Last Updated**: October 16, 2025
**Status**: ✅ All systems operational and production-ready
