# PRD: LVM Test Flow Diagrams
**Created**: October 12, 2025
**Purpose**: Visual documentation of LVM testing workflows

---

## Test 2.1: Top-K Retrieval Accuracy (HIGHEST PRIORITY)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LVM TOP-K RETRIEVAL TEST FLOW                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT DATA (artifacts/lvm/training_sequences_ctx5.npz)
┌─────────────────────────────────────────────────────────────────┐
│  Context Sequence (5 vectors)          Target Vector            │
│  ┌──────────┐                          ┌──────────┐            │
│  │ [768D]   │  "The quick brown"       │ [768D]   │ "fox"     │
│  │ [768D]   │  "quick brown fox"       │          │           │
│  │ [768D]   │  "brown fox jumps"       └──────────┘           │
│  │ [768D]   │  "fox jumps over"              ▲                │
│  │ [768D]   │  "jumps over the"              │                │
│  └──────────┘                          Ground Truth            │
│       │                                                         │
│       │ TYPE: float32[5, 768]                                  │
│       │ LLM: None (pre-computed GTR-T5 embeddings)             │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINED LVM MODEL                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Model Type: LSTM / GRU / Transformer                     │  │
│  │  Input:  [1, 5, 768] (batch=1, seq=5, dim=768)           │  │
│  │  Output: [1, 768]    (predicted next vector)             │  │
│  │                                                            │  │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐              │  │
│  │  │ LSTM/   │───▶│  Hidden │───▶│  Linear │              │  │
│  │  │ GRU/    │    │  Layers │    │  +Norm  │              │  │
│  │  │ Trans   │    └─────────┘    └─────────┘              │  │
│  │  └─────────┘                                              │  │
│  │                                                            │  │
│  │  TYPE: PyTorch Model (float32)                           │  │
│  │  LLM: None (pure vector transformation)                  │  │
│  │  DEVICE: MPS (Apple Silicon GPU)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Predicted    │
                    │ Vector       │
                    │ [768D]       │
                    └──────┬───────┘
                           │ TYPE: float32[768]
                           │
        ┌──────────────────┴────────────────────┐
        │                                        │
        ▼                                        ▼
┌───────────────────┐                  ┌─────────────────────┐
│  NORMALIZATION    │                  │  NORMALIZATION      │
│  L2-normalize     │                  │  L2-normalize       │
│  prediction       │                  │  target vector      │
└────────┬──────────┘                  └──────────┬──────────┘
         │                                        │
         │ TYPE: float32[768] (unit vector)       │
         │                                        │
         ▼                                        │
┌─────────────────────────────────────────────────┴──────────┐
│            FAISS VECTOR DATABASE (Inner Product)            │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Database: wikipedia_42113_ordered.npz             │    │
│  │  Vectors:  42,113 concepts × 768D                  │    │
│  │  Index:    IndexFlatIP (cosine similarity)         │    │
│  │                                                      │    │
│  │  Contents:                                          │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐ │    │
│  │  │ Vec 0  │ │ Vec 1  │ │ Vec 2  │ ... │ Vec N  │ │    │
│  │  │[768D]  │ │[768D]  │ │[768D]  │     │[768D]  │ │    │
│  │  │"The"   │ │"quick" │ │"brown" │     │"fox"   │ │    │
│  │  └────────┘ └────────┘ └────────┘     └────────┘ │    │
│  │                                                      │    │
│  │  TYPE: numpy float32[42113, 768]                   │    │
│  │  LLM: None (search is vector similarity)           │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │  K-NN SEARCH           │
            │  (K = 1, 5, 10, 20)    │
            │                        │
            │  Returns: Top-K        │
            │  indices & distances   │
            └────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Top-K Results                 │
        │  ┌──────────────────────────┐  │
        │  │ Rank 1: idx=4521, d=0.89 │  │
        │  │ Rank 2: idx=1234, d=0.85 │  │
        │  │ Rank 3: idx=9876, d=0.82 │  │
        │  │ ...                       │  │
        │  │ Rank K: idx=5555, d=0.65 │  │
        │  └──────────────────────────┘  │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  FIND TARGET IN DATABASE       │
        │  (normalize target, search)    │
        │                                │
        │  Target index: 7890            │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  ACCURACY CHECK                │
        │                                │
        │  Is target_idx in top-K?       │
        │  ✓ Top-1:  idx in [4521]       │
        │  ✓ Top-5:  idx in [4521...K5]  │
        │  ✓ Top-10: idx in [4521...K10] │
        │  ✓ Top-20: idx in [4521...K20] │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  AGGREGATE RESULTS             │
        │  (Repeat for all test samples) │
        │                                │
        │  Top-1:  25/100 = 25%          │
        │  Top-5:  52/100 = 52%          │
        │  Top-10: 68/100 = 68%          │
        │  Top-20: 79/100 = 79%          │
        │                                │
        │  TYPE: JSON metrics            │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  SAVE RESULTS                  │
        │  artifacts/lvm/evaluation/     │
        │  retrieval_results.json        │
        └────────────────────────────────┘

LEGEND:
  [768D]  = 768-dimensional float32 vector (GTR-T5 embedding)
  TYPE    = Data type at this stage
  LLM     = Language model used (if any)
  DEVICE  = Hardware accelerator
```

---

## Test 3.1: Vec2Text Integration (Full Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FULL VECTOR → TEXT GENERATION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: CONTEXT PREPARATION
┌──────────────────────────────────────────────────┐
│  Input: 5 Wikipedia concepts (text)              │
│  ┌────────────────────────────────────────────┐  │
│  │ 1. "Neural networks are computing systems" │  │
│  │ 2. "computing systems inspired by brains"  │  │
│  │ 3. "inspired by brains biological neural"  │  │
│  │ 4. "biological neural networks animals"    │  │
│  │ 5. "networks animals comprise neurons"     │  │
│  └────────────────────────────────────────────┘  │
│  TYPE: List[str]                                 │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  GTR-T5 ENCODER       │
         │  (Pre-computed)       │
         │                       │
         │  sentence-transformers │
         │  /gtr-t5-base         │
         │                       │
         │  LLM: GTR-T5 (350M)   │
         │  DEVICE: MPS/CPU      │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Context Vectors     │
         │  [5, 768]            │
         │  TYPE: float32       │
         └──────────┬───────────┘
                    │
                    │
┌───────────────────┴────────────────────────────────┐
│                                                     │
│           LVM MODEL (TRAINED)                      │
│  ┌──────────────────────────────────────────────┐ │
│  │  LSTM / GRU / Transformer                    │ │
│  │  Input:  [1, 5, 768]                         │ │
│  │  Output: [1, 768] (predicted next vector)   │ │
│  │                                               │ │
│  │  LLM: None (vector transformer only)         │ │
│  │  DEVICE: MPS                                  │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────┬───────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │  Predicted Vector    │
         │  [768D]              │
         │  (next concept)      │
         └──────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  VEC2TEXT (JXE)  │    │  VEC2TEXT (IELab)│
│  ┌────────────┐  │    │  ┌────────────┐  │
│  │ Corrector  │  │    │  │ Corrector  │  │
│  │ Model      │  │    │  │ Model      │  │
│  │ (seq2seq)  │  │    │  │ (seq2seq)  │  │
│  └────────────┘  │    │  └────────────┘  │
│                  │    │                  │
│  Input: [768D]   │    │  Input: [768D]   │
│  Output: tokens  │    │  Output: tokens  │
│                  │    │                  │
│  LLM: T5-based   │    │  LLM: Different  │
│  Steps: 1-5      │    │  Steps: 1-5      │
│  DEVICE: CPU     │    │  DEVICE: CPU     │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
  ┌─────────────┐         ┌─────────────┐
  │ JXE Output  │         │IELab Output │
  │ "comprise"  │         │"neurons are"│
  │ "neurons"   │         │"fundamental"│
  └─────────────┘         └─────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌──────────────────────┐
         │  COMPARISON          │
         │  Ground truth:       │
         │  "comprise neurons"  │
         │                      │
         │  JXE:    "comprise   │
         │           neurons"   │
         │  Match: ✓ EXACT      │
         │                      │
         │  IELab:  "neurons are│
         │           fundamental"│
         │  Match: ✓ PARTIAL    │
         │                      │
         │  TYPE: Text strings  │
         └──────────────────────┘

METRICS COMPUTED:
  - BLEU score (token overlap)
  - ROUGE score (n-gram overlap)
  - Semantic similarity (re-encode and compare vectors)
  - Exact match (%)
```

---

## Test 1.3: Inference Speed Benchmark

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE SPEED BENCHMARK FLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

SETUP PHASE
┌────────────────────────────────────┐
│  Model: LSTM / GRU / Transformer   │
│  Load to device: MPS               │
│  Set to eval mode                  │
│  TYPE: PyTorch model               │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│  WARMUP (10 iterations)            │
│  Purpose: Initialize GPU kernels   │
│  Dummy input: [32, 5, 768]         │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│  BENCHMARK LOOP (100 iterations per batch size)        │
│                                                         │
│  For batch_size in [1, 8, 16, 32, 64]:                │
│    ┌─────────────────────────────────────────────┐    │
│    │  1. Create input: [batch_size, 5, 768]      │    │
│    │     TYPE: torch.FloatTensor                 │    │
│    │                                              │    │
│    │  2. Synchronize GPU: torch.mps.synchronize()│    │
│    │                                              │    │
│    │  3. Start timer: time.time()                │    │
│    │                                              │    │
│    │  4. Forward pass: model(input)              │    │
│    │     (100 iterations)                        │    │
│    │                                              │    │
│    │  5. Synchronize GPU: torch.mps.synchronize()│    │
│    │                                              │    │
│    │  6. Stop timer: time.time()                 │    │
│    │                                              │    │
│    │  7. Compute metrics:                        │    │
│    │     - ms per batch = (total_time / 100)     │    │
│    │     - samples/sec  = batch_size / time      │    │
│    └─────────────────────────────────────────────┘    │
│                                                         │
│  NO LLM: Pure tensor operations                       │
│  DEVICE: MPS (Apple Silicon)                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │  RESULTS TABLE             │
            │                            │
            │  Batch  │  ms/batch │ samp/sec │
            │  ──────────────────────────     │
            │    1    │   2.3     │  435      │
            │    8    │   5.1     │  1569     │
            │   16    │   8.9     │  1798     │
            │   32    │  16.2     │  1975     │
            │   64    │  31.5     │  2032     │
            │                            │
            │  TYPE: Performance metrics │
            └────────────────────────────┘

COMPARISON OUTPUT:
┌──────────────────────────────────────────────────┐
│  Model Performance Comparison (batch=32)         │
│  ┌──────────────────────────────────────────┐   │
│  │ LSTM:        16.2 ms/batch (1975 samp/s) │   │
│  │ GRU:         18.5 ms/batch (1730 samp/s) │   │
│  │ Transformer: 45.3 ms/batch ( 706 samp/s) │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  Winner: LSTM (2.8x faster than Transformer)    │
└──────────────────────────────────────────────────┘
```

---

## Component Summary Table

| Component | Type | LLM Used | Device | Purpose |
|-----------|------|----------|---------|---------|
| **GTR-T5 Encoder** | Text→Vector | GTR-T5 (350M) | MPS/CPU | Convert text to 768D vectors |
| **Training Data** | Vector | None (pre-computed) | Disk | Context sequences for LVM |
| **LVM Model** | Vector→Vector | None | MPS | Predict next vector |
| **FAISS Index** | Vector Search | None | CPU | K-NN similarity search |
| **Vec2Text (JXE)** | Vector→Text | T5-based corrector | CPU | Decode vector to text |
| **Vec2Text (IELab)** | Vector→Text | Different corrector | CPU | Alternative decoder |

**Total Pipeline**: Text → GTR-T5 → Vectors → LVM → Predicted Vector → Vec2Text → Text

---

**Last Updated**: October 12, 2025
