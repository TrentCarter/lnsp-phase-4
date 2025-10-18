# LNSP Data Flow Architecture

**Last Updated**: October 16, 2025
**Purpose**: Visual architecture showing data flow through the complete LNSP system

---

## 🎯 System Overview

LNSP (Latent-space Next-Sequence Prediction) is a **tokenless, vector-native retrieval and prediction system** that operates entirely in semantic vector space.

**Key Components**:
1. **vecRAG**: Vector-based retrieval (FAISS + PostgreSQL)
2. **LVM**: Latent Vector Models (predict next semantic vector)
3. **Vec2Text**: Bidirectional text↔vector conversion
4. **GraphRAG**: (Planned) Graph-based ontology traversal via Neo4j

---

## 📊 Complete System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         RAW DATA SOURCES                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ Wikipedia Articles (500k)          🗑️ FactoidWiki (deprecated)         │
│  Location: data/datasets/wikipedia/   Location: data/factoidwiki_1k.jsonl │
│  Purpose: LVM training (sequential)   Purpose: DO NOT USE (taxonomic)     │
│                                                                             │
│  🔄 Ontologies (SWO, GO, DBpedia)     │
│  Location: artifacts/ontology_chains/  │
│  Purpose: GraphRAG only, NOT for LVM   │
│                                        │
└────────────────────────────────────────┘
            │
            ↓
┌────────────────────────────────────────┐
│   PREPROCESSING PIPELINE                │
├────────────────────────────────────────┤
│                                         │
│  • Chunking (semantic boundaries)      │
│  • CPESH generation (LLM)              │
│  • Metadata extraction                 │
│  • Quality filtering                   │
│                                         │
└────────────────────────────────────────┘
            │
            ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                    PRIMARY STORAGE LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────┐    ┌────────────────────────────┐   │
│  │  PostgreSQL (ACTIVE)             │    │  Neo4j (EMPTY)             │   │
│  │  ✅ 80,636 concepts               │    │  ⚠️ 0 nodes                 │   │
│  │  Location: /opt/homebrew/var/    │    │  Location: /opt/homebrew/  │   │
│  │           postgresql@17/          │    │           var/neo4j/       │   │
│  │                                   │    │                            │   │
│  │  Tables:                          │    │  Planned Use:              │   │
│  │  • cpe_entry (concepts+metadata)  │    │  • Graph relationships     │   │
│  │  • cpe_vectors (768D vectors)     │    │  • Ontology traversal      │   │
│  │  • tmd_codes (domain/task/mod)    │    │  • 6-degree shortcuts      │   │
│  │                                   │    │                            │   │
│  └──────────────────────────────────┘    └────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
            │
            ↓
┌────────────────────────────────────────┐
│   GTR-T5 ENCODING LAYER                 │
├────────────────────────────────────────┤
│                                         │
│  ✅ CORRECT: Vec2Text-Compatible        │
│  Method: IsolatedVecTextVectOrchestrator│
│  • T5EncoderModel (transformers)       │
│  • Mean pooling                        │
│  • L2 normalization                    │
│  • Output: 768D vectors                │
│                                         │
│  ❌ WRONG: sentence-transformers        │
│  Produces incompatible vectors!        │
│  (9.9x worse vec2text decoding)        │
│                                         │
└────────────────────────────────────────┘
            │
            ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                    VECTOR STORAGE LAYER                                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  NPZ Vector Files (ACTIVE)                                      │      │
│  │  ✅ wikipedia_500k_corrected_vectors.npz (230 MB)               │      │
│  │     • vectors: [500k, 768] float32                              │      │
│  │     • texts: [500k] strings                                     │      │
│  │     • ids: [500k] UUIDs → correlates with PostgreSQL            │      │
│  │                                                                  │      │
│  │  ✅ training_sequences_ctx5.npz (449 MB)                        │      │
│  │     • context_vectors: [80k, 5, 768] - LVM training input      │      │
│  │     • target_vectors: [80k, 768] - LVM training targets        │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │  FAISS Index (ACTIVE)                                           │      │
│  │  ✅ wikipedia_500k_corrected_ivf_flat_ip.index (238 MB)         │      │
│  │     • Type: IVF_FLAT_IP                                         │      │
│  │     • Vectors: 500k                                             │      │
│  │     • Dimension: 768                                            │      │
│  │     • nlist: 512, nprobe: 16                                    │      │
│  │                                                                  │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
            │
            ├─────────────────────────────┐
            │                             │
            ↓ (Path A: vecRAG)            ↓ (Path B: LVM Training)
┌────────────────────────────┐   ┌────────────────────────────────┐
│  vecRAG RETRIEVAL           │   │  LVM TRAINING PIPELINE         │
├────────────────────────────┤   ├────────────────────────────────┤
│                             │   │                                 │
│  Query Text                 │   │  1. Create Training Sequences  │
│      ↓                      │   │     Sliding window: [5→1]      │
│  GTR-T5 Encode              │   │                                 │
│      ↓                      │   │  2. Train 4 Architectures:     │
│  Query Vector [768]         │   │     • AMN (1.5M params)        │
│      ↓                      │   │     • LSTM (5.1M params)       │
│  FAISS Search               │   │     • GRU (7.1M params)        │
│      ↓                      │   │     • Transformer (17.9M)      │
│  Top-K Neighbors            │   │                                 │
│      ↓                      │   │  3. Loss: MSE (regression)     │
│  Get CPE IDs                │   │     20 epochs                  │
│      ↓                      │   │                                 │
│  PostgreSQL Lookup          │   │  4. Save Best Models           │
│      ↓                      │   │                                 │
│  Return Concept Texts       │   └────────────────────────────────┘
│                             │                   │
└────────────────────────────┘                   ↓
                                   ┌────────────────────────────────┐
                                   │  TRAINED LVM MODELS (ACTIVE)   │
                                   ├────────────────────────────────┤
                                   │                                 │
                                   │  ./artifacts/lvm/models/       │
                                   │                                 │
                                   │  ✅ AMN (0.5664 cosine)         │
                                   │     0.49 ms/query ⚡            │
                                   │                                 │
                                   │  ✅ LSTM (0.5758 cosine) ⭐      │
                                   │     0.56 ms/query              │
                                   │     RECOMMENDED                │
                                   │                                 │
                                   │  ✅ GRU (0.5754 cosine)         │
                                   │     2.08 ms/query              │
                                   │                                 │
                                   │  ✅ Transformer (0.5820) 🏆     │
                                   │     2.68 ms/query              │
                                   │     Best accuracy              │
                                   │                                 │
                                   └────────────────────────────────┘
                                               │
                                               ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                    LVM INFERENCE PIPELINE                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: 5 Context Chunks (text)                                            │
│      │                                                                      │
│      ↓                                                                      │
│  ┌─────────────────────────────────────┐                                   │
│  │  STEP 1: Text Encoding              │    ~300 ms (3%)                   │
│  │  GTR-T5 Encoder (vec2text method)   │                                   │
│  │  5 texts → 5 vectors [5, 768]       │                                   │
│  └─────────────────────────────────────┘                                   │
│      │                                                                      │
│      ↓                                                                      │
│  ┌─────────────────────────────────────┐                                   │
│  │  STEP 2: LVM Prediction             │    ~0.5 ms (0.05%) ⚡             │
│  │  LSTM Model                          │                                   │
│  │  [5, 768] → [768]                   │                                   │
│  └─────────────────────────────────────┘                                   │
│      │                                                                      │
│      ↓                                                                      │
│  ┌─────────────────────────────────────┐                                   │
│  │  STEP 3: Vector Decoding            │    ~10,000 ms (97%) ⏱️            │
│  │  Vec2Text Decoder (JXE)             │    ** BOTTLENECK **              │
│  │  [768] → Text                       │                                   │
│  └─────────────────────────────────────┘                                   │
│      │                                                                      │
│      ↓                                                                      │
│  Output: Predicted Next Chunk (text)                                       │
│                                                                             │
│  TOTAL LATENCY: ~10,300 ms per prediction                                  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Synchronization (PostgreSQL ↔ FAISS)

```
┌─────────────────────────────────────────┐
│  INGESTION API (Port 8004)              │
│  Atomic writes to PostgreSQL + FAISS    │
└─────────────────────────────────────────┘
                │
                ↓
        ┌───────────────┐
        │ Concept Entry │
        │ + Metadata    │
        └───────────────┘
                │
        ┌───────┴───────┐
        ↓               ↓
┌──────────────┐   ┌─────────────┐
│ PostgreSQL   │   │ FAISS       │
│              │   │             │
│ INSERT INTO  │   │ index.add() │
│ cpe_entry    │   │             │
│              │   │ index.save()│ ← CRITICAL!
│              │   │             │
└──────────────┘   └─────────────┘
        │               │
        └───────┬───────┘
                ↓
        ✅ Synchronized
   CPE ID links both stores
```

**Critical Rules**:
- Every concept MUST have a unique CPE ID (UUID)
- PostgreSQL entry and FAISS vector MUST be written atomically
- FAISS index MUST call `.save()` after `.add()` (Oct 4 fix)
- NPZ files correlate: vector index → CPE ID → concept text

**Verification**:
```bash
# Check sync status
psql -U lnsp -d lnsp -c "SELECT COUNT(*) FROM cpe_entry;"
# Result: 80,636

python3 -c "import faiss; idx = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index'); print(idx.ntotal)"
# Result: ~500,000 (includes variations)
```

---

## 🧠 LVM Training Data Flow

```
Wikipedia Articles (500k chunks)
          │
          ↓
┌──────────────────────────┐
│ Sequential Ordering      │
│ Maintain article order   │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────┐
│ GTR-T5 Encoding          │
│ Each chunk → 768D vector │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ wikipedia_500k_corrected_vectors.npz │
│ [500k, 768] + texts + IDs            │
└──────────────────────────────────────┘
          │
          ↓
┌──────────────────────────┐
│ Create Training Pairs    │
│ Sliding window:          │
│ [chunk_i-5 ... chunk_i-1]│
│          ↓               │
│      chunk_i             │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ training_sequences_ctx5.npz          │
│ • context: [80k, 5, 768]             │
│ • target: [80k, 768]                 │
└──────────────────────────────────────┘
          │
          ↓
┌──────────────────────────┐
│ Train 4 LVM Models       │
│ MSE Loss, 20 epochs      │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ Best Models (Oct 16, 2025)           │
│ • AMN: 0.5664 (fastest)              │
│ • LSTM: 0.5758 (best balance) ⭐     │
│ • GRU: 0.5754                        │
│ • Transformer: 0.5820 (most accurate)│
└──────────────────────────────────────┘
```

---

## 🔍 vecRAG Query Flow

```
User Query: "What is machine learning?"
          │
          ↓
┌──────────────────────────┐
│ GTR-T5 Encoding          │
│ Text → [768] vector      │
│ (vec2text-compatible)    │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ FAISS Similarity Search              │
│ Query vector vs 500k corpus          │
│ Inner Product (IP) similarity        │
│ IVF search: nlist=512, nprobe=16     │
└──────────────────────────────────────┘
          │
          ↓
┌──────────────────────────┐
│ Top-K Results            │
│ [k] indices + distances  │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────┐
│ Map Index → CPE ID       │
│ Using NPZ metadata       │
└──────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ PostgreSQL Lookup                    │
│ SELECT * FROM cpe_entry              │
│ WHERE cpe_id IN (...)                │
└──────────────────────────────────────┘
          │
          ↓
┌──────────────────────────────────────┐
│ Results with Full Metadata           │
│ • Concept text                       │
│ • Probe questions                    │
│ • Expected answers                   │
│ • CPESH negatives                    │
│ • Cosine similarity scores           │
└──────────────────────────────────────┘
          │
          ↓
Return to User
```

---

## 🎯 LVM Inference Flow (Text→Text)

```
User Provides Context:
  "Chunk 1: AI is...",
  "Chunk 2: Machine learning...",
  "Chunk 3: Neural networks...",
  "Chunk 4: Deep learning...",
  "Chunk 5: Training data..."
          │
          ↓
┌───────────────────────────────────────┐
│ ENCODING (300ms)                      │
│ GTR-T5 Encoder                        │
│ 5 texts → 5 vectors [5, 768]         │
│ Method: IsolatedVecTextVectOrchestrator │
└───────────────────────────────────────┘
          │
          ↓
┌───────────────────────────────────────┐
│ LVM INFERENCE (0.5ms) ⚡               │
│ LSTM Model                            │
│ Input: [1, 5, 768] (batch=1)         │
│ Output: [1, 768] predicted vector     │
│ model.eval(), torch.no_grad()         │
└───────────────────────────────────────┘
          │
          ↓
┌───────────────────────────────────────┐
│ DECODING (10,000ms) ⏱️                 │
│ Vec2Text Decoder (JXE)                │
│ Iterative refinement (1 step default) │
│ Input: [768] vector                   │
│ Output: Text string                   │
│ ** THIS IS THE BOTTLENECK **          │
└───────────────────────────────────────┘
          │
          ↓
Predicted Next Chunk:
  "Models require supervised learning..."

QUALITY METRICS:
  • Vector cosine: 0.45-0.55 (semantic alignment)
  • ROUGE-1: 0.10-0.15 (low expected)
  • BLEU: 0.00-0.02 (very low expected)

Note: Low ROUGE/BLEU is normal!
This is semantic prediction, not text generation.
```

---

## 📦 Storage Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     PERSISTENT STORAGE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tier 1: Structured Data                                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ PostgreSQL: /opt/homebrew/var/postgresql@17/         │      │
│  │ • cpe_entry: 80,636 rows (concepts, metadata)        │      │
│  │ • cpe_vectors: 80,636 rows (768D vectors)            │      │
│  │ • Size: ~1.5 GB                                       │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Tier 2: Vector Indexes                                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ FAISS: ./artifacts/*.index                            │      │
│  │ • wikipedia_500k_corrected_ivf_flat_ip.index          │      │
│  │ • Size: 238 MB                                        │      │
│  │ • Vectors: 500k                                       │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Tier 3: Raw Vectors                                            │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ NPZ Files: ./artifacts/*.npz                          │      │
│  │ • wikipedia_500k_corrected_vectors.npz (230 MB)      │      │
│  │ • training_sequences_ctx5.npz (449 MB)               │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Tier 4: Trained Models                                         │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ LVM Models: ./artifacts/lvm/models/                   │      │
│  │ • 4 architectures × ~50-200 MB each                  │      │
│  │ • Total: ~200 MB                                      │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Tier 5: Caches                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ SQLite: ./artifacts/*.db                              │      │
│  │ • cpesh_index.db (20 KB)                              │      │
│  │ • mlflow.db                                           │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

TOTAL ACTIVE STORAGE: ~2.4 GB
```

---

## ⚡ Performance Characteristics

### Latency Breakdown by Component

```
┌────────────────────────────────────────────────────────────┐
│ COMPONENT LATENCY (single query)                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  GTR-T5 Encoding (per chunk):        ~50-100 ms           │
│  ├─ Tokenization:                     ~5 ms               │
│  ├─ Transformer forward:              ~40 ms              │
│  └─ Pooling + normalization:          ~5 ms               │
│                                                             │
│  LVM Inference:                       0.49-2.68 ms ⚡      │
│  ├─ AMN:                              0.49 ms (fastest)   │
│  ├─ LSTM:                             0.56 ms (best)      │
│  ├─ GRU:                              2.08 ms             │
│  └─ Transformer:                      2.68 ms (best acc)  │
│                                                             │
│  Vec2Text Decoding (JXE):             ~10,000 ms ⏱️        │
│  ├─ Iterative refinement (1 step):   ~10,000 ms          │
│  └─ THIS IS THE BOTTLENECK            (97% of total)      │
│                                                             │
│  FAISS Search (vecRAG):               ~5-20 ms            │
│  ├─ IVF quantization:                 ~2 ms               │
│  ├─ Search nprobe clusters:           ~10 ms              │
│  └─ Rerank top-K:                     ~5 ms               │
│                                                             │
│  PostgreSQL Lookup:                   ~1-5 ms             │
│  └─ Indexed UUID lookup:              ~1 ms               │
│                                                             │
└────────────────────────────────────────────────────────────┘

FULL PIPELINE LATENCY:
  Encoding (5 chunks):     ~300 ms   (3%)
  LVM Inference:           ~0.5 ms   (0.05%)
  Vec2Text Decoding:       ~10,000 ms (97%)
  ─────────────────────────────────────
  TOTAL:                   ~10,300 ms

OPTIMIZATION PRIORITY:
  1. 🎯 Vec2text decoding (10s → 2s)
  2. Encoding batching (300ms → 100ms)
  3. LVM (already excellent at 0.5ms)
```

---

## 🔄 Critical Data Correlations

All data must correlate via **CPE ID (UUID)**:

```
PostgreSQL cpe_entry
    ├─ cpe_id: "123e4567-e89b-12d3-a456-426614174000"
    ├─ concept_text: "Machine learning uses algorithms..."
    └─ metadata: {...}
          │
          │ (Same UUID)
          ↓
PostgreSQL cpe_vectors
    ├─ cpe_id: "123e4567-e89b-12d3-a456-426614174000"
    └─ concept_vec: [0.123, 0.456, ...] (768D)
          │
          │ (Same UUID)
          ↓
FAISS Index
    ├─ index position: 42
    └─ vector: [0.123, 0.456, ...] (768D)
          │
          │ (via NPZ metadata)
          ↓
NPZ File (wikipedia_500k_corrected_vectors.npz)
    ├─ vectors[42]: [0.123, 0.456, ...] (768D)
    ├─ texts[42]: "Machine learning uses algorithms..."
    └─ ids[42]: "123e4567-e89b-12d3-a456-426614174000"

WITHOUT CORRELATION: System breaks!
  ❌ vecRAG: Can't map FAISS index → concept text
  ❌ LVM Training: Can't create text-aligned sequences
  ❌ LVM Inference: Can't decode vector → text
```

---

## 🎓 Best Practices Summary

### ✅ DO:
1. **Always use vec2text-compatible GTR-T5 encoding**
   - Method: `IsolatedVecTextVectOrchestrator.encode_texts()`
   - Never use `sentence-transformers` directly

2. **Maintain CPE ID correlation across all stores**
   - PostgreSQL, FAISS, NPZ must share same UUID

3. **Use sequential data for LVM training**
   - Wikipedia ✅, Textbooks ✅, Papers ✅
   - Ontologies ❌ (taxonomic, not sequential)

4. **Call FAISS `.save()` after every `.add()`**
   - Critical for persistence

5. **Monitor sync status regularly**
   ```bash
   ./scripts/verify_data_sync.sh
   ```

### ❌ DON'T:
1. **Don't use ontology data for LVM training**
   - Reason: Taxonomic hierarchies, not sequential narrative

2. **Don't use InfoNCE loss for LVM**
   - Reason: LVM is regression (MSE), not contrastive learning

3. **Don't skip vec2text encoding method**
   - sentence-transformers is 9.9x worse for vec2text decoding

4. **Don't expect high ROUGE/BLEU scores**
   - LVM does semantic prediction, not token generation
   - Low scores are expected and normal

---

## 📚 Related Documentation

- **Database Locations**: `docs/DATABASE_LOCATIONS.md`
- **LVM Training Data**: `docs/LVM_DATA_MAP.md`
- **Training Rules**: `CLAUDE.md` + `LNSP_LONG_TERM_MEMORY.md`
- **Performance**: `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md`
- **Architecture**: `docs/TOKENLESS_MAMBA_ARCHITECTURE.md`

---

## 🚀 Quick Start Commands

```bash
# Check system status
psql -U lnsp -d lnsp -c "SELECT COUNT(*) FROM cpe_entry;"
brew services list | grep -E 'postgresql|neo4j'

# Verify FAISS index
python3 -c "import faiss; idx = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index'); print(f'Vectors: {idx.ntotal}')"

# Test LVM inference
VEC2TEXT_FORCE_PROJECT_VENV=1 TOKENIZERS_PARALLELISM=false \
    ./venv/bin/python3 tools/lvm_text_output_examples.py

# Train new LVM model
./.venv/bin/python app/lvm/train_unified.py \
    --model-type lstm \
    --data artifacts/lvm/training_sequences_ctx5.npz \
    --epochs 20
```

---

**Last Updated**: October 16, 2025
**System Status**: ✅ All active components operational
**Next Review**: When adding Neo4j graph data
