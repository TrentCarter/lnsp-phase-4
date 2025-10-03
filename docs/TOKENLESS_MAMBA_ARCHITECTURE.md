# Tokenless Mamba Architecture: Deep Dive

**Date**: October 2, 2025
**Status**: Architecture Review & Validation
**Source**: PRD_P15_Latent_LVM_Implementation_Plan.md, architecture.md, QUICKSTART_P13_P15.md

---

## Executive Summary

The **Tokenless Mamba LVM (Latent-only Large Vector Model)** is a novel architecture that eliminates token-based processing entirely, operating purely in 768D+16D vector space. This is a fundamental departure from traditional LLMs.

### Key Innovation
**No tokens in or out** - The LVM processes and generates vectors natively, using vecRAG as both input (T→V) and output (V→T) interface, with Vec2Text as fallback for novel concepts.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    P15 LATENT-ONLY LVM SYSTEM                 │
└─────────────────────────────────────────────────────────────┘

INFERENCE PATH (Main):
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ User Query  │──►│  vecRAG T→V │──►│ LVM (Mamba) │──►│ vecRAG V→T  │
│   (text)    │   │  (GTR-T5)   │   │  768D→768D  │   │  (Faiss +   │
└─────────────┘   └─────────────┘   └─────────────┘   │   Vec2Text) │
                                                        └──────┬──────┘
                                                               │
                  ┌────────────────────────────────────────────┘
                  ▼
           ┌─────────────┐   ┌─────────────┐
           │ LLM Smoother│──►│ Final Output│
           │ (Llama3.1)  │   │   (text)    │
           └─────────────┘   └─────────────┘

TRAINING PATH:
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ GWOM Chains │──►│   Vectorize │──►│ LVM Training│
│ (sequences) │   │   (GTR-T5)  │   │ (next-vec   │
│             │   │   + TMD     │   │  prediction)│
└─────────────┘   └─────────────┘   └─────────────┘
```

---

## Core Components

### 1. Input Layer: vecRAG Text→Vector

**Purpose**: Convert text queries into dense vector representations

**Architecture**:
```
Text Query
    ↓
GTR-T5 Encoder (768D semantic embedding)
    ↓
TMD Classifier (16D metadata: domain/task/modifier)
    ↓
Fused 784D Vector OR Separate 768D+16D
```

**Implementation**:
- **Encoder**: `sentence-transformers/gtr-t5-base` (frozen weights)
- **TMD**: Deterministic bit-encoding from LLM-extracted metadata
- **Fusion**: Concatenate + L2 normalize to unit vector
- **Output**: 768D concept vector (core) + 16D TMD (routing metadata)

**Key Design Decision**:
- **Option A**: Feed 768D only to Mamba (pure semantic)
- **Option B**: Feed 784D fused (semantic + metadata)
- **Recommendation**: Start with 768D for simplicity, add TMD routing in MoE layer

---

### 2. LVM Core: Mamba Vector-Native Processor

**Purpose**: Process sequences of vectors without tokenization

#### Architecture Options

##### Option A: Mamba-2 (Hybrid SSM+Attention)
```python
class MambaVectorLM(nn.Module):
    def __init__(self, d_model=768, n_layers=12):
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=16,      # SSM state dimension
                d_conv=4,        # Conv kernel size
                expand=2         # MLP expansion factor
            )
            for _ in range(n_layers)
        ])

    def forward(self, vec_seq):
        # vec_seq: (batch, seq_len, 768)
        # NO EMBEDDING LAYER - vectors ARE the input!

        x = vec_seq
        for layer in self.layers:
            x = layer(x)  # SSM processing

        return x[:, -1, :]  # Return last vector (next prediction)
```

**Key Differences from Token Mamba**:
- ❌ **NO** embedding layer (`nn.Embedding` removed)
- ❌ **NO** vocabulary projection (`lm_head` removed)
- ✅ **Direct vector input**: `(batch, seq_len, d_model)`
- ✅ **Direct vector output**: `(batch, d_model)` prediction

##### Option B: Pure Mamba (SSM Only)
```python
class PureMambaVectorLM(nn.Module):
    def __init__(self, d_model=768, n_layers=8):
        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=16,
                use_attention=False  # Pure SSM
            )
            for _ in range(n_layers)
        ])
```

##### Option C: Vector-MoE (Mixture of Vector Experts)
```python
class VectorMoE(nn.Module):
    def __init__(self, d_model=768, n_experts=8, n_layers=12):
        self.tmd_router = TMDRouter(n_experts=n_experts)

        self.experts = nn.ModuleList([
            MambaBlock(d_model=d_model)
            for _ in range(n_experts)
        ])

    def forward(self, vec_seq, tmd_codes):
        # Route based on TMD metadata
        expert_weights = self.tmd_router(tmd_codes)

        expert_outputs = [
            expert(vec_seq) for expert in self.experts
        ]

        # Weighted combination
        return sum(w * out for w, out in zip(expert_weights, expert_outputs))
```

#### Model Sizing

| Model Size | Parameters | Layers | d_model | d_state | Notes |
|------------|------------|--------|---------|---------|-------|
| **Tiny** | 10M | 6 | 768 | 16 | Proof of concept |
| **Small** | 50M | 12 | 768 | 16 | Initial training |
| **Medium** | 100M | 16 | 768 | 32 | Production target |
| **Large** | 500M | 24 | 768 | 64 | Future scaling |

**Starting Point**: 50M parameters (12 layers, 768D)

---

### 3. Output Layer: vecRAG Vector→Text

**Purpose**: Convert predicted vectors back to text

#### Primary Path: Faiss Similarity Search

```python
def vector_to_text_faiss(pred_vec: np.ndarray, threshold=0.85):
    """
    Convert predicted 768D vector to text via nearest neighbor search
    """
    # Query Faiss index
    distances, indices = faiss_index.search(
        pred_vec.reshape(1, -1),
        k=5  # Top-5 candidates
    )

    # Convert distances to cosine similarity
    # (assumes normalized vectors and inner product metric)
    similarities = distances[0]  # Already cosine for normalized vecs

    # Check if top match exceeds threshold
    if similarities[0] >= threshold:
        # High confidence - use corpus concept
        best_idx = indices[0][0]
        return corpus_concepts[best_idx], "faiss"

    # Low confidence - fall back to vec2text
    return vector_to_text_vec2text(pred_vec), "vec2text"
```

**Threshold Tuning**:
- **0.95+**: Exact match (rare, use for verification)
- **0.85-0.95**: High confidence (primary production range)
- **0.75-0.85**: Medium confidence (may need vec2text)
- **<0.75**: Low confidence (definitely use vec2text)

#### Fallback Path: Vec2Text Decoder

```python
def vector_to_text_vec2text(pred_vec: np.ndarray):
    """
    Reconstruct text from vector using trained decoders
    """
    # Normalize vector
    pred_vec = pred_vec / np.linalg.norm(pred_vec)

    # Run both decoders
    jxe_text = jxe_decoder.decode(pred_vec, steps=5)
    ielab_text = ielab_decoder.decode(pred_vec, steps=5)

    # Ensemble strategy
    if jxe_text == ielab_text:
        return jxe_text  # Agreement

    # Vote or use confidence scores
    jxe_score = compute_likelihood(jxe_text, pred_vec)
    ielab_score = compute_likelihood(ielab_text, pred_vec)

    return jxe_text if jxe_score > ielab_score else ielab_text
```

**Key Design Decisions**:
1. **Faiss-first**: Always try nearest neighbor (fast, exact for known concepts)
2. **Vec2Text fallback**: Only for novel/OOD vectors (slower, generative)
3. **Threshold learning**: Tune 0.85 threshold on validation set
4. **Ensemble voting**: Combine JXE + IELab for robustness

---

### 4. LLM Smoother: Text Refinement (Optional)

**Purpose**: Generate fluent natural language from vector-retrieved concepts

```python
def llm_smoother(user_query: str, lvm_concept: str, context: List[str]):
    """
    Llama 3.1:8b generates fluent response
    """
    prompt = f"""
    User asked: {user_query}

    The vector model retrieved: {lvm_concept}

    Additional context: {', '.join(context)}

    Provide a natural, helpful response that answers the user's question.
    """

    return call_local_llama(prompt)
```

**When to Use**:
- ✅ User expects conversational response
- ✅ LVM output is terse/technical
- ✅ Need to combine multiple retrieved concepts
- ❌ Skip for speed-critical applications
- ❌ Skip when concept text is already fluent

---

## Training Architecture

### Training Objective: Next Vector Prediction

**Core Task**: Given sequence of vectors, predict next vector in sequence

```python
def train_step(vec_sequence, target_vec):
    """
    vec_sequence: (batch, seq_len, 768) - input vectors
    target_vec: (batch, 768) - ground truth next vector
    """
    # Forward pass through Mamba
    pred_vec = mamba_lvm(vec_sequence)

    # Loss: Cosine similarity (maximize)
    # OR: MSE (minimize L2 distance)
    loss = 1 - F.cosine_similarity(pred_vec, target_vec)

    # Backprop
    loss.backward()
    optimizer.step()

    return loss.item()
```

### Training Data: CPESH + GWOM

#### CPESH (Concept-Probe-Expected-Soft-Hard)

**Purpose**: Contrastive learning for concept retrieval

```
Format:
{
  "concept_text": "oxidoreductase activity",
  "concept_vec": [768D vector],
  "probe_question": "What is the function of oxidoreductase?",
  "probe_vec": [768D vector],
  "expected_answer": "Catalyzes oxidation-reduction reactions",
  "soft_negatives": ["transferase activity", "hydrolase activity"],
  "hard_negatives": ["lipid binding", "protein kinase activity"]
}
```

**Training Strategy**:
1. **Positive pairs**: (probe_vec, concept_vec) should be close
2. **Soft negatives**: Related concepts (cosine ~0.6-0.8)
3. **Hard negatives**: Unrelated concepts (cosine <0.5)

**Loss Function**:
```python
def cpesh_contrastive_loss(anchor, positive, soft_negs, hard_negs):
    # Triplet loss with multiple negatives
    pos_sim = F.cosine_similarity(anchor, positive)

    soft_sim = [F.cosine_similarity(anchor, neg) for neg in soft_negs]
    hard_sim = [F.cosine_similarity(anchor, neg) for neg in hard_negs]

    # Encourage: pos_sim > soft_sim > hard_sim
    margin_soft = 0.2
    margin_hard = 0.4

    loss_soft = max(0, margin_soft - (pos_sim - max(soft_sim)))
    loss_hard = max(0, margin_hard - (pos_sim - max(hard_sim)))

    return loss_soft + loss_hard
```

#### GWOM (Graph Walk Ordered Memory)

**Purpose**: Sequential reasoning chains for autoregressive training

```
Format (sequence of concept vectors):
[
  vec_1: "enzyme",
  vec_2: "catalysis",
  vec_3: "oxidoreductase",
  vec_4: "NAD+ binding",
  vec_5: "electron transfer"
]
```

**Generation Strategy**:
1. **Graph walks**: Random walks through Neo4j knowledge graph
2. **WikiSearch anchoring**: Start from Wikipedia article, traverse citations
3. **Ordered sequences**: Ensure semantic coherence (cosine between adjacent >0.6)

**Training Examples**:
```python
# Input: vecs 1-4 → Predict: vec 5
train_examples = [
    (seq[:i], seq[i]) for seq in gwom_chains for i in range(1, len(seq))
]
```

### Training Pipeline

```bash
# Step 1: Prepare CPESH data (already complete)
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry WHERE validation_status='passed'"
# Expected: ~4,500 entries

# Step 2: Generate GWOM chains
python -m src.training.gwom_generator \
  --neo4j-uri bolt://localhost:7687 \
  --n-chains 10000 \
  --min-length 3 \
  --max-length 10 \
  --out artifacts/gwom_chains.jsonl

# Step 3: Vectorize GWOM chains
python -m src.training.vectorize_gwom \
  --chains artifacts/gwom_chains.jsonl \
  --embedder gtr-t5-base \
  --out artifacts/gwom_vectors.npz

# Step 4: Train Mamba LVM
python -m src.training.train_mamba_lvm \
  --cpesh-data artifacts/cpesh_validated.npz \
  --gwom-data artifacts/gwom_vectors.npz \
  --model-size small \
  --batch-size 32 \
  --epochs 10 \
  --checkpoint-dir checkpoints/mamba_lvm_v1
```

---

## Why Tokenless Architecture?

### Problems with Token-Based LLMs

1. **Vocabulary Bottleneck**: Fixed vocab (50K-100K tokens) limits concepts
2. **Tokenization Overhead**: Encode text → tokens → embeddings → vectors
3. **Discrete Space**: Tokens are categorical, not continuous
4. **Alignment Tax**: Must learn token→vector→semantic mapping

### Advantages of Tokenless Mamba

1. **Native Vector Processing**: Skip tokenization entirely
2. **Infinite Vocabulary**: Any concept expressible in 768D space
3. **Continuous Semantics**: Smooth interpolation between concepts
4. **Faster Inference**: No embedding lookup, direct vector ops
5. **vecRAG Integration**: Perfect alignment with retrieval system

### Performance Implications

| Metric | Token-based LLM | Tokenless Mamba LVM |
|--------|-----------------|---------------------|
| **Input latency** | Tokenize + embed (~5-10ms) | Direct vector (0ms) |
| **Vocab size** | 50K-100K discrete | ∞ continuous |
| **OOV handling** | UNK token / subword | Native (any vector) |
| **Retrieval alignment** | Separate embedding model | Direct (same 768D space) |
| **Training data** | Text sequences | Vector sequences (CPESH + GWOM) |

---

## Implementation Roadmap

### Phase 1: Foundation (Current - Week 1)
- [x] vecRAG proven with +10.1% P@1 over BM25
- [x] CPESH data quality validated (94.9% complete)
- [x] TMD encoding working (16D metadata)
- [ ] P13 echo validation (systematic run on 4,993 entries)

### Phase 2: Training Data (Week 2)
- [ ] GWOM chain generator (graph walks)
- [ ] WikiSearch anchoring (article→citation chains)
- [ ] Vectorize GWOM sequences (GTR-T5)
- [ ] Validate sequence quality (cosine coherence >0.6)

### Phase 3: Model Development (Week 3-4)
- [ ] Implement Mamba vector-only architecture
- [ ] Train 50M parameter model (12 layers, 768D)
- [ ] Integrate Faiss V→T retrieval (threshold=0.85)
- [ ] Integrate Vec2Text fallback (JXE + IELab)

### Phase 4: Evaluation (Week 5)
- [ ] Echo test: predict known concepts (P@1 target: >0.80)
- [ ] Novel concept test: OOD vector decoding (Vec2Text quality)
- [ ] Latency benchmark: <50ms inference (P95)
- [ ] Compare vs GPT-3.5 baseline on QA tasks

### Phase 5: Production (Week 6+)
- [ ] API integration with LLM smoother
- [ ] A/B test vs current vecRAG-only system
- [ ] Scale to 100M parameter model
- [ ] Deploy to production

---

## Critical Design Questions

### Q1: 768D only or 784D (768+16 TMD)?

**Option A: Pure 768D Semantic**
- ✅ Simpler model architecture
- ✅ Proven GTR-T5 embeddings
- ❌ Loses TMD routing information

**Option B: 784D Fused**
- ✅ Preserves TMD metadata
- ✅ Enables domain-specific routing
- ❌ Slightly more complex

**Recommendation**: Start with **768D** for simplicity. Add TMD routing in separate MoE layer if needed.

### Q2: Which Mamba variant?

**Option A: Mamba-2 (Hybrid)**
- ✅ State-of-the-art performance
- ✅ Attention + SSM benefits
- ❌ More parameters

**Option B: Pure Mamba (SSM)**
- ✅ Faster inference (no attention)
- ✅ Linear complexity
- ❌ May need more layers

**Recommendation**: Start with **Mamba-2** (12 layers) for best quality, optimize later.

### Q3: V→T threshold tuning?

**Current**: 0.85 cosine for Faiss, else Vec2Text

**Validation Strategy**:
1. Run on validation set with varying thresholds (0.75, 0.80, 0.85, 0.90, 0.95)
2. Measure:
   - Faiss hit rate (% resolved without Vec2Text)
   - Accuracy (% correct concept retrieved)
   - Latency (Faiss vs Vec2Text)
3. Choose threshold that maximizes accuracy while keeping Faiss hit rate >70%

**Expected**: 0.85 is good starting point based on echo validation

---

## Success Metrics

### Training Metrics
- **CPESH contrastive loss**: <0.1 (converged)
- **GWOM next-vector MSE**: <0.05 (768D unit vectors)
- **Echo score**: >0.82 cosine (probe→concept alignment)

### Inference Metrics
- **P@1 (Faiss)**: >0.80 (80% exact match on known concepts)
- **Vec2Text quality**: >0.70 human eval on novel concepts
- **Latency P95**: <50ms (Faiss) + <500ms (Vec2Text fallback)
- **Throughput**: >100 QPS on single GPU

### Comparison Baselines
- **vs BM25**: Expect +15-20% P@1 (already proven at +10.1%)
- **vs GPT-3.5**: Target 80% quality at 10x lower latency
- **vs GraphRAG**: Target comparable quality at 100x lower latency

---

## References

- **PRD Source**: `docs/PRDs/PRD_P15_Latent_LVM_Implementation_Plan.md:10-187`
- **Architecture Context**: `docs/architecture.md:901-936`
- **Quickstart Guide**: `docs/PRDs/QUICKSTART_P13_P15.md:404-421`
- **vecRAG Benchmark**: `RAG/results/VECRAG_PERFORMANCE_REPORT.md`
- **Mamba Paper**: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- **Vec2Text**: Morris et al. (2023) "Text Embeddings Reveal (Almost) As Much As Text"

---

**Status**: Architecture validated ✅
**Next Step**: Execute P13 echo validation, then begin GWOM generation
**Owner**: LNSP Phase 4 Team
**Last Updated**: October 2, 2025
