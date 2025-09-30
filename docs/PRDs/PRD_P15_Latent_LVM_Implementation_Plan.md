# P15 Latent-Only LVM Implementation Plan
**Date**: 2025-09-30
**Author**: Trent Carter
**Status**: Planning Phase

---

## Executive Summary

This document outlines the complete implementation plan for **P15: Latent-Only Large Vector Model (LVM)** using Mamba or alternative state-space models. The system will operate entirely in 768D+16D vector space, replacing traditional token-based processing with vecRAG-based dynamic "vector tokens."

### Key Innovation
**No tokens in or out** - The LVM processes and generates vectors natively, using vecRAG as both input (T→V) and output (V→T) interface, with Vec2Text as fallback for novel concepts.

---

## Part A: Updated P1-P12 Pipeline Status

### Current Completion Status (as of 2025-09-30)

| Stage | Status | Scale | Quality | Notes |
|-------|--------|-------|---------|-------|
| **P1: Corpus Ingestion** | ✅ Complete | 4,993 items | 100% | FactoidWiki dataset |
| **P2: Smart Chunking** | ✅ Complete | 4,993 chunks | Good | Semantic chunking working |
| **P3: Content Classification** | ✅ Complete | 4,993 labeled | Good | Type classification functional |
| **P4: Mission Generation** | ✅ Complete | 4,993 missions | Good | Template-based generation |
| **P5: LLM Interrogation** | ✅ Complete | 4,993 CPE | 94.9% | Ollama+Llama3.1:8b |
| **P6: TMD Encoding** | ✅ Complete | 4,993 TMD | 100% | 16D metadata vectors |
| **P7: Concept Embedding** | ✅ Complete | 4,993 vecs | 100% | GTR-T5 768D embeddings |
| **P8: Vector Fusion** | ✅ Complete | 4,993 fused | 100% | 784D vectors ready |
| **P9: Graph Extraction** | ✅ Complete | 10,070 edges | Good | Neo4j relationships |
| **P10: Entity Resolution** | ✅ Complete | 7,446 entities | Good | Cross-document linking |
| **P11: Vector DB Storage** | ✅ Complete | 4,993 indexed | Excellent | Faiss + PostgreSQL |
| **P12: Graph DB Storage** | ✅ Complete | 12,439 nodes | Good | Neo4j operational |
| **P13: Echo Validation** | ⚠️ Partial | ~500 sampled | Pending | Needs systematic run |
| **P14: Batch Optimization** | 🔄 Integrated | N/A | N/A | Built into ingestion |
| **P15: LNSP Training** | ❌ Not Started | 0 | - | **THIS DOCUMENT** |
| **P16: Multi-RAG Query** | ✅ API Ready | - | Good | Retrieval tested |
| **P17: MoE Inference** | ❌ Not Started | 0 | - | Depends on P15 |

### Key Gaps Identified

1. **P5 CPESH Quality**:
   - ✅ 94.9% (4,736/4,993) have soft+hard negatives
   - ❌ 257 entries missing negatives (5.1% gap)
   - **Action**: Re-run P5 on failed entries OR accept as training noise

2. **P13 Echo Validation**:
   - ✅ Spot checks show 0.82+ cosine similarity
   - ❌ No systematic validation of all 4,993 entries
   - **Action**: Execute full P13 sweep before P15 training

3. **GWOM Data Generation**:
   - ❌ No ordered concept sequences yet
   - ❌ GraphRAG walks not implemented
   - ❌ WikiSearch anchoring not implemented
   - **Action**: Implement GWOM pipeline (see Part C)

---

## Part B: Execute P13 (Echo Validation)

### P13 Objectives
Validate that retrieved concepts match their probe questions with high fidelity (cosine ≥ 0.82).

### Implementation Plan

#### Step 1: Create P13 Validation Script
```bash
# Location: src/pipeline/p13_echo_validation.py
```

**Core Logic**:
1. For each CPE entry:
   - Encode probe_question → query_vec (768D)
   - Encode concept_text → concept_vec (768D)
   - Compute cosine similarity
   - Update echo_score in database
   - Flag validation_status: passed/failed (threshold 0.82)

2. Statistics:
   - Mean/median/p95 cosine scores
   - Pass rate by domain/task/modifier
   - Identify low-quality lanes for re-interrogation

#### Step 2: Run Full Echo Validation
```bash
# Execute P13 on all 4,993 entries
LNSP_TEST_MODE=0 ./.venv/bin/python -m src.pipeline.p13_echo_validation \
  --batch-size 100 \
  --threshold 0.82 \
  --update-db \
  --report-out artifacts/p13_echo_report.json
```

Expected outputs:
- `artifacts/p13_echo_report.json` - validation statistics
- `artifacts/p13_failed_entries.jsonl` - entries below threshold
- Database updated: `echo_score`, `validation_status` columns

#### Step 3: Quality Gates
- ✅ Pass: ≥90% entries with echo_score ≥ 0.82
- ⚠️ Review: 80-90% pass rate (selective re-interrogation)
- ❌ Fail: <80% pass rate (re-run P5 with better prompts)

#### Step 4: Integration with Training
- Only use `validation_status='passed'` entries for P15 initial training
- Use `validation_status='failed'` entries as hard negatives or curriculum tail

---

## Part C: P15 Latent-Only LVM Architecture

### System Overview

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

### Core Components

#### 1. Input Layer: vecRAG Text→Vector
- **Input**: User text query (arbitrary length)
- **Process**:
  1. Encode query with GTR-T5 → 768D vector
  2. TMD classification → 16D metadata vector
  3. Fuse to 784D (optional, or keep separate)
- **Output**: 768D concept vector (core) + 16D TMD (routing)

#### 2. LVM Core: Vector-Native Processor
- **Architecture Options**:
  - **Option A**: Mamba-2 (hybrid SSM+attention)
  - **Option B**: Pure Mamba (SSM only)
  - **Option C**: Vector-MoE (mixture of vector experts)

- **Model Size**: 10M-100M parameters (start small)
- **Input**: Sequence of 768D vectors (no tokens!)
- **Output**: 768D vector (predicted next concept)

#### 3. Output Layer: vecRAG Vector→Text
- **Primary Path**: Faiss similarity search
  1. Query Faiss with output 768D vector
  2. Retrieve top-K concepts (K=1-5)
  3. If top result cosine ≥ 0.85 → use that concept_text

- **Fallback Path**: Vec2Text decoder
  1. If no good Faiss match (cosine < 0.85)
  2. Run JXE + IELab vec2text decoders
  3. Use ensemble or highest-confidence output

- **Output**: Concept text or reconstructed text

#### 4. LLM Smoother: Text Refinement
- **Input**:
  - Original user query (text)
  - LVM output concept (text)
  - Optional: retrieval context

- **Process**:
  - Llama 3.1:8b generates fluent response
  - Combines user intent + LVM reasoning

- **Output**: Natural language response

---

## Part D: GWOM Training Data Generation

### Overview
GWOM generates **ordered concept sequences** for training the LVM to predict "next concept" given a chain.

### Three Parallel Methods

#### Method 1: GraphRAG Walks (42% of training data)
```python
# Weighted random walk through Neo4j graph
START: Random concept node
FOR step in range(5, 15):
    edges = get_outgoing_edges(current_node, confidence ≥ 0.6)
    next_node = weighted_sample(edges)
    chain.append(next_node)
END
```

**Implementation**:
- File: `src/gwom/graphrag_walks.py`
- Cypher queries to Neo4j
- Target: 100K chains from 4,993 concepts
- Quality gate: Mean cosine between neighbors ≥ 0.70

#### Method 2: WikiSearch Anchoring (38% of training data)
```python
# Map concept to Wikipedia, extract ordered links
concept = "Photosynthesis"
wiki_page = fetch_wikipedia(concept)
links = extract_subsection_links(wiki_page)
candidates = filter_by_cosine(links, concept_vec, threshold=0.82)
chain = order_by_page_position(candidates)
```

**Implementation**:
- File: `src/gwom/wikisearch_anchor.py`
- Wikipedia API (local dump preferred)
- Validate against CPESH vectors
- Target: 100K chains

#### Method 3: Ontology Traversal (20% of training data)
```python
# Use curated ontologies (e.g., ConceptNet, DBpedia)
ontology_node = "Cell Division"
relations = traverse_edges(ontology_node, depth=3, types=['is_a', 'part_of', 'causes'])
chain = breadth_first_order(relations)
```

**Implementation**:
- File: `src/gwom/ontology_traverse.py`
- ConceptNet API or local Neo4j ontology import
- Target: 50K chains

### GWOM Data Format

```jsonl
{
  "seq_id": "uuid-1234",
  "method": "graphrag",
  "concept_chain": [
    "Light-dependent reactions split water",
    "Photolysis of water produces oxygen",
    "Oxygen diffuses from chloroplasts",
    "Stomata release oxygen to atmosphere"
  ],
  "concept_ids": ["cpe-001", "cpe-042", "cpe-091", "cpe-112"],
  "vectors": [[0.1, 0.2, ...], [0.15, 0.22, ...], ...],  // 768D each
  "tmd_vectors": [[0,0,1,0,...], [0,0,1,0,...], ...],    // 16D each
  "quality_score": 0.84,
  "coherence_scores": [0.89, 0.85, 0.78],  // pairwise cosine
  "created_at": "2025-09-30T12:00:00Z"
}
```

### Storage: GWOM Data Lake

```
artifacts/gwom/
  gwom_active.jsonl          # Hot append-only log
  gwom_segments/
    seg_20250930.parquet     # Daily rotation
    seg_20251001.parquet
  gwom_index.db              # DuckDB: seq_id → segment mapping
  gwom_manifest.jsonl        # Lineage and stats
```

### Quality Metrics

| Metric | Target | Gate |
|--------|--------|------|
| Mean chain length | 7-12 | 5-20 acceptable |
| Mean coherence (cosine) | ≥0.78 | ≥0.70 minimum |
| Pass rate | ≥80% | ≥70% acceptable |
| Coverage | All 32K lanes | ≥90% lanes |

---

## Part E: LVM Training Procedure

### Training Data Preparation

#### Step 1: Generate GWOM Chains
```bash
# Generate 250K training sequences (100K GraphRAG + 100K Wiki + 50K Ontology)
./.venv/bin/python -m src.gwom.generate_all \
  --graphrag-chains 100000 \
  --wiki-chains 100000 \
  --ontology-chains 50000 \
  --min-length 5 \
  --max-length 15 \
  --coherence-threshold 0.70 \
  --out artifacts/gwom/gwom_active.jsonl
```

#### Step 2: Vectorize All Chains
```bash
# Pre-compute all vectors for training
./.venv/bin/python -m src.gwom.vectorize_chains \
  --input artifacts/gwom/gwom_active.jsonl \
  --embedder gtr-t5-base \
  --out artifacts/gwom/gwom_vectors.npz
```

Expected output:
- `gwom_vectors.npz`: shape [250K, 15, 768] (max length padded)
- `gwom_masks.npz`: shape [250K, 15] (valid positions)

### LVM Architecture Details

#### Option A: Mamba-2 Hybrid (Recommended)

```python
class LatentMamba(nn.Module):
    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        # Input projection (optional, can skip if already 768D)
        self.input_proj = nn.Linear(768, d_model)

        # Mamba-2 SSM layers
        self.mamba_layers = nn.ModuleList([
            Mamba2Layer(d_model, d_state, d_conv, expand)
            for _ in range(6)  # 6 layers for small model
        ])

        # TMD conditioning (inject 16D metadata)
        self.tmd_adapter = nn.Linear(16, d_model)

        # Output head
        self.output_head = nn.Linear(d_model, 768)

    def forward(self, x, tmd=None):
        # x: [batch, seq_len, 768]
        # tmd: [batch, 16]

        h = self.input_proj(x)

        if tmd is not None:
            tmd_emb = self.tmd_adapter(tmd).unsqueeze(1)  # [batch, 1, d_model]
            h = h + tmd_emb  # Broadcast conditioning

        for layer in self.mamba_layers:
            h = layer(h)

        # Predict next vector
        output = self.output_head(h[:, -1, :])  # Take last position
        return output
```

**Model Size**: ~15M parameters
**Memory**: ~500MB
**Training Time**: ~8 hours on single GPU (250K sequences)

#### Option B: Vector-MoE

```python
class VectorMoE(nn.Module):
    def __init__(self, d_model=768, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 2048),
                nn.GELU(),
                nn.Linear(2048, 768)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(768, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [batch, seq_len, 768]
        router_logits = self.router(x[:, -1, :])  # [batch, num_experts]
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        output = torch.zeros_like(x[:, -1, :])
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)
            for batch_idx, exp_idx in enumerate(expert_idx):
                output[batch_idx] += expert_weight[batch_idx] * self.experts[exp_idx](x[batch_idx, -1, :])

        return output
```

### Training Loop

```python
# File: src/lvm/train_latent_mamba.py

import torch
from torch.utils.data import DataLoader
from src.lvm.models import LatentMamba

# Load data
train_dataset = GWOMDataset("artifacts/gwom/gwom_vectors.npz", mode='train')
val_dataset = GWOMDataset("artifacts/gwom/gwom_vectors.npz", mode='val')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model
model = LatentMamba(d_model=768, d_state=16, d_conv=4, expand=2)
model = model.to('cuda')

# Loss: cosine similarity maximization
def loss_fn(pred, target):
    # pred, target: [batch, 768]
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return (1 - cos_sim).mean()  # Minimize distance

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop
for epoch in range(10):
    model.train()
    for batch in train_loader:
        # batch: {
        #   'vectors': [batch, seq_len, 768],
        #   'tmd': [batch, 16],
        #   'target': [batch, 768]  # next vector in sequence
        # }

        optimizer.zero_grad()
        pred = model(batch['vectors'], tmd=batch['tmd'])
        loss = loss_fn(pred, batch['target'])
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch['vectors'], tmd=batch['tmd'])
            loss = loss_fn(pred, batch['target'])
            val_losses.append(loss.item())

    print(f"Epoch {epoch}: val_loss={np.mean(val_losses):.4f}")
    scheduler.step()

# Save model
torch.save(model.state_dict(), "artifacts/lvm/latent_mamba_v1.pt")
```

### Training Curriculum

#### Stage 1: Clean Data Only (Epochs 1-3)
- Use only GWOM chains with coherence ≥ 0.85
- Ontology chains prioritized
- Objective: Learn basic sequential patterns

#### Stage 2: Mixed Quality (Epochs 4-7)
- Introduce coherence ≥ 0.75 chains
- Mix GraphRAG + Wiki chains
- Objective: Generalization to noisy data

#### Stage 3: Full Dataset (Epochs 8-10)
- All GWOM chains (coherence ≥ 0.70)
- Include P13 failed entries as hard negatives
- Objective: Robustness to distribution

### Evaluation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Next-vector cosine | ≥0.80 | Pred vs ground truth |
| Retrieval precision@1 | ≥0.70 | After Faiss lookup |
| Echo loop similarity | ≥0.82 | Full inference chain |
| Vec2Text fallback rate | ≤20% | How often Faiss fails |

---

## Part F: Complete Inference Pipeline

### End-to-End Flow

```python
# File: src/lvm/inference.py

class LatentLVMInference:
    def __init__(self, model_path, faiss_index_path, postgres_conn, vec2text_models):
        self.model = LatentMamba.load(model_path)
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.postgres = postgres_conn
        self.vec2text_jxe = vec2text_models['jxe']
        self.vec2text_ielab = vec2text_models['ielab']
        self.llm_client = OllamaClient("llama3.1:8b")

    def infer(self, query_text: str) -> str:
        """Full inference pipeline: text → vector → LVM → vector → text."""

        # Step 1: Text → Vector (vecRAG input)
        query_vec = self.encode_query(query_text)  # GTR-T5 → 768D
        tmd_vec = self.classify_tmd(query_text)     # TMD classifier → 16D

        # Step 2: Retrieve context (optional, for conditioning)
        context_vecs = self.faiss_index.search(query_vec, k=5)

        # Step 3: LVM forward pass
        input_seq = torch.cat([context_vecs, query_vec.unsqueeze(0)], dim=0)  # [6, 768]
        output_vec = self.model(input_seq.unsqueeze(0), tmd=tmd_vec)  # [1, 768]

        # Step 4: Vector → Text (vecRAG output)
        concept_text = self.vector_to_text(output_vec[0].cpu().numpy())

        # Step 5: LLM Smoothing
        final_response = self.smooth_with_llm(query_text, concept_text)

        return final_response

    def vector_to_text(self, vector: np.ndarray) -> str:
        """V→T with Faiss primary, Vec2Text fallback."""

        # Primary: Faiss lookup
        distances, indices = self.faiss_index.search(vector.reshape(1, -1), k=1)
        cosine_sim = 1 - distances[0][0]  # Assuming IP distance

        if cosine_sim >= 0.85:
            # High confidence - use retrieved concept
            cpe_id = self.postgres.get_cpe_id_by_index(indices[0][0])
            concept_text = self.postgres.get_concept_text(cpe_id)
            return concept_text
        else:
            # Low confidence - use Vec2Text
            jxe_output = self.vec2text_jxe.decode(vector)
            ielab_output = self.vec2text_ielab.decode(vector)

            # Ensemble: pick higher cosine to original vector
            jxe_sim = cosine_similarity(vector, self.encode_query(jxe_output))
            ielab_sim = cosine_similarity(vector, self.encode_query(ielab_output))

            return jxe_output if jxe_sim > ielab_sim else ielab_output

    def smooth_with_llm(self, original_query: str, concept_response: str) -> str:
        """LLM combines query + concept into fluent response."""

        prompt = f"""You are a helpful assistant. The user asked: "{original_query}"

Based on retrieved knowledge, the core concept is: "{concept_response}"

Generate a clear, concise response that directly answers the user's question using this knowledge."""

        response = self.llm_client.generate(prompt, max_tokens=200)
        return response
```

### Example Inference Trace

```
User Query: "How does photosynthesis produce oxygen?"

[Step 1: Text→Vector]
  GTR-T5 encode → vector_query: [0.12, -0.45, 0.89, ...]  (768D)
  TMD classify → Science/Explanation/Biochemical → [0,0,1,0,...]  (16D)

[Step 2: Retrieve Context]
  Faiss top-5 nearest concepts:
    1. "Light-dependent reactions split water" (cosine 0.91)
    2. "Photolysis releases oxygen" (cosine 0.88)
    3. "Chloroplasts contain thylakoids" (cosine 0.82)
    ...

[Step 3: LVM Forward]
  Input sequence: [context_vecs (5×768), query_vec (1×768)]
  Mamba processes sequence → output_vec: [0.15, -0.40, 0.87, ...]  (768D)

[Step 4: Vector→Text]
  Faiss search on output_vec → top match: "Photolysis of water produces oxygen gas"
  Cosine similarity: 0.93 → HIGH CONFIDENCE, use this text
  (Vec2Text not needed)

[Step 5: LLM Smoothing]
  Prompt to Llama3.1:
    User asked: "How does photosynthesis produce oxygen?"
    Core concept: "Photolysis of water produces oxygen gas"

  LLM response: "During photosynthesis, light-dependent reactions split water molecules
  through a process called photolysis, which releases oxygen gas as a byproduct."

[Final Output] → User sees smoothed response
```

---

## Part G: Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Execute P13 Echo Validation on all 4,993 entries
- [ ] Fix 257 missing CPESH negatives (re-run P5 or accept as noise)
- [ ] Create GWOM generation scripts (GraphRAG, Wiki, Ontology stubs)
- [ ] Set up GWOM data lake structure

### Phase 2: GWOM Data Generation (Week 2-3)
- [ ] Implement GraphRAG walks (target 100K chains)
- [ ] Implement WikiSearch anchoring (target 100K chains)
- [ ] Implement Ontology traversal (target 50K chains)
- [ ] Validate GWOM quality metrics (coherence, coverage)

### Phase 3: LVM Training (Week 4-5)
- [ ] Implement Mamba-2 architecture
- [ ] Create training dataset from GWOM chains
- [ ] Run training loop (10 epochs, curriculum stages)
- [ ] Evaluate on held-out test set

### Phase 4: Inference Integration (Week 6)
- [ ] Build inference pipeline (T→V→LVM→V→T)
- [ ] Integrate Vec2Text fallback
- [ ] Add LLM smoothing layer
- [ ] End-to-end testing

### Phase 5: Evaluation & Iteration (Week 7-8)
- [ ] Benchmark against baselines (pure retrieval, GPT-4)
- [ ] A/B testing with human evaluators
- [ ] Performance optimization (quantization, caching)
- [ ] Production deployment

---

## Part H: Success Criteria

### Training Success
- ✅ 250K GWOM chains generated with ≥0.75 mean coherence
- ✅ LVM achieves ≥0.80 next-vector cosine on validation set
- ✅ Training completes in <12 hours on single GPU
- ✅ Model size ≤100MB (deployable)

### Inference Success
- ✅ End-to-end latency <2 seconds (query → response)
- ✅ Faiss retrieval precision@1 ≥70%
- ✅ Vec2Text fallback rate ≤20%
- ✅ Final LLM response rated "good" or "excellent" by humans ≥80%

### System Integration
- ✅ P13 validation pass rate ≥90%
- ✅ GWOM data lake operational with rotation
- ✅ Inference API deployed and stress-tested
- ✅ Monitoring dashboards for LVM quality metrics

---

## Part I: Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| GWOM chains too noisy | High | Multi-stage curriculum; filter by coherence |
| LVM doesn't learn patterns | Critical | Start with small model; validate on toy task first |
| Vec2Text quality poor | Medium | Ensemble JXE+IELab; tune decoding steps |
| Faiss doesn't scale | Medium | Use IVF-PQ compression; shard by lane |
| LLM smoothing adds latency | Low | Cache frequent queries; use streaming |
| Training data biased | Medium | Balance GWOM sources; oversample rare lanes |

---

## Part J: Next Steps

### Immediate Actions (Next 48 hours)
1. ✅ Review and approve this plan
2. Execute P13 full validation run
3. Start GWOM GraphRAG walk implementation
4. Set up training infrastructure (GPU instance, data loaders)

### This Week
1. Complete GWOM data generation (at least 10K chains for MVP)
2. Implement Mamba-2 architecture skeleton
3. Run toy training task (100 chains, 3 epochs) to validate pipeline

### This Month
1. Full GWOM generation (250K chains)
2. Complete LVM training (10 epochs)
3. Deploy inference API
4. Begin evaluation and iteration

---

## Appendix A: File Structure

```
lnsp-phase-4/
├── src/
│   ├── gwom/
│   │   ├── __init__.py
│   │   ├── graphrag_walks.py       # Method 1: Graph walks
│   │   ├── wikisearch_anchor.py    # Method 2: Wikipedia
│   │   ├── ontology_traverse.py    # Method 3: Ontologies
│   │   ├── generate_all.py         # Orchestrator
│   │   └── vectorize_chains.py     # Pre-compute vectors
│   ├── lvm/
│   │   ├── __init__.py
│   │   ├── models.py               # Mamba, MoE architectures
│   │   ├── train_latent_mamba.py   # Training loop
│   │   ├── inference.py            # Inference pipeline
│   │   └── dataset.py              # GWOM dataset loader
│   └── pipeline/
│       └── p13_echo_validation.py  # P13 implementation
├── artifacts/
│   ├── gwom/
│   │   ├── gwom_active.jsonl
│   │   ├── gwom_segments/
│   │   ├── gwom_vectors.npz
│   │   └── gwom_index.db
│   └── lvm/
│       ├── latent_mamba_v1.pt
│       └── checkpoints/
└── docs/PRDs/
    └── PRD_P15_Latent_LVM_Implementation_Plan.md  # This file
```

---

## Appendix B: References

1. **Mamba**: [Gu & Dao, 2023 - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"](https://arxiv.org/abs/2312.00752)
2. **Vec2Text**: [Morris et al., 2023 - "Text Embeddings Reveal (Almost) As Much As Text"](https://arxiv.org/abs/2310.06816)
3. **GraphRAG**: Microsoft Research, 2024
4. **LNSP Pipeline**: `docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md`
5. **GWOM PRD**: `docs/PRDs/PRD_GWOM_design_Options.md`

---

**Document Status**: Draft v1.0 - Awaiting Review
**Next Update**: After P13 completion and GWOM MVP
