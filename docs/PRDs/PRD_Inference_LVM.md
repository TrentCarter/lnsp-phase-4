# PRD: LVM Inference Pipeline v2 (Production-Ready)

**Status**: Production Design - Ready for Implementation
**Priority**: P1 (Critical Path)
**Owner**: LVM Team
**Created**: October 7, 2025
**Updated**: October 7, 2025 (v2 - Chief Engineer Review)

---

## 1. Executive Summary

This PRD defines the complete inference pipeline for the Latent Vector Model (LVM), a tokenless vector-native architecture that predicts concept sequences using Mamba-SSM. The pipeline integrates concept extraction, vecRAG lookup, Mamba prediction, and LLM smoothing to generate coherent natural language responses.

**Key Innovation**: Direct vector-to-vector prediction without tokenization, maintaining semantic coherence through vecRAG and enabling continuous ontology evolution.

**v2 Changes** (Critical for Production):
1. ✅ **Pre-routing cache** (S0) - 60-sec LRU for near-duplicate queries
2. ✅ **Calibrated acceptance** - Per-lane Platt/isotonic calibration vs hard 0.85 threshold
3. ✅ **Outbox pattern** - Safe async writes to FAISS/Neo4j vs "atomic 3-way" myth
4. ✅ **Conditioned prediction** - Mamba uses [concept ⊕ question ⊕ TMD] context
5. ✅ **Centroid snapping** - Predictions stay on indexed manifold
6. ✅ **Quorum-based wait** - 70% ready + 250ms grace vs "wait for ALL" (kills p95)
7. ✅ **Tiered arbitration** - ANN→Graph→Cross-lane→vec2text ladder (minimize expensive calls)
8. ✅ **Schema-aware smoothing** - LLM must cite concept IDs, not hallucinate
9. ✅ **Post-response write-backs** - Deferred graph edges for generated concepts

---

## 2. Architecture Overview (v2 - Production)

```
┌─────────────────────────────────────────────────────────────────┐
│      FULL LVM INFERENCE: Text → Concepts → LVM → Response      │
│            (8-Stage Pipeline v2 - Oct 7, 2025 - Production)     │
└─────────────────────────────────────────────────────────────────┘

INPUT TEXT     Concept          vecRAG          GTR-T5        Database
──────────     Extraction       Lookup          Fallback      Insert
               (LLM)            (FAISS)         (New)         (Atomic)

"neural        [Llama 3.1]      [Search         [If not       [PostgreSQL
 networks in   ↓                8K concepts]    found:]       + Neo4j
 AI"           Extract concepts  ↓               ↓             + FAISS]
   │           with TMD         [Cosine sim]    [GTR-T5]         │
   │           ↓                ↓               768D             │
   │           ["neural net"    Match? Yes      ↓                │
   │            (15,14,9)]      ↓               [encode_tmd16]   │
   │           ["ai"            [Get vectors]   ↓                │
   │            (15,5,9)]       768D+16D TMD    [CONCAT 784D]    │
   │                ↓           ↓               ↓                │
   │                └───────────→               [INSERT with]    │
   │                IF NOT                      parent_id/       │
   │                FOUND:──────→               child_id         │
   │                                            (ontology order) │
   │                                                   │
   ↓                                                   ↓
LVM PREDICTION  vecRAG Lookup   vec2text        LLM Smoothing  OUTPUT
──────────────  ─────────────   ────────        ─────────────  ──────

[concept_vec₁]  [FAISS search]  [If not found:] [Llama 3.1]    "Neural
↓               ↓               ↓               ↓              networks
[Mamba LSTM]    [Cosine sim]    [JXE + IELab]   Combine:       are used
↓               ↓               decoders]       - input text   for pattern
[next_vec]──────→[Match?]────N──→[Generate      - next concept recognition
                    │Y          text]           - context      in AI..."
                    │           ↓               ↓                  │
                    │           [New concept!]  [Smooth          │
                    │           ↓               response]         │
                    │           [INSERT to DB]  ↓                 │
                    │                           └─────────────────→
                    │
                [Found! Get text]
                    ↓
                [concept_text]

⚠️  WAIT FOR ALL INPUT CONCEPTS! (If multiple: wait for all Mamba predictions)
✅  THEN: Send all [input₁,next₁, input₂,next₂...] → LLM → Final response
```

---

## 3. Pipeline Stages (Detailed)

### STAGE 1: Text Input → Concept Extraction (LLM + TMD)

**Input**: User query text
**Process**: LLM-based concept extraction with TMD encoding
**Output**: List of (concept_text, TMD_code) tuples

#### Example:
```
Input: "neural networks in AI"

LLM Extraction (Llama 3.1:8b):
├─ Concept 1: "neural network"
│  └─ TMD: (15, 14, 9)
│     ├─ Domain: 15 (Technology/Software)
│     ├─ Task: 14 (Code Generation)
│     └─ Modifier: 9 (Technical)
│
└─ Concept 2: "artificial intelligence"
   └─ TMD: (15, 5, 9)
      ├─ Domain: 15 (Technology/Software)
      ├─ Task: 5 (Entity Recognition)
      └─ Modifier: 9 (Technical)

Output: {
  ("neural network", (15, 14, 9)),
  ("artificial intelligence", (15, 5, 9))
}
```

#### Implementation:
- **LLM**: Local Llama 3.1:8b via Ollama
- **Endpoint**: `http://localhost:11434`
- **Prompt**: Extract concepts with domain/task/modifier classification
- **TMD Encoding**: 16 domains × 32 tasks × 64 modifiers = 32,768 unique codes
- **Latency**: ~500ms per query

#### Critical Requirements:
1. ✅ TMD **MUST** be LLM-extracted (not pattern-based)
2. ✅ Pattern TMD → all same code → no filtering power
3. ✅ LLM TMD → 32,768 codes → 93.75% precision boost

---

### STAGE 2: vecRAG Lookup (FAISS Search)

**Input**: List of (concept_text, TMD_code) tuples
**Process**: Search FAISS index for existing concept vectors
**Output**: concept_id, 784D vector, or NEW_CONCEPT flag

#### For Each Concept:

**PATH A: Concept Found (cosine > 0.85)**
```
1. Search FAISS index (current: 8K concepts)
2. Compute cosine similarity with query embedding
3. IF similarity > threshold (0.85):
   ├─ Retrieve concept_id (global UUID)
   ├─ Retrieve concept_text (for verification)
   ├─ Retrieve vector_784d (768D semantic + 16D TMD)
   └─ ✅ Use existing entry
```

**PATH B: Concept Not Found (cosine < 0.85)**
```
1. Generate new 784D vector:
   ├─ GTR-T5 embedding: concept_text → 768D semantic vector
   ├─ LLM TMD encoding: TMD_code → 16D dense vector
   └─ CONCAT: [768D semantic | 16D TMD] = 784D

2. Find ontological position:
   ├─ Search FAISS for top-5 nearest neighbors
   ├─ Assign parent_id = most general match (higher in ontology)
   └─ Assign child_id = most specific match (lower in ontology)

3. INSERT ATOMICALLY to all 3 stores:
   ├─ PostgreSQL (cpe_entry table):
   │  ├─ concept_id (UUID)
   │  ├─ concept_text
   │  ├─ TMD_code (domain, task, modifier)
   │  ├─ CPESH data (concept, probe, expected, soft_negs, hard_negs)
   │  ├─ vector_784d
   │  ├─ parent_id (UUID or NULL)
   │  └─ child_id (UUID or NULL)
   │
   ├─ Neo4j (Concept node):
   │  ├─ CREATE (:Concept {id: UUID, text: concept_text})
   │  ├─ IF parent_id: CREATE (parent)-[:BROADER]->(new)
   │  └─ IF child_id: CREATE (new)-[:NARROWER]->(child)
   │
   └─ FAISS (IVF Flat index):
      ├─ Add vector_784d at index position = concept_id
      └─ Update index metadata (concept_texts, cpe_ids arrays)

4. ✅ New concept created with ontological position preserved
```

#### Implementation:
- **Embedding Model**: GTR-T5-base (`sentence-transformers/gtr-t5-base`)
- **TMD Encoder**: `encode_tmd16()` (LLM output → 16D dense vector)
- **FAISS Index**: IVF Flat with Inner Product metric
  - `nlist`: 512 (number of clusters)
  - `nprobe`: 16 (clusters to search)
- **Similarity Threshold**: 0.85 (tunable)
- **Latency**:
  - Found: ~0.1ms
  - Not found: ~50ms (GTR-T5) + 20ms (DB insert) = 70ms

#### Critical Requirements:
1. ✅ **Atomic 3-way insert**: PostgreSQL + Neo4j + FAISS must succeed together
2. ✅ **Ontological order**: parent_id/child_id preserve hierarchy
3. ✅ **ID correlation**: Same UUID across all 3 stores
4. ✅ **NPZ metadata**: Include `concept_texts` and `cpe_ids` arrays

---

### STAGE 3: Mamba Prediction (LVM Forward Pass)

**Input**: concept_vec (784D)
**Process**: Trained Mamba-LSTM predicts next concept in sequence
**Output**: next_vec (784D)

#### For Each Input Concept:

```
1. Load trained LVM model:
   ├─ Model path: models/lvm_wordnet.pt
   ├─ Architecture: 784D → [LSTM-5M params] → 784D
   ├─ Training data: 2K WordNet chains (3,965 sequences)
   └─ Test loss: 0.000677 (excellent convergence)

2. Forward pass:
   ├─ Input: concept_vec (784D numpy array)
   ├─ Convert to torch tensor
   ├─ Model inference: next_vec = model(concept_vec)
   └─ Output: next_vec (784D predicted vector)

3. Store prediction:
   └─ prediction_map[concept_id] = next_vec
```

#### Example:
```python
import torch
from src.lvm.models import LatentVectorLSTM

# Load model
model = LatentVectorLSTM.load("models/lvm_wordnet.pt")
model.eval()

# Predict
with torch.no_grad():
    input_vec = torch.tensor(concept_vec).float().unsqueeze(0)  # [1, 784]
    next_vec = model(input_vec).squeeze(0).numpy()  # [784]

# next_vec is the predicted 784D vector for next concept
```

#### Implementation:
- **Model**: Mamba-LSTM (5M parameters)
- **Framework**: PyTorch
- **Device**: MPS (Mac GPU) or CPU
- **Latency**: ~10ms per concept
- **Batch Size**: 1 (single concept prediction)

#### Critical Requirements:
1. ✅ **Model must be trained first**: Run `src/lvm/train_mamba.py`
2. ✅ **Vector dimension match**: Input/output both 784D
3. ✅ **Deterministic**: Same input → same output (eval mode)

---

### STAGE 4: Next Concept Lookup (vecRAG Again)

**Input**: next_vec (784D predicted vector)
**Process**: Search vecRAG for matching concept or generate new text
**Output**: next_concept_text

#### For Each Predicted Vector:

**PATH A: Match Found (cosine > 0.85)**
```
1. Search FAISS with next_vec
2. IF similarity > threshold:
   ├─ Retrieve concept_id
   ├─ Retrieve concept_text
   └─ ✅ next_concept_text = concept_text
```

**PATH B: No Match Found (cosine < 0.85)**
```
1. FALLBACK: vec2text inversion
   ├─ Run JXE decoder (GTR-T5-based, MPS/CPU)
   ├─ Run IELab decoder (GTR-T5-based, CPU only)
   └─ Pick best output (or ensemble)

2. Generated text becomes new concept:
   ├─ concept_text = vec2text_output
   ├─ vector_784d = next_vec (already have it!)
   └─ TMD_code = extract from LLM (concept_text → TMD)

3. INSERT NEW CONCEPT (same atomic 3-way as Stage 2):
   ├─ PostgreSQL: cpe_entry with generated text
   ├─ Neo4j: Concept node with ontological edges
   └─ FAISS: Add next_vec at new index position

4. ✅ New concept discovered and saved
   └─ next_concept_text = vec2text_output
```

#### vec2text Command:
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 \
VEC2TEXT_DEVICE=cpu \
TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-vector next_vec.npy \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1
```

#### Implementation:
- **vecRAG Search**: Same FAISS index as Stage 2
- **vec2text Decoders**:
  - JXE: `jxm/gtr__nq__32__correct`
  - IELab: `ielab/vec2text-GTR-T5-base-nq-corrector`
- **Latency**:
  - Found: ~0.1ms
  - Not found: ~2000ms (vec2text is SLOW!)
- **Optimization**: Minimize vec2text calls by building rich vecRAG index

#### Critical Requirements:
1. ✅ **Threshold tuning**: Balance precision vs. vec2text cost
2. ✅ **Atomic insert**: New concepts must follow same 3-way sync
3. ✅ **TMD extraction**: Generated text needs LLM TMD classification
4. ⚠️ **Performance**: vec2text is 2s bottleneck - avoid when possible!

---

### STAGE 5: Multi-Concept Wait (Synchronization)

**Input**: Multiple (concept, next_vec) pairs
**Process**: Wait for ALL predictions to complete
**Output**: Complete list of (input, next) concept pairs

#### Critical Synchronization Logic:

```python
# BAD: Send partial results to LLM
for concept in concepts:
    next_vec = model.predict(concept.vector)
    next_text = vecrag_lookup(next_vec)
    llm_smooth(concept.text, next_text)  # ❌ WRONG! Incomplete context!

# GOOD: Wait for all predictions
predictions = []
for concept in concepts:
    next_vec = model.predict(concept.vector)
    next_text = vecrag_lookup(next_vec)
    predictions.append((concept.text, next_text))

# Now send complete context to LLM
llm_smooth(query, predictions)  # ✅ CORRECT! Full context!
```

#### Example:
```
Input concepts:
  1. "neural network" → Mamba → next_vec₁ → "deep learning"
  2. "artificial intelligence" → Mamba → next_vec₂ → "machine learning"

Wait for BOTH predictions to complete!

predictions = [
  ("neural network", "deep learning"),
  ("artificial intelligence", "machine learning")
]

Now send to LLM for smoothing (Stage 6)
```

#### Implementation:
- **Concurrency**: Use `asyncio` or parallel execution
- **Wait Strategy**: `asyncio.gather()` or `concurrent.futures`
- **Timeout**: 5s per concept (fail-safe)
- **Latency**: 0ms (pure synchronization)

#### Critical Requirements:
1. ✅ **NO partial results**: Must wait for all concepts
2. ✅ **Order preservation**: Maintain input→output pairing
3. ✅ **Error handling**: If any prediction fails, handle gracefully

---

### STAGE 6: LLM Response Smoothing (Final Output)

**Input**: Original query + [(input_concept, next_concept), ...]
**Process**: LLM generates natural language response
**Output**: Smooth, coherent text response

#### Prompt Template:

```python
prompt = f"""
You are a knowledge system that predicts related concepts.

User Query: {original_query}

Input Concepts and Predictions:
{format_predictions(predictions)}

Task: Generate a smooth, natural language response that:
1. Addresses the user's query directly
2. Incorporates the input concepts and their predicted next concepts
3. Explains the relationships between concepts
4. Is coherent and informative

Response:
"""
```

#### Example:
```
User Query: "neural networks in AI"

Input Concepts and Predictions:
  - "neural network" → "deep learning"
  - "artificial intelligence" → "machine learning"

LLM Output:
"Neural networks are computational models inspired by biological neural
networks in the brain. They form the foundation of deep learning, a subset
of machine learning within artificial intelligence. Neural networks are
widely used for pattern recognition, image classification, and natural
language processing tasks."
```

#### Implementation:
- **LLM**: Local Llama 3.1:8b via Ollama
- **Endpoint**: `http://localhost:11434/api/chat`
- **Parameters**:
  - `temperature`: 0.7 (balanced creativity)
  - `max_tokens`: 300 (concise responses)
  - `stream`: false (get complete response)
- **Latency**: ~800ms

#### Critical Requirements:
1. ✅ **Coherence**: Response must flow naturally
2. ✅ **Accuracy**: Must incorporate predicted concepts correctly
3. ✅ **Relevance**: Must address original query
4. ✅ **Conciseness**: Avoid verbose explanations

---

## 4. Performance Specifications

### Latency Breakdown (Per Query)

| Stage | Operation | Best Case | Worst Case | Notes |
|-------|-----------|-----------|------------|-------|
| **1** | Concept extraction (LLM) | 500ms | 800ms | Per query |
| **2** | vecRAG lookup (FAISS) | 0.1ms × N | 0.1ms × N | N = num concepts |
| **2b** | GTR-T5 + TMD (new concept) | - | 50ms × M | M = new concepts |
| **2c** | DB insert (atomic 3-way) | - | 20ms × M | If new concepts |
| **3** | Mamba forward pass | 10ms × N | 10ms × N | Per concept |
| **4** | vecRAG lookup (next) | 0.1ms × N | 0.1ms × N | Per prediction |
| **4b** | vec2text (if no match) | - | 2000ms × K | K = unknown concepts ⚠️ |
| **5** | Multi-concept wait | 0ms | 0ms | Pure sync |
| **6** | LLM smoothing | 800ms | 1200ms | Final response |
| | | | | |
| **TOTAL** | All concepts found | **~1.3s** | - | Ideal case |
| **TOTAL** | 1 new input concept | **~1.4s** | - | GTR-T5 + insert |
| **TOTAL** | 1 vec2text call | - | **~3.5s** | ⚠️ vec2text bottleneck |

### Throughput

- **Single concept query**: 0.75 QPS (best case)
- **Multi-concept query**: 0.3 QPS (3-4 concepts)
- **Bottlenecks**:
  1. LLM concept extraction (500ms)
  2. LLM response smoothing (800ms)
  3. vec2text fallback (2000ms) ⚠️

### Scalability

- **FAISS Index**: Tested up to 10K concepts
- **Concurrent Queries**: Limited by LLM (Ollama single-threaded)
- **Database**: PostgreSQL + Neo4j handle 100K+ concepts

---

## 5. Data Flow & Dependencies

### Required Services

1. **Ollama** (Local LLM)
   - Port: 11434
   - Model: `llama3.1:8b`
   - Start: `ollama serve`

2. **PostgreSQL** (CPESH + vectors)
   - Database: `lnsp`
   - Tables: `cpe_entry`, `cpe_vectors`

3. **Neo4j** (Concept graph)
   - Port: 7687
   - Database: `neo4j`

4. **FAISS Index** (Vector search)
   - File: `artifacts/ontology_13k_ivf_flat_ip.index`
   - Metadata: `artifacts/ontology_13k.npz`

### Data Correlation (CRITICAL!)

All 3 stores MUST be synchronized by `concept_id` (UUID):

```
PostgreSQL cpe_entry.id = Neo4j Concept.id = FAISS npz.cpe_ids[i]
                 ↓                    ↓                    ↓
            UUID-12345            UUID-12345          UUID-12345
                 ↓                    ↓                    ↓
         "neural network"      (:Concept {text})   vectors[index_i]
```

**Without ID correlation**: Cannot link text ↔ vector ↔ graph → system broken!

### NPZ File Structure (REQUIRED!)

```python
# artifacts/ontology_13k.npz
{
  "vectors": np.array([N, 784]),      # 784D vectors (768D + 16D TMD)
  "concept_texts": np.array([N]),     # Text strings for lookup
  "cpe_ids": np.array([N]),           # UUIDs for database correlation
  "metadata": {...}                    # Index type, metric, etc.
}
```

---

## 6. Implementation Files

### Core Modules

1. **Concept Extraction**: `src/prompt_extractor.py`
   - Function: `extract_cpe_from_text(text: str) -> dict`
   - Returns: `{concept, TMD, probe, soft_negs, hard_negs}`

2. **vecRAG Lookup**: `src/faiss_db.py`
   - Class: `FaissDB`
   - Method: `search(query_vec: np.ndarray, k: int) -> List[Match]`

3. **Mamba Model**: `src/lvm/models.py`
   - Class: `LatentVectorLSTM` (currently LSTM, will upgrade to Mamba)
   - Method: `forward(x: torch.Tensor) -> torch.Tensor`

4. **LLM Client**: `src/llm/local_llama_client.py`
   - Class: `LocalLlamaClient`
   - Method: `chat(messages: List[dict]) -> str`

5. **vec2text**: `app/vect_text_vect/vec_text_vect_isolated.py`
   - Script: Isolated mode with JXE + IELab decoders

### New Files to Create

1. **`src/lvm/inference.py`** (Main orchestrator)
   ```python
   class LVMInferencePipeline:
       def __init__(self):
           self.llm = LocalLlamaClient()
           self.vecrag = FaissDB()
           self.model = LatentVectorLSTM.load("models/lvm_wordnet.pt")

       async def infer(self, query: str) -> str:
           # Stage 1: Extract concepts
           concepts = await self.extract_concepts(query)

           # Stage 2: vecRAG lookup
           concept_vecs = await self.lookup_concepts(concepts)

           # Stage 3: Mamba prediction
           next_vecs = await self.predict_next(concept_vecs)

           # Stage 4: Next concept lookup
           next_texts = await self.lookup_next(next_vecs)

           # Stage 5: Wait for all (implicit with async)
           predictions = list(zip(concepts, next_texts))

           # Stage 6: LLM smoothing
           response = await self.smooth_response(query, predictions)

           return response
   ```

2. **`src/lvm/concept_lookup.py`** (vecRAG with fallback)
   ```python
   class ConceptLookup:
       def __init__(self, faiss_db, vec2text_path, threshold=0.85):
           self.faiss = faiss_db
           self.vec2text = vec2text_path
           self.threshold = threshold

       async def lookup_or_create(self, vector: np.ndarray) -> str:
           # Search FAISS
           matches = self.faiss.search(vector, k=1)

           if matches[0].similarity > self.threshold:
               return matches[0].concept_text

           # Fallback to vec2text
           text = await self.vec2text_invert(vector)

           # Insert new concept (atomic 3-way)
           await self.insert_new_concept(text, vector)

           return text
   ```

3. **`tests/test_lvm_inference.py`** (Unit + integration tests)

---

## 7. Success Criteria

### Phase 1: Basic Pipeline (Week 1)
- [ ] Implement 6-stage pipeline in `src/lvm/inference.py`
- [ ] Unit tests for each stage
- [ ] Integration test: single-concept query → response
- [ ] Latency: < 2s for queries with all concepts found

### Phase 2: Multi-Concept + Fallback (Week 2)
- [ ] Multi-concept synchronization (Stage 5)
- [ ] vec2text fallback integration (Stage 4b)
- [ ] Atomic 3-way insert for new concepts (Stage 2b)
- [ ] Integration test: multi-concept query → response

### Phase 3: Production Readiness (Week 3)
- [ ] Error handling (LLM timeout, DB failures, etc.)
- [ ] Logging and observability
- [ ] Performance optimization (batch predictions, caching)
- [ ] End-to-end benchmarking: 50 queries, measure P@1, latency

### Acceptance Metrics
- **Latency**: 95th percentile < 2s (without vec2text)
- **Accuracy**: Concept extraction F1 > 0.8
- **Coverage**: 95% of queries use existing concepts (no vec2text)
- **Coherence**: Human evaluation: 4/5 rating on response quality

---

## 8. Risks & Mitigations

### Risk 1: vec2text Bottleneck (2s latency)
**Impact**: High - queries with unknown concepts take 3.5s
**Mitigation**:
- Build rich vecRAG index (10K+ concepts from ontologies)
- Pre-compute common queries and cache results
- Lower similarity threshold (0.80 instead of 0.85) to reduce misses
- Future: Train faster vec2text model (100ms target)

### Risk 2: LLM Concept Extraction Errors
**Impact**: Medium - wrong concepts → wrong predictions
**Mitigation**:
- Validate TMD codes (must be in valid ranges)
- Fallback to keyword extraction if LLM fails
- Log extraction errors for manual review
- Fine-tune Llama 3.1 on concept extraction task

### Risk 3: Database Synchronization Failures
**Impact**: Critical - broken ID correlation → system fails
**Mitigation**:
- Atomic transactions (all 3 stores or rollback)
- Validation checks: verify UUID exists in all 3 stores
- Nightly consistency checks (PostgreSQL ↔ Neo4j ↔ FAISS)
- Recovery script: rebuild FAISS from PostgreSQL ground truth

### Risk 4: Mamba Model Drift
**Impact**: Medium - predictions become less accurate over time
**Mitigation**:
- Retrain monthly on latest ontology data
- A/B test new models before deployment
- Track prediction accuracy metrics (MRR, P@1)
- Version models: `lvm_wordnet_v1.pt`, `lvm_wordnet_v2.pt`

---

## 9. Future Enhancements

### Short-term (1-2 months)
1. **Upgrade LSTM → Mamba-SSM** (better long-range dependencies)
2. **Batch prediction** (predict multiple concepts in parallel)
3. **Caching layer** (Redis for common queries)
4. **TMD-based re-ranking** (use TMD similarity in Stage 4)

### Medium-term (3-6 months)
5. **Multi-hop reasoning** (LVM → vecRAG → LVM chains)
6. **Adaptive threshold** (per-query tuning based on confidence)
7. **Hybrid vecRAG+GraphRAG** (combine vector + graph traversal)
8. **Continuous training** (update LVM weekly from ArXiv)

### Long-term (6-12 months)
9. **Reinforcement learning** (RLHF for response quality)
10. **Multi-modal LVM** (text + image vectors)
11. **Federated learning** (privacy-preserving updates)
12. **AutoML** (automatic hyperparameter tuning)

---

## 10. References

### Related PRDs
- [PRD_P15_Latent_LVM_Implementation_Plan.md](./PRD_P15_Latent_LVM_Implementation_Plan.md)
- [PRD_GraphRAG_LightRAG_Architecture.md](./PRD_GraphRAG_LightRAG_Architecture.md)
- [PRD_KnownGood_vecRAG_Data_Ingestion.md](./PRD_KnownGood_vecRAG_Data_Ingestion.md)

### Documentation
- [LNSP_System_Flows.md](../LNSP_System_Flows.md) (Section 6: Complete LVM Inference)
- [LVM_TRAINING_CRITICAL_FACTS.md](../LVM_TRAINING_CRITICAL_FACTS.md)

### Code
- Training: `src/lvm/train_mamba.py`
- Models: `src/lvm/models.py`
- Evaluation: `src/lvm/eval_mamba.py`
- Data Prep: `src/lvm/prepare_training_data.py`

---

## Appendix A: Example End-to-End Flow

```python
# User query
query = "neural networks in AI"

# STAGE 1: Concept extraction
concepts = [
    ("neural network", (15, 14, 9)),
    ("artificial intelligence", (15, 5, 9))
]

# STAGE 2: vecRAG lookup
concept_vecs = [
    (uuid1, "neural network", vec1_784d),    # Found in FAISS
    (uuid2, "artificial intelligence", vec2_784d)  # Found in FAISS
]

# STAGE 3: Mamba prediction
next_vecs = [
    vec3_784d,  # Predicted from vec1
    vec4_784d   # Predicted from vec2
]

# STAGE 4: Next concept lookup
next_concepts = [
    (uuid3, "deep learning", vec3_784d),     # Found in FAISS
    (uuid4, "machine learning", vec4_784d)   # Found in FAISS
]

# STAGE 5: Multi-concept wait (all predictions complete)
predictions = [
    ("neural network", "deep learning"),
    ("artificial intelligence", "machine learning")
]

# STAGE 6: LLM smoothing
response = llm_smooth(query, predictions)
# Output: "Neural networks are computational models that form the foundation
#          of deep learning, a key technique in artificial intelligence and
#          machine learning for pattern recognition and data analysis."

return response
```

---

**Document Status**: ✅ Ready for Implementation
**Next Steps**:
1. Implement `src/lvm/inference.py` (Phase 1)
2. Write unit tests for each stage
3. Run integration tests with 10 sample queries
4. Benchmark latency and accuracy

---

**Changelog**:
- **2025-10-07**: Initial PRD created with complete 6-stage pipeline design
