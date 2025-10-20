# PRD: Extended Context LVM Experiments

**Date:** October 19, 2025
**Status:** READY TO IMPLEMENT
**Goal:** Test 3 architectures with extended context + TMD routing

---

## 🎯 Context Window Calculation (You're Absolutely Right!)

### Current State (TINY!)
```
Current LVM:    5 vectors
Token equivalent: 5 × 20 = 100 tokens
Context window: TINY (like GPT-1 from 2018!)
```

### Competitive Comparison

| Model | Token Context | Vector Context (÷20) | Our Status |
|-------|---------------|---------------------|------------|
| **GPT-4** | 128,000 | 6,400 vectors | ❌ Need to scale |
| **Claude 3** | 200,000 | 10,000 vectors | ❌ Need to scale |
| **GPT-3.5** | 16,384 | 819 vectors | ❌ Need to scale |
| **Llama 3.1** | 128,000 | 6,400 vectors | ❌ Need to scale |
| **Our LVM** | 100 | **5 vectors** | 🔴 CRITICAL GAP |

### Proposed Scaling (Phase 1)

| Phase | Vectors | Token Equiv | Feasibility |
|-------|---------|-------------|-------------|
| **Current** | 5 | 100 | ✅ Implemented |
| **Phase 1a** | 50 | 1,000 | ✅ Easy (10x) |
| **Phase 1b** | 100 | 2,000 | ✅ Moderate |
| **Phase 1c** | 500 | 10,000 | ⚠️ Hierarchical needed |
| **Phase 2** | 2,000 | 40,000 | ⚠️ Sparse attention |
| **Phase 3** | 5,000 | 100,000 | ⚠️ Advanced arch |

**Recommendation:** Start with 100 vectors (2k tokens) for initial experiments

---

## 🏗️ Three Architectures to Test

### Architecture A: **Hierarchical Attention GRU**
**Goal:** Extend context to 100 vectors via two-level processing

```python
class HierarchicalGRU(nn.Module):
    """
    Two-level hierarchical processing:
    - Level 1 (Local): Process 10 chunks of 10 vectors each
    - Level 2 (Global): Attend over chunk summaries

    Total context: 100 vectors (2,000 tokens equivalent)
    """
    def __init__(self, input_dim=768, d_model=512):
        super().__init__()

        # Level 1: Local chunk processing (10 vectors → 512D summary)
        self.local_encoder = GRUStack(
            input_dim=768,
            d_model=512,
            num_layers=2,  # Lighter per chunk
            dropout=0.0
        )

        # Level 2: Global attention over chunks (10 summaries → final)
        self.global_encoder = GRUStack(
            input_dim=512,
            d_model=512,
            num_layers=2,
            dropout=0.0
        )

        # TMD-aware routing (optional)
        self.tmd_gate = nn.Linear(16, 512)  # TMD → attention weights

        # Output projection
        self.output_proj = nn.Linear(512, 768)

    def forward(self, x, tmd=None):
        """
        Args:
            x: [batch, 100, 768] - long context
            tmd: [batch, 16] - optional TMD vector for routing
        Returns:
            [batch, 768] - predicted next vector
        """
        batch_size = x.size(0)
        chunk_size = 10
        num_chunks = 10  # 100 vectors / 10 = 10 chunks

        # Level 1: Process each chunk independently
        chunk_summaries = []
        for i in range(num_chunks):
            chunk = x[:, i*chunk_size:(i+1)*chunk_size, :]  # [batch, 10, 768]
            summary = self.local_encoder(chunk)  # [batch, 512]
            chunk_summaries.append(summary)

        # Stack chunk summaries
        chunk_seq = torch.stack(chunk_summaries, dim=1)  # [batch, 10, 512]

        # Level 2: Global attention over chunks
        global_output = self.global_encoder(chunk_seq)  # [batch, 512]

        # Optional: TMD-aware gating
        if tmd is not None:
            tmd_weights = torch.sigmoid(self.tmd_gate(tmd))
            global_output = global_output * tmd_weights

        # Project to output
        output = self.output_proj(global_output)  # [batch, 768]

        return F.normalize(output, p=2, dim=-1)
```

**Parameters:** ~14M (2 × 7M)
**Context:** 100 vectors (2,000 tokens)
**Complexity:** O(10 × 10²) = O(1,000) vs O(100²) = O(10,000)

---

### Architecture B: **Memory-Augmented GRU (MemGRU)**
**Goal:** Add external memory for long-term concept retention

```python
class MemoryAugmentedGRU(nn.Module):
    """
    GRU with External Memory Bank:
    - 2048 memory slots (768D each)
    - Content-based read/write
    - Differentiable addressing

    Context: Still 5 vectors, but augmented with persistent memory
    """
    def __init__(self, input_dim=768, d_model=512, memory_size=2048):
        super().__init__()

        # Base GRU encoder
        self.gru = GRUStack(
            input_dim=768,
            d_model=512,
            num_layers=4,
            dropout=0.0
        )

        # External memory bank (persistent across batches during inference)
        self.register_buffer(
            'memory_bank',
            torch.randn(memory_size, input_dim) / math.sqrt(input_dim)
        )

        # Memory addressing
        self.read_query = nn.Linear(512, 768)
        self.write_query = nn.Linear(512, 768)
        self.write_value = nn.Linear(512, 768)

        # Gating
        self.memory_gate = nn.Linear(512 + 768, 1)

        # TMD-aware memory indexing
        self.tmd_memory_router = nn.Linear(16, memory_size)

        # Output
        self.output_proj = nn.Linear(768, 768)

    def forward(self, x, tmd=None, update_memory=True):
        """
        Args:
            x: [batch, 5, 768] - short context
            tmd: [batch, 16] - optional TMD for memory routing
            update_memory: Whether to write to memory (False during eval)
        Returns:
            [batch, 768] - predicted next vector
        """
        batch_size = x.size(0)

        # Encode context with GRU
        hidden = self.gru(x)  # [batch, 512]

        # === MEMORY READ ===

        # Generate read query
        read_query = self.read_query(hidden)  # [batch, 768]

        # Optional: TMD-based memory routing
        if tmd is not None:
            tmd_scores = self.tmd_memory_router(tmd)  # [batch, memory_size]
            tmd_mask = torch.sigmoid(tmd_scores)  # Soft mask
        else:
            tmd_mask = 1.0

        # Content-based addressing
        memory_similarities = F.cosine_similarity(
            read_query.unsqueeze(1),  # [batch, 1, 768]
            self.memory_bank.unsqueeze(0),  # [1, memory_size, 768]
            dim=-1
        )  # [batch, memory_size]

        # Apply TMD mask
        memory_similarities = memory_similarities * tmd_mask

        # Read from top-k slots
        k = 10
        topk_vals, topk_idx = torch.topk(memory_similarities, k, dim=-1)
        read_weights = F.softmax(topk_vals, dim=-1)  # [batch, k]

        # Gather memory content
        memory_slots = self.memory_bank[topk_idx]  # [batch, k, 768]
        memory_content = (read_weights.unsqueeze(-1) * memory_slots).sum(1)  # [batch, 768]

        # === MEMORY WRITE ===

        if update_memory and self.training:
            write_query = self.write_query(hidden)  # [batch, 768]
            write_value = self.write_value(hidden)  # [batch, 768]

            # Find best slot to update (replace least similar)
            write_similarities = F.cosine_similarity(
                write_query.unsqueeze(1),
                self.memory_bank.unsqueeze(0),
                dim=-1
            )
            write_idx = torch.argmin(write_similarities, dim=-1)  # [batch]

            # Update memory (detached to prevent gradient issues)
            for b in range(batch_size):
                self.memory_bank[write_idx[b]] = write_value[b].detach()

        # === COMBINE GRU + MEMORY ===

        # Gating: how much to use memory vs GRU?
        gate_input = torch.cat([hidden, memory_content], dim=-1)
        memory_gate = torch.sigmoid(self.memory_gate(gate_input))  # [batch, 1]

        # Fuse GRU output with memory
        gru_output = self.output_proj(read_query)  # [batch, 768]
        fused = memory_gate * memory_content + (1 - memory_gate) * gru_output

        return F.normalize(fused, p=2, dim=-1)
```

**Parameters:** ~8.5M (7.1M GRU + 1.4M memory addressing)
**Memory Slots:** 2,048 × 768D = 1.6M persistent parameters
**Context:** 5 vectors + 2,048 memory slots (effective context = infinite!)

---

### Architecture C: **Baseline GRU (Current Best)**
**Goal:** Control experiment - no changes

```python
# Use existing GRUStack from app/lvm/models.py
model = GRUStack(
    input_dim=768,
    d_model=512,
    num_layers=4,
    dropout=0.0
)
```

**Parameters:** 7.1M
**Context:** 5 vectors
**Performance:** 0.5625 cosine (367k data)

---

## 🎯 TMD Integration (Existing Framework!)

### TMD as Mixture-of-Experts Router

Your TMD framework (32,768 combinations) is **perfect** for MoE routing!

#### Option 1: Domain-Based Expert Routing (16 experts)

```python
class TMD_DomainExperts(nn.Module):
    """
    Route to 16 domain-specific expert GRUs
    Use TMD domain code (0-15) for routing
    """
    def __init__(self):
        # 16 expert GRUs (one per domain)
        self.experts = nn.ModuleList([
            GRUStack(768, 512, 4) for _ in range(16)
        ])

        # TMD domain extractor
        self.tmd_domain_proj = nn.Linear(16, 16)

    def forward(self, x, tmd_vector):
        # Extract domain from TMD (first 4 bits)
        domain_logits = self.tmd_domain_proj(tmd_vector)  # [batch, 16]
        domain_weights = F.softmax(domain_logits, dim=-1)

        # Route to experts
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weighted_out = domain_weights[:, i:i+1] * expert_out
            outputs.append(weighted_out)

        # Combine expert outputs
        final = sum(outputs)
        return F.normalize(final, dim=-1)
```

#### Option 2: Task-Based Expert Routing (32 experts)

```python
class TMD_TaskExperts(nn.Module):
    """
    Route to 32 task-specific expert GRUs
    Use TMD task code (0-31) for routing
    """
    def __init__(self):
        # 32 expert GRUs (one per task type)
        self.experts = nn.ModuleList([
            GRUStack(768, 512, 2) for _ in range(32)  # Smaller per expert
        ])

        self.tmd_task_proj = nn.Linear(16, 32)

    def forward(self, x, tmd_vector):
        # Extract task from TMD (bits 7-11)
        task_logits = self.tmd_task_proj(tmd_vector)
        task_weights = F.softmax(task_logits, dim=-1)

        # Sparse routing: activate top-2 experts
        topk_weights, topk_idx = torch.topk(task_weights, k=2, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Only run top-2 experts (sparse MoE!)
        outputs = []
        for i in range(2):
            expert_idx = topk_idx[:, i]
            expert_weight = topk_weights[:, i:i+1]

            # Batch-wise expert selection (simplified)
            expert_out = self.experts[expert_idx[0]](x)
            outputs.append(expert_weight * expert_out)

        final = sum(outputs)
        return F.normalize(final, dim=-1)
```

**Parameters:**
- 16 domain experts: 16 × 7.1M = 113.6M total, 7.1M active
- 32 task experts: 32 × 3.5M = 112M total, 7M active (top-2)

---

## 🔬 CPESH for Reinforcement Learning

### Current Status
- ✅ CPESH schema defined (Concept-Probe-Expected-SoftNegatives-HardNegatives)
- ✅ Infrastructure implemented
- ❌ **Disabled for speed** (not being generated during ingestion)
- ❌ Database shows: 0 entries with CPESH data

### CPESH Structure
```python
class CPESH(BaseModel):
    concept: str              # "oxidoreductase activity"
    probe: str               # "What catalyzes redox reactions?"
    expected: str            # "Enzymes with oxidoreductase activity"
    soft_negatives: List[str]  # ["kinase", "transferase", ...]
    hard_negatives: List[str]  # ["dehydrogenase", "reductase", ...]
```

### How CPESH Enables RL/Contrastive Learning

#### Approach 1: Contrastive Prediction Loss

```python
def cpesh_contrastive_loss(model, context, cpesh_data):
    """
    Train model to:
    - Predict vectors closer to 'expected' concepts
    - Stay away from 'hard_negatives'
    - Moderately avoid 'soft_negatives'
    """
    pred = model(context)  # [batch, 768]

    # Positive: expected next concept
    expected_vec = encode(cpesh_data['expected'])
    pos_sim = F.cosine_similarity(pred, expected_vec)

    # Hard negatives: should be dissimilar
    hard_neg_vecs = encode_batch(cpesh_data['hard_negatives'])
    hard_neg_sim = F.cosine_similarity(
        pred.unsqueeze(1),
        hard_neg_vecs,
        dim=-1
    )

    # Soft negatives: moderately dissimilar
    soft_neg_vecs = encode_batch(cpesh_data['soft_negatives'])
    soft_neg_sim = F.cosine_similarity(
        pred.unsqueeze(1),
        soft_neg_vecs,
        dim=-1
    )

    # Contrastive loss
    loss = (
        -torch.log(pos_sim + 1e-8) +  # Maximize expected similarity
        0.5 * torch.mean(hard_neg_sim) +  # Minimize hard negative similarity
        0.25 * torch.mean(soft_neg_sim)   # Moderate soft negative penalty
    )

    return loss
```

#### Approach 2: RLHF with CPESH Rewards

```python
class CPESHRewardModel(nn.Module):
    """
    Reward model trained on CPESH data:
    - High reward: Predictions close to 'expected'
    - Low reward: Predictions close to 'hard_negatives'
    """
    def __init__(self):
        self.scorer = nn.Linear(768 * 2, 1)

    def compute_reward(self, pred, cpesh_data):
        expected_vec = encode(cpesh_data['expected'])
        concat = torch.cat([pred, expected_vec], dim=-1)
        reward = torch.sigmoid(self.scorer(concat))
        return reward

# Use in PPO/REINFORCE training
# (deferred to Phase 2)
```

---

## 📋 Implementation Plan

### Experiment Setup

| Experiment | Architecture | Context | TMD | CPESH | Priority |
|------------|--------------|---------|-----|-------|----------|
| **A** | Hierarchical GRU | 100 vecs | Optional | No | 🔴 HIGH |
| **B** | Memory GRU | 5 vecs | Yes | No | 🟡 MEDIUM |
| **C** | Baseline GRU | 5 vecs | No | No | ✅ DONE |
| **D** | TMD-MoE (16 experts) | 5 vecs | Yes | No | 🟢 LOW |
| **E** | CPESH-Contrastive | 5 vecs | No | Yes | ⚪ FUTURE |

### Phase 1: Core Experiments (This Week)

#### Step 1: Prepare Extended Context Data

```bash
# Export training data with 100-vector context
python tools/export_lvm_training_data_extended.py \
  --input artifacts/wikipedia_500k_corrected_vectors.npz \
  --context-length 100 \
  --output-dir artifacts/lvm/data_extended/

# Expected output:
# - artifacts/lvm/data_extended/training_sequences_ctx100.npz
# - ~367,373 samples (same as ctx5)
# - Context shape: [batch, 100, 768]
```

#### Step 2: Implement Architectures

```bash
# Create new model files
touch app/lvm/hierarchical_gru.py
touch app/lvm/memory_gru.py

# Implement HierarchicalGRU class
# Implement MemoryAugmentedGRU class
```

#### Step 3: Train All 3 Models

```bash
# Experiment A: Hierarchical (100-vector context)
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type hierarchical_gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 20 \
  --batch-size 32 \
  --device mps \
  --output-dir artifacts/lvm/models_extended/hierarchical_gru/

# Experiment B: Memory GRU (5-vector + memory)
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type memory_gru \
  --data artifacts/lvm/data/training_sequences_ctx5.npz \
  --epochs 20 \
  --batch-size 64 \
  --device mps \
  --output-dir artifacts/lvm/models_extended/memory_gru/

# Experiment C: Baseline (already done)
# Use: artifacts/lvm/models_367k/gru/best_model.pt
```

#### Step 4: Fair Comparison

```bash
# Evaluate all 3 on same test set
python tools/compare_extended_context_models.py \
  --baseline artifacts/lvm/models_367k/gru/best_model.pt \
  --hierarchical artifacts/lvm/models_extended/hierarchical_gru/best_model.pt \
  --memory artifacts/lvm/models_extended/memory_gru/best_model.pt \
  --test-data artifacts/lvm/data/test_set_10k.npz
```

---

### Phase 2: TMD Integration (Next Week)

```bash
# Re-enable TMD generation during ingestion
LNSP_TMD_MODE=llm \
LNSP_LLM_ENDPOINT="http://localhost:11434" \
LNSP_LLM_MODEL="llama3.1:8b" \
./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 0 \
  --limit 1000

# Train TMD-aware models
# (implementation deferred)
```

---

### Phase 3: CPESH Integration (Future)

```bash
# Re-enable CPESH generation
LNSP_CPESH_ENABLED=1 \
./tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 0 \
  --limit 1000

# Train with contrastive loss
# (implementation deferred)
```

---

## 📊 Expected Results

### Baseline (Current)
- Model: GRU (7.1M params)
- Context: 5 vectors
- Performance: 0.5625 cosine (367k data)

### Experiment A: Hierarchical
- Model: Hierarchical GRU (14M params)
- Context: 100 vectors (20x expansion!)
- **Expected:** 0.58-0.60 cosine (+3-6% improvement)
- **Rationale:** Long-range dependencies, better narrative flow

### Experiment B: Memory GRU
- Model: Memory GRU (8.5M params)
- Context: 5 vectors + 2,048 memory slots
- **Expected:** 0.57-0.59 cosine (+1-4% improvement)
- **Rationale:** Persistent concept memory, domain knowledge retention

### Experiment C: Baseline
- **Confirmed:** 0.5625 cosine

---

## 🎯 Success Metrics

| Metric | Baseline | Target (A or B) | Stretch Goal |
|--------|----------|-----------------|--------------|
| **Val Cosine** | 0.5625 | >0.58 (+3%) | >0.60 (+6%) |
| **Inference Speed** | 0.56ms | <5ms | <2ms |
| **Parameters** | 7.1M | <20M | <15M |
| **Context Window** | 100 tokens | >1,000 tokens | >5,000 tokens |

---

## 📁 Deliverables

### Code
- `app/lvm/hierarchical_gru.py` - Hierarchical GRU implementation
- `app/lvm/memory_gru.py` - Memory-augmented GRU implementation
- `tools/export_lvm_training_data_extended.py` - Extended context data export
- `tools/compare_extended_context_models.py` - Fair comparison script

### Documentation
- Training logs for all 3 experiments
- Performance comparison table
- Inference speed benchmarks
- Context length ablation study

### Models
- `artifacts/lvm/models_extended/hierarchical_gru/best_model.pt`
- `artifacts/lvm/models_extended/memory_gru/best_model.pt`
- Baseline: `artifacts/lvm/models_367k/gru/best_model.pt`

---

## 🚀 Next Steps

**Immediate (Tonight/Tomorrow):**
1. Wait for 18-hour ingestion to complete (~600k concepts)
2. Export training data with ctx=100
3. Implement Hierarchical GRU
4. Implement Memory GRU
5. Train all 3 models

**This Week:**
6. Run fair comparison
7. Analyze results
8. Document findings

**Next Week:**
9. Integrate TMD routing (if experiments A/B succeed)
10. Scale to larger contexts (500-1000 vectors)

---

**Status:** Ready to implement! Partner mode activated! 🤝

Let's build the best vector-native LVM in the world! 🚀
