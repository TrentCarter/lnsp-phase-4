# PRD: Two-Tower Retriever Training Specification

**Status**: APPROVED - Ready for Implementation  
**Priority**: P0 (Blocks production deployment)  
**Owner**: TBD  
**Created**: 2025-10-20  
**Timeline**: 3-5 days implementation

---

## Executive Summary

### The Problem

Current cascade retrieval achieves only **0.65% Hit@5** on full-bank retrieval (771k vectors), despite:
- ✅ FAISS infrastructure working perfectly (97.40% oracle recall)
- ✅ Phase-3 LVM working well on small-set re-ranking (75.65% Hit@5 on 8-candidate batches)
- ✅ RRF fusion combining signals effectively

**Root Cause**: Query formation. Using last context vector achieves only 4.55% Recall@5, while oracle (true target) achieves 97.40%. The 60% gap cannot be closed with heuristics:

| Query Strategy | Recall@500 | vs Oracle |
|----------------|------------|-----------|
| Oracle (true target) | 97.40% | - |
| Best heuristic (exp weighted) | 38.96% | -60% |
| Last vector (baseline) | 35.71% | -63% |
| Mean averaging | 1.30% | -99% |

### The Solution

Train a **two-tower retriever** that learns: `context → query_vector` for full-bank search.

**Expected Impact**:
- Stage-1 Recall@500: 35.71% → **55-60%** (+55% relative)
- End-to-end Hit@5: 0.65% → **10-20%** (+1,438% relative)
- Production-ready: Full-bank retrieval with <50ms P95 latency

---

## Architecture

### Two-Tower Design

```
Input Context (1000 vectors, 768D each)
    ↓
[Query Tower f_q]          [Document Tower f_d]
    ↓                              ↓
Query Vector (768D)         Document Vectors (771k × 768D)
    ↓                              ↓
    └──────── Cosine Similarity ──┘
                    ↓
            Top-K Candidates
                    ↓
         [Phase-3 LVM Re-rank]
                    ↓
            [TMD Re-rank]
                    ↓
              Final Results
```

### Query Tower (f_q)

**Input**: Sequence of vectors `[v_1, v_2, ..., v_1000]` (context)  
**Output**: Single query vector `q ∈ ℝ^768`

**Architecture options** (ranked by complexity):

1. **Lightweight GRU + Pooling** (RECOMMENDED - fast, proven)
   ```python
   class QueryTower(nn.Module):
       def __init__(self):
           self.gru = nn.GRU(768, 512, num_layers=2, batch_first=True)
           self.proj = nn.Linear(512, 768)
       
       def forward(self, context):
           # context: [batch, seq_len, 768]
           _, hidden = self.gru(context)  # hidden: [2, batch, 512]
           pooled = hidden[-1]  # Last layer: [batch, 512]
           query = self.proj(pooled)  # [batch, 768]
           return F.normalize(query, dim=-1)
   ```
   - Params: ~2.5M
   - Inference: ~5ms P95
   - Proven: Similar to GTR-T5 query encoder

2. **Mean Pooling + MLP** (FASTEST - baseline)
   ```python
   query = context.mean(dim=1)  # [batch, 768]
   query = MLP(query)  # [batch, 768]
   ```
   - Params: ~1M
   - Inference: ~1ms P95
   - Risk: May not capture sequential dependencies

3. **Hierarchical Transformer** (HIGHEST QUALITY - slow)
   ```python
   # Chunk 1000 → 10 chunks of 100
   # Attend within chunks, then across chunks
   ```
   - Params: ~10M
   - Inference: ~50ms P95
   - Risk: Latency bottleneck

**Recommendation**: Start with **Option 1 (GRU + Pooling)** for balance of speed and quality.

### Document Tower (f_d)

**Input**: Single vector `v ∈ ℝ^768` (candidate document)  
**Output**: Single embedding `d ∈ ℝ^768`

**Options**:

1. **Identity (tied weights)** (RECOMMENDED)
   ```python
   def forward(self, vec):
       return F.normalize(vec, dim=-1)
   ```
   - No additional params
   - Uses existing GTR-T5 embeddings directly
   - Fastest inference
   - **Advantage**: Leverages pre-trained semantic space

2. **Untied small MLP**
   ```python
   def forward(self, vec):
       return F.normalize(self.mlp(vec), dim=-1)
   ```
   - ~0.5M params
   - Can refine embeddings for retrieval task
   - Risk: May diverge from semantic space

**Recommendation**: Start with **Option 1 (Identity)** to leverage GTR-T5 quality. Add MLP only if needed.

---

## Training Data

### Data Sources

**Primary**: Phase-3 validation sequences (1,540 pairs from 771k bank)

```python
# Each training sample:
{
    'context': np.array([v_1, ..., v_1000]),  # (1000, 768)
    'target_id': int,  # Index in 771k bank
    'target_vec': np.array(768,)  # The true next vector
}
```

**Negatives Strategy**:

1. **In-batch negatives** (free)
   - Every other document in batch is a negative
   - Effective batch size: 256 → 255 negatives per positive

2. **Memory-bank queue** (10-50k recent docs)
   - FIFO queue of recently seen documents
   - Smooth negative distribution over training

3. **ANN-mined hard negatives** (crucial for quality)
   - Every 1-2 epochs, mine hard negatives using current model
   - Target: Cosine similarity 0.80-0.95 to positive
   - These are "confusors" - similar but wrong answers

### Data Augmentation

Optional techniques to increase effective dataset size:

1. **Context truncation**: Use last 500/700 vectors instead of 1000
2. **Temporal offset**: Shift target by ±1 position
3. **Lane-specific mining**: Over-sample weak lanes (if TMD available)

**Recommendation**: Start without augmentation (1,540 pairs may be sufficient with good negatives).

---

## Loss Function

### InfoNCE (Contrastive Loss)

```python
def infonce_loss(query, positive, negatives, temperature=0.07):
    """
    Args:
        query: [batch, 768]
        positive: [batch, 768]
        negatives: [batch, num_neg, 768] or [num_neg, 768]
    """
    # Similarities
    pos_sim = (query * positive).sum(dim=-1) / temperature  # [batch]
    neg_sim = torch.matmul(query, negatives.T) / temperature  # [batch, num_neg]
    
    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1+num_neg]
    labels = torch.zeros(batch_size, dtype=torch.long)  # Positive is index 0
    
    loss = F.cross_entropy(logits, labels)
    return loss
```

**Temperature (τ)**: Critical hyperparameter
- τ = 0.05: Sharp distribution, aggressive contrastive learning
- τ = 0.07: **RECOMMENDED** (GTR-T5 default)
- τ = 0.10: Softer distribution, easier optimization

### Optional: Margin Loss for Hard Negatives

```python
def margin_loss(query, positive, hard_negatives, margin=0.2):
    """
    Additive loss: ensure hard negatives are at least 'margin' worse than positive.
    """
    pos_sim = (query * positive).sum(dim=-1)
    hard_sim = torch.matmul(query, hard_negatives.T).max(dim=-1)[0]
    
    loss = F.relu(margin - (pos_sim - hard_sim))
    return loss.mean()
```

**Recommendation**: Start with InfoNCE only. Add margin loss if plateauing.

---

## Evaluation During Training

### Must-Have Metrics

Evaluate **every epoch** on held-out validation set (10-20% of data):

```python
# Recall@K on full bank
for context, target_id in val_set:
    query = query_tower(context)
    distances, indices = faiss_index.search(query, k=1000)
    
    # Check if target_id in top-K
    for k in [10, 100, 500, 1000]:
        recall_at_k[k] += (target_id in indices[:k])
```

**Primary metric**: **Recall@500** (Stage-1 recall)  
**Target**: ≥ 55-60%

**Secondary metrics**:
- Recall@100: Track precision-recall tradeoff
- Recall@1000: Sanity check (should be high)
- MRR (Mean Reciprocal Rank): Average rank of true target

### Early Stopping

```python
patience = 5-8 epochs
monitor = 'recall@500'
min_delta = 0.01  # Stop if improvement < 1%
```

**SWA (Stochastic Weight Averaging)**: Average last 3-5 checkpoints for +1-2% recall boost.

---

## Training Configuration

### Recommended Hyperparameters

```python
config = {
    # Model
    'query_tower': 'gru_pooling',  # Options: gru_pooling, mean_mlp, transformer
    'doc_tower': 'identity',  # Options: identity, mlp
    'hidden_dim': 512,
    'num_layers': 2,
    
    # Optimization
    'optimizer': 'adamw',
    'lr': 2e-5,  # Start low (pre-trained embeddings)
    'weight_decay': 0.01,
    'lr_schedule': 'cosine',  # Cosine annealing
    'warmup_epochs': 1,
    
    # Training
    'batch_size': 32,  # Per-GPU
    'accumulation_steps': 8,  # Effective batch = 256
    'effective_batch_size': 256,
    'epochs': 50,
    'gradient_clip': 1.0,
    
    # Loss
    'temperature': 0.07,
    'use_margin_loss': False,  # Enable if needed
    'margin': 0.2,
    
    # Negatives
    'in_batch_negatives': True,
    'memory_bank_size': 10000,
    'hard_negative_mining_freq': 2,  # Every 2 epochs
    'hard_negative_count': 64,
    
    # Evaluation
    'eval_every': 1,  # Every epoch
    'recall_k_values': [10, 100, 500, 1000],
    'patience': 5,
    'swa_start_epoch': 45,
    
    # Infrastructure
    'device': 'mps',  # or 'cuda'
    'num_workers': 4,
    'pin_memory': True,
}
```

### Expected Training Time

**With 1,540 training pairs**:
- Batch size: 32 → 49 batches/epoch
- Time per batch: ~200ms (forward + backward + optimizer)
- Time per epoch: ~10 seconds
- With eval: ~15 seconds/epoch
- **Total: 50 epochs × 15s = 12.5 minutes per training run**

Fast iteration allows extensive hyperparameter search!

---

## Implementation Plan

### Phase 1: MVP (Day 1-2)

**Goal**: Prove two-tower retriever can beat heuristics (>40% Recall@500)

**Tasks**:
1. ✅ Data preparation (export context/target pairs) - 2 hours
2. ✅ Implement QueryTowerGRU - 2 hours
3. ✅ Implement InfoNCE loss with in-batch negatives - 2 hours
4. ✅ Training loop + Recall@K evaluation - 3 hours
5. ✅ First training run - 15 minutes
6. ✅ Validate Recall@500 > 40% - 1 hour

**Deliverables**:
- `tools/train_two_tower.py` - Training script
- `artifacts/two_tower/run_001/` - First checkpoint
- `artifacts/two_tower/run_001/metrics.json` - Recall@K results

**Success Criteria**: Recall@500 > 40% (beats best heuristic 38.96%)

### Phase 2: Hard Negatives (Day 2-3)

**Goal**: Reach 55-60% Recall@500 with hard negative mining

**Tasks**:
1. ✅ Implement memory-bank queue - 2 hours
2. ✅ Implement ANN-based hard negative mining - 3 hours
3. ✅ Re-train with full negative strategy - 30 minutes
4. ✅ Hyperparameter tuning (τ, lr, batch size) - 2 hours
5. ✅ Validate Recall@500 ≥ 55% - 1 hour

**Deliverables**:
- `artifacts/two_tower/run_best/` - Best checkpoint (Recall@500 ≥ 55%)
- `artifacts/two_tower/grid_search.json` - Hyperparameter grid results

**Success Criteria**: Recall@500 ≥ 55-60%

### Phase 3: Integration (Day 3-5)

**Goal**: Deploy two-tower + Phase-3 + TMD cascade to production

**Tasks**:
1. ✅ Build FAISS index with query tower embeddings - 1 hour
2. ✅ Implement cascade pipeline (TwoTower → FAISS → LVM → TMD) - 3 hours
3. ✅ End-to-end evaluation (Hit@K, latency) - 2 hours
4. ✅ Optimize for P95 latency < 50ms - 3 hours
5. ✅ Integration testing - 2 hours
6. ✅ Documentation + deployment guide - 2 hours

**Deliverables**:
- `tools/eval_two_tower_cascade.py` - End-to-end evaluator
- `docs/two_tower_deployment.md` - Deployment guide
- **Production-ready cascade API**

**Success Criteria**:
- End-to-end Hit@5 ≥ 10-20%
- P95 latency ≤ 50ms
- Recall@500 stable at ≥ 55%

---

## Success Metrics

### Training Metrics (Phase 2 Target)

| Metric | Current (Heuristic) | Target (Two-Tower) | Improvement |
|--------|---------------------|--------------------| ------------|
| Recall@10 | 7.79% | 15-20% | +2x |
| Recall@100 | 27.92% | 40-45% | +50% |
| **Recall@500** | **38.96%** | **≥55-60%** | **+50%** |
| Recall@1000 | 44.16% | 65-70% | +50% |

### End-to-End Metrics (Phase 3 Target)

| Stage | Metric | Target | Current |
|-------|--------|--------|---------|
| Stage-1 (Two-Tower) | Recall@500 | 55-60% | 38.96% |
| Stage-2 (LVM Re-rank) | Hit@50 | 8-12% | 0.65% |
| Stage-3 (TMD) | Hit@10 | 6-10% | 0.65% |
| **Final** | **Hit@5** | **10-20%** | **0.65%** |

**Latency Budget**:
- Two-tower query encoding: 5-10ms
- FAISS search (K=500): 2-5ms
- LVM re-rank (500→50): 10-15ms
- TMD re-rank (50→10): 1-2ms
- **Total P95: ≤50ms** (20x faster than current 614ms!)

---

## Risk Mitigation

### Risk 1: Insufficient Training Data (1,540 pairs)

**Mitigation**:
- ✅ Use aggressive data augmentation (context truncation, temporal offset)
- ✅ Strong negative mining (hard negatives from 771k bank)
- ✅ If still data-starved: Generate synthetic pairs from Wikipedia ingestion

**Fallback**: If Recall@500 plateaus at 45-50%, ingest 10-20k more Wikipedia articles

### Risk 2: Query Tower Overfitting

**Symptoms**: High train Recall@500, low val Recall@500

**Mitigation**:
- ✅ Early stopping with patience=5
- ✅ Strong regularization (dropout=0.1, weight_decay=0.01)
- ✅ SWA (stochastic weight averaging)

### Risk 3: Latency Regression

**Symptoms**: Two-tower inference >50ms P95

**Mitigation**:
- ✅ Profile query tower (target: ≤10ms)
- ✅ Batch query encoding (8-16 queries at once)
- ✅ If needed: Switch to mean pooling + MLP (1ms inference)

### Risk 4: Stage-1 Recall Still Too Low (<50%)

**Mitigation**:
- ✅ Hybrid Stage-1: Combine two-tower + BM25 (RRF fusion)
- ✅ Increase K₀: 500 → 1000 (trade latency for recall)
- ✅ Fine-tune GTR-T5 document tower (currently identity)

---

## Code Structure

### File Organization

```
app/lvm/two_tower/
├── __init__.py
├── models.py              # QueryTower, DocTower classes
├── loss.py                # InfoNCE, margin loss
├── data.py                # Dataset, DataLoader, negatives
├── train.py               # Training loop
└── evaluate.py            # Recall@K evaluation

tools/
├── train_two_tower.py     # Training CLI
├── eval_two_tower.py      # Offline evaluation
└── eval_two_tower_cascade.py  # End-to-end cascade eval

artifacts/two_tower/
├── run_001/
│   ├── config.json
│   ├── checkpoints/
│   │   ├── epoch_001.pt
│   │   ├── epoch_010.pt
│   │   └── best_recall500.pt
│   ├── metrics.json
│   └── training.log
└── production/
    ├── query_tower.pt     # Final query tower
    ├── faiss_index.bin    # FAISS index (if needed)
    └── metadata.json
```

---

## Deployment

### Inference Pipeline

```python
class TwoTowerCascade:
    def __init__(self):
        self.query_tower = load_query_tower('artifacts/two_tower/production/query_tower.pt')
        self.faiss_index = load_faiss_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
        self.lvm_model = load_phase3_model('artifacts/lvm/models_phase3/.../best_val_hit5.pt')
    
    def search(self, context, k=10):
        # Stage 1: Two-tower retrieval
        query = self.query_tower(context)  # 5-10ms
        candidates_500 = self.faiss_index.search(query, k=500)  # 2-5ms
        
        # Stage 2: LVM re-rank
        candidates_50 = self.lvm_model.rerank(context, candidates_500)  # 10-15ms
        
        # Stage 3: TMD re-rank
        final_10 = tmd_rerank(candidates_50, lambda_weight=0.7)  # 1-2ms
        
        return final_10[:k]
```

**Expected latency**: 20-35ms P95 (30x faster than current!)

---

## Next Steps

1. **Architect approval** - Review and approve PRD
2. **Assign owner** - ML engineer to implement
3. **Kickoff Phase 1** - Start MVP training (Day 1-2)
4. **Daily standup** - Track Recall@500 progress
5. **Gate review** - After Phase 2, decide on Phase 3 deployment

**Ready to start implementation!**

