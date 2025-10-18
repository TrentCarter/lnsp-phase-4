# Iterative Training & Testing Strategy
## Wikipedia 500k LVM Training

**Status**: Ingestion in progress (Batch 2/50)  
**Created**: 2025-10-15  
**Critical**: Do NOT wait for full ingestion to start validation!

---

## 🎯 Philosophy: Test Early, Test Often, Fail Fast

**Key Principle**: Validate data quality and training feasibility at EVERY stage.  
**Risk**: Waiting 17 days only to discover Wikipedia chunks aren't sequential = disaster.

---

## 📊 4-Phase Strategy

### Phase 1: Data Quality Validation (NOW - Batch 1 Complete ✅)

**When**: Immediately after Batch 1 completes  
**Data**: 10,000 articles, ~200,000 chunks  
**Goal**: Verify Wikipedia is suitable for LVM training

#### 1.1 Sequential Coherence Test

```bash
# Test if consecutive chunks have narrative flow
PYTHONPATH=. ./.venv/bin/python tools/test_sequential_coherence.py \
  --dataset wikipedia_500k \
  --limit 1000 \
  --output artifacts/validation/coherence_batch1.json
```

**Pass Criteria**:
- Mean similarity ≥ 0.3 (consecutive chunks are topically related)
- High coherence % ≥ 60% (most pairs have good flow)
- Low coherence % ≤ 20% (few disconnected pairs)

**Fail Action**: 
- If mean < 0.2: **STOP INGESTION**, fix chunking strategy
- If mean 0.2-0.3: Acceptable, but investigate outliers

#### 1.2 TMD Distribution Check

```bash
# Verify TMD codes are meaningful
psql lnsp << SQL
SELECT 
  domain_code,
  count(*) as chunks,
  round(100.0 * count(*) / sum(count(*)) OVER (), 2) as pct
FROM cpe_entry 
WHERE dataset_source = 'wikipedia_500k'
GROUP BY domain_code 
ORDER BY count DESC;
SQL
```

**Pass Criteria**:
- Top domain < 50% (not all articles in one domain)
- "Unknown" (code 0) < 30% (most articles classified)
- At least 5 domains with > 5% representation

**Fail Action**:
- If one domain > 70%: TMD is broken, investigate LLM prompts
- If Unknown > 50%: Hybrid mode falling back too often

#### 1.3 Chunk Boundary Analysis

```bash
# Check if chunks end cleanly (sentence boundaries)
PYTHONPATH=. ./.venv/bin/python tools/analyze_chunk_boundaries.py \
  --limit 1000 \
  --dataset wikipedia_500k \
  --output artifacts/validation/boundaries_batch1.json
```

**Pass Criteria**:
- Sentence-complete endings ≥ 70%
- Average chunk length: 250-400 chars
- No chunks < 50 chars or > 1000 chars

**Fail Action**:
- If < 50% sentence-complete: Adjust semantic chunker settings
- If chunks too short/long: Re-tune chunk size parameters

#### 1.4 Vec2Text Baseline Test

```bash
# Establish baseline reconstruction quality
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu \
./venv/bin/python3 tools/test_vec2text_baseline.py \
  --sample-size 500 \
  --dataset wikipedia_500k \
  --backends jxe,ielab \
  --output artifacts/validation/vec2text_baseline_batch1.json
```

**Metrics to Record**:
- BLEU-1, BLEU-4 scores
- Cosine similarity (vector space preservation)
- Exact match rate (perfect reconstructions)
- Average inference time (latency baseline)

**Baseline to Beat**:
- BLEU-4 ≥ 0.15 (known vec2text limitation)
- Cosine similarity ≥ 0.7 (semantic preservation)
- LVM must beat BOTH metrics to be worthwhile

---

### Phase 2: Early Training Experiment (After Batch 5-10)

**When**: Day 3-4 of ingestion (~100k articles, ~2M chunks)  
**Goal**: Prove LVM training is feasible, catch major issues early  
**Duration**: ~6-8 hours training time

#### 2.1 Prepare Training Data

```bash
# Create training sequences with context windows
PYTHONPATH=. ./.venv/bin/python tools/prepare_wikipedia_training_data.py \
  --limit 100000 \
  --context-window 5 \
  --stride 2 \
  --output artifacts/lvm/wikipedia_100k_ctx5.npz \
  --val-split 0.1
```

**Output Files**:
- `wikipedia_100k_ctx5.npz`: Training sequences (vector indices)
- `wikipedia_100k_ctx5_val.npz`: Validation set (10%)
- `wikipedia_100k_ctx5_meta.json`: Metadata (concept IDs, stats)

**Validation**:
- Check NPZ shape: (N, context_window, 768)
- Verify sequences are in-order (not shuffled across articles)
- Confirm no duplicate sequences

#### 2.2 Train Small Mamba LVM

```bash
# Train 130M parameter Mamba model
PYTHONPATH=. ./.venv/bin/python app/mamba/train.py \
  --data artifacts/lvm/wikipedia_100k_ctx5.npz \
  --val-data artifacts/lvm/wikipedia_100k_ctx5_val.npz \
  --model-size small \
  --d-model 768 \
  --n-layers 12 \
  --epochs 3 \
  --batch-size 32 \
  --lr 5e-4 \
  --output models/lvm_wikipedia_100k_v1.pt \
  --log-dir logs/training/lvm_100k_v1
```

**Monitor During Training**:
- Loss should decrease (not flatline or spike)
- Validation loss should track training loss (no overfitting)
- GPU memory usage < 16GB (for single GPU training)
- Training time ~6-8 hours for 3 epochs

**Red Flags**:
- Loss not decreasing after 1 epoch: **STOP**, check data format
- Validation loss diverging: Overfitting, reduce model size
- NaN/Inf in loss: Gradient explosion, lower learning rate

#### 2.3 Reconstruction Quality Test

```bash
# Compare LVM vs vec2text baseline
PYTHONPATH=. ./.venv/bin/python tools/compare_reconstruction.py \
  --lvm-model models/lvm_wikipedia_100k_v1.pt \
  --vec2text-backends jxe,ielab \
  --test-set artifacts/validation/wikipedia_test_1k.jsonl \
  --output artifacts/validation/reconstruction_lvm_vs_vec2text.json
```

**Success Metrics**:
- LVM BLEU-4 ≥ Vec2Text BLEU-4 (equal or better)
- LVM cosine similarity ≥ 0.6 (semantic preservation)
- LVM inference time < 1s per query (practical latency)

**Decision Point**:
- ✅ LVM ≥ baseline on all metrics: **Continue to Phase 3**
- ⚠️ LVM ~= baseline (within 5%): Acceptable, tune hyperparams
- ❌ LVM < baseline significantly: **CRITICAL ISSUE**
  - Investigate: Is data truly sequential?
  - Check: Model architecture correct?
  - Verify: Vectors normalized properly?
  - **DO NOT PROCEED** until issue resolved

#### 2.4 Semantic Vector Arithmetic Test

```bash
# Test if LVM learns meaningful vector space
PYTHONPATH=. ./.venv/bin/python tools/semantic_vector_arithmetic_eval.py \
  --lvm-model models/lvm_wikipedia_100k_v1.pt \
  --output artifacts/validation/vector_arithmetic_lvm.json
```

**Test Cases**:
- "Paris" - "France" + "Italy" ≈ "Rome"
- "King" - "Man" + "Woman" ≈ "Queen"
- "Walking" - "Walk" + "Swim" ≈ "Swimming"

**Pass Criteria**:
- Top-3 accuracy ≥ 50% (LVM preserves analogies)
- Better than random (>10% for vocab of 1000)

---

### Phase 3: Mid-Scale Training (After Batch 20-25)

**When**: Day 8-10 of ingestion (~250k articles, ~5M chunks)  
**Goal**: Tune hyperparameters, establish production config  
**Duration**: ~24-36 hours training time

#### 3.1 Hyperparameter Grid Search

**Test Matrix**:

| Param | Values to Test |
|-------|----------------|
| Context Window | 5, 8, 12 |
| Learning Rate | 1e-4, 5e-4, 1e-3 |
| Batch Size | 32, 64, 128 |
| Model Size | small (130M), medium (350M) |
| Epochs | 3, 5, 10 |

**Recommended Experiments** (prioritize these):

1. **Baseline Config**:
   - Context: 5, LR: 5e-4, Batch: 32, Size: small, Epochs: 3

2. **Larger Context**:
   - Context: 8, LR: 5e-4, Batch: 32, Size: small, Epochs: 3

3. **Larger Model**:
   - Context: 5, LR: 5e-4, Batch: 64, Size: medium, Epochs: 3

4. **Longer Training**:
   - Context: 5, LR: 5e-4, Batch: 32, Size: small, Epochs: 10

**Tracking**:
- Use W&B or TensorBoard for experiment tracking
- Record: Loss curves, BLEU scores, training time, memory usage
- Save all model checkpoints for comparison

#### 3.2 Benchmark Suite

```bash
# Full vecRAG benchmark with LVM
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 500 \
  --topk 10 \
  --backends vec,bm25,lex,lvm \
  --lvm-model models/lvm_wikipedia_250k_best.pt \
  --out artifacts/validation/bench_250k_lvm.jsonl
```

**Metrics to Compare**:
- Recall@10 (retrieval quality)
- Precision@10 (ranking quality)
- Inference latency (speed)
- Memory usage (scalability)

**Target Performance**:
- LVM Recall@10 > Vec baseline + 5% (clear improvement)
- LVM latency < 500ms (2x slower than vec2text acceptable)
- LVM memory < 4GB (fits on consumer GPU)

#### 3.3 Error Analysis

```bash
# Analyze failure cases
PYTHONPATH=. ./.venv/bin/python tools/analyze_lvm_failures.py \
  --lvm-model models/lvm_wikipedia_250k_best.pt \
  --test-set artifacts/validation/wikipedia_test_1k.jsonl \
  --output artifacts/validation/failure_analysis.json
```

**Questions to Answer**:
- What types of queries fail? (abstract concepts, named entities, etc.)
- Are failures consistent across vec2text and LVM?
- Do certain domains have higher error rates?
- Are long/short chunks harder to reconstruct?

---

### Phase 4: Full-Scale Production Training (After All 50 Batches)

**When**: Day 17+ (~500k articles, ~10M chunks)  
**Goal**: Train production-ready LVM  
**Duration**: ~3-5 days training time

#### 4.1 Final Data Preparation

```bash
# Full dataset with augmentation
PYTHONPATH=. ./.venv/bin/python tools/prepare_wikipedia_training_data.py \
  --limit 500000 \
  --context-window 12 \
  --stride 6 \
  --augment \
  --val-split 0.05 \
  --output artifacts/lvm/wikipedia_500k_ctx12_aug.npz
```

**Augmentation Techniques**:
- Dropout: Randomly mask 10% of context vectors
- Noise injection: Add Gaussian noise (σ=0.01) to vectors
- Sequence shuffling: Shuffle within-context order (preserve article boundaries)

#### 4.2 Train Large Production LVM

```bash
# 760M parameter model with gradient checkpointing
PYTHONPATH=. ./.venv/bin/python app/mamba/train.py \
  --data artifacts/lvm/wikipedia_500k_ctx12_aug.npz \
  --val-data artifacts/lvm/wikipedia_500k_ctx12_aug_val.npz \
  --model-size large \
  --d-model 768 \
  --n-layers 24 \
  --epochs 10 \
  --batch-size 128 \
  --lr 5e-4 \
  --mixed-precision \
  --gradient-checkpointing \
  --output models/lvm_wikipedia_500k_production.pt \
  --log-dir logs/training/lvm_500k_production
```

**Hardware Requirements**:
- GPU: A100 40GB or 2x RTX 3090 24GB
- RAM: 64GB+
- Disk: 200GB+ for checkpoints
- Time: 3-5 days

**Checkpointing Strategy**:
- Save checkpoint every epoch
- Keep best 3 checkpoints (by validation loss)
- Save final checkpoint after 10 epochs

#### 4.3 Final Validation & Benchmarking

```bash
# Comprehensive benchmark on held-out test set
PYTHONPATH=. ./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 10000 \
  --topk 10 \
  --backends vec,bm25,lex,lvm \
  --lvm-model models/lvm_wikipedia_500k_production.pt \
  --out artifacts/validation/bench_500k_production.jsonl
```

**Production Metrics**:
- Recall@10 > 0.85 (high retrieval quality)
- Precision@10 > 0.70 (good ranking)
- Latency < 500ms (acceptable for production)
- Memory < 8GB (deployable on consumer GPU)

#### 4.4 Deployment Readiness Check

- [ ] Model weights < 4GB (compressed)
- [ ] ONNX export successful
- [ ] Quantization (INT8) tested
- [ ] Batch inference optimized
- [ ] API integration tested
- [ ] Load testing completed (1000 QPS)
- [ ] Monitoring/logging configured

---

## 🚨 Critical Decision Points

### STOP Points (Do NOT Continue)

1. **Phase 1 Coherence Test Fails** (mean similarity < 0.2)
   - Action: Fix chunking strategy before continuing ingestion

2. **Phase 2 LVM << Baseline** (>20% worse)
   - Action: Investigate root cause, do NOT scale up

3. **Phase 3 No Improvement** (LVM ~= baseline after tuning)
   - Action: Question if LVM is right approach for this data

### GO Points (Proceed Confidently)

1. **Phase 1 All Tests Pass**
   - Coherence ≥ 0.3, TMD reasonable, boundaries clean
   - Action: Continue ingestion, plan Phase 2

2. **Phase 2 LVM ≥ Baseline**
   - BLEU and cosine equal or better than vec2text
   - Action: Scale up to Phase 3

3. **Phase 3 Clear Improvement**
   - LVM Recall@10 > baseline + 5%
   - Action: Proceed to full-scale training

---

## 📊 Progress Tracking

### Phase 1: Data Validation
- [ ] Sequential coherence test (Target: ≥0.3)
- [ ] TMD distribution check (Target: <50% in one domain)
- [ ] Chunk boundary analysis (Target: ≥70% clean)
- [ ] Vec2text baseline (Target: BLEU ≥0.15)

### Phase 2: Early Training (100k articles)
- [ ] Training data prepared
- [ ] Small Mamba model trained (130M params)
- [ ] Reconstruction test (Target: ≥ baseline)
- [ ] Vector arithmetic test (Target: ≥50% top-3)

### Phase 3: Mid-Scale Training (250k articles)
- [ ] Hyperparameter grid search (12 experiments)
- [ ] Benchmark suite (Target: +5% recall)
- [ ] Error analysis completed
- [ ] Best config identified

### Phase 4: Production Training (500k articles)
- [ ] Full dataset prepared with augmentation
- [ ] Large model trained (760M params)
- [ ] Final benchmarks (Target: R@10 > 0.85)
- [ ] Deployment readiness verified

---

## 🛠️ Tools Reference

| Tool | Purpose | Usage |
|------|---------|-------|
| `test_sequential_coherence.py` | Validate consecutive chunks have narrative flow | Phase 1 |
| `prepare_wikipedia_training_data.py` | Convert chunks → training sequences | Phase 2-4 |
| `app/mamba/train.py` | Train Mamba LVM | Phase 2-4 |
| `compare_reconstruction.py` | LVM vs vec2text comparison | Phase 2-4 |
| `semantic_vector_arithmetic_eval.py` | Test vector space quality | Phase 2 |
| `RAG/bench.py` | Full retrieval benchmark | Phase 3-4 |
| `analyze_lvm_failures.py` | Error analysis | Phase 3 |

---

## 📝 Reporting

### After Each Phase

Create a report:
- `artifacts/reports/phase_N_report.md`
- Include: All metrics, decision rationale, next steps
- Commit to git for version control

### Example Report Template

```markdown
# Phase N Report: [Title]

**Date**: YYYY-MM-DD  
**Data**: N articles, M chunks  
**Duration**: X hours

## Summary

[2-3 sentence summary of phase]

## Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ... | ... | ... | ✅/❌ |

## Key Findings

- Finding 1
- Finding 2

## Decision

✅ **GO** / ❌ **STOP** / ⚠️ **WARN**

Rationale: [why?]

## Next Steps

- [ ] Action 1
- [ ] Action 2
```

---

## 🎯 Success Criteria (Final)

**LVM is production-ready if ALL of:**
- ✅ Recall@10 > 0.85 (high retrieval quality)
- ✅ BLEU-4 > 0.20 (better than vec2text)
- ✅ Latency < 500ms (acceptable speed)
- ✅ Memory < 8GB (deployable)
- ✅ Passes load testing (1000 QPS)

If any criterion fails, LVM is NOT production-ready. Ship vec2text baseline instead.

---

**Remember**: It's better to discover LVM doesn't work on Day 4 than Day 17. Test early, fail fast, iterate quickly.
