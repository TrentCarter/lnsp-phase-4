# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL: READ LONG-TERM MEMORY FIRST

**Before doing ANYTHING, read [LNSP_LONG_TERM_MEMORY.md](LNSP_LONG_TERM_MEMORY.md)**

That file contains the cardinal rules that must NEVER be violated:
1. Data Synchronization is Sacred (PostgreSQL + Neo4j + FAISS must stay synchronized)
2. NO FactoidWiki Data - Ontologies ONLY
3. Complete Data Pipeline: CPESH + TMD + Graph (all together, atomically)
4. LVM Architecture: Tokenless Vector-Native
5. Six Degrees of Separation + Shortcuts (0.5-3% shortcut edges)

---

## 🔴 CRITICAL: CORRECT ENCODER/DECODER (2025-10-31)

**DO NOT USE PORT 8766 FOR DECODING!** It is NOT compatible with port 8767 encoder despite both claiming to be "GTR-T5".

### ✅ PRODUCTION SERVICES: Ports 7001 (Encode) and 7002 (Decode)

**Start Services:**
```bash
# Start encoder on port 7001
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app --host 127.0.0.1 --port 7001 &

# Start decoder on port 7002
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app --host 127.0.0.1 --port 7002 &

# Check health
curl http://localhost:7001/health
curl http://localhost:7002/health
```

**Usage Example (FastAPI):**
```python
import requests

text = "The Eiffel Tower was built in 1889."

# Encode via port 7001
encode_resp = requests.post("http://localhost:7001/encode", json={"texts": [text]})
vector = encode_resp.json()["embeddings"][0]

# Decode via port 7002
decode_resp = requests.post("http://localhost:7002/decode", json={
    "vectors": [vector],
    "subscriber": "ielab",
    "steps": 5,
    "original_texts": [text]
})
decoded_text = decode_resp.json()["results"][0]

# Result: Meaningful output with 80-100% keyword matches
```

### ✅ CORRECT: Direct Python Usage (IsolatedVecTextVectOrchestrator)

```python
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Initialize orchestrator (ONLY do this ONCE per session)
orchestrator = IsolatedVecTextVectOrchestrator(steps=5, debug=False)

# Encode text to vectors
text = "The Eiffel Tower was built in 1889."
vectors = orchestrator.encode_texts([text])  # Returns torch.Tensor [1, 768]

# Decode vectors back to text
result = orchestrator._run_subscriber_subprocess(
    'ielab',  # or 'jxe'
    vectors.cpu(),
    metadata={'original_texts': [text]},
    device_override='cpu'
)
decoded_text = result['result'][0]  # Actual decoded text
```

### ❌ WRONG: Using port 8767 encoder + port 8766 decoder

```python
# ❌ DON'T DO THIS - Port 8766 decoder is NOT compatible with port 8767 encoder!
encode_resp = requests.post("http://localhost:8767/embed", json={"texts": [text]})
vector = encode_resp.json()["embeddings"][0]

decode_resp = requests.post("http://localhost:8766/decode", json={"vectors": [vector]})
# Result: Gibberish output with cosine ~0.05 (nearly orthogonal)
```

### Why This Matters

- Port 8767 + Port 8766 give **cosine similarity ~0.05** (gibberish)
- Port 7001 + Port 7002 give **80-100% keyword matches** (meaningful output)
- Orchestrator encode + decode gives **meaningful output** with actual keyword matches
- **ONLY use the orchestrator** (ports 7001/7002 or direct Python) - do NOT mix ports 8767/8766

### CPU vs MPS Performance (Oct 31, 2025)

**TL;DR: CPU is 2.93x FASTER than MPS for vec2text decoding**

- ✅ **CPU (ports 7001/7002)**: 1,288ms per decode, 0.78 req/sec
- ⚠️ **MPS (ports 7003/7004)**: 3,779ms per decode, 0.26 req/sec
- 🎯 **Why**: Vec2text's iterative refinement is fundamentally sequential - cannot benefit from GPU parallelism
- 📊 **Batch experiments**: No improvement from batching (3,700ms per item regardless of batch size)
- 🔬 **Root cause**: Each decoding step must wait for the previous one (algorithmic bottleneck, not hardware)

**Production recommendation**: Use CPU services (7001/7002). Even with 12 CPU cores at 100%, it's still faster than MPS.

**See**: `docs/how_to_use_jxe_and_ielab.md` (CPU vs MPS Performance Analysis section) for detailed explanation and test results.

### Port Reference

| Port | Service | Status | Use For |
|------|---------|--------|---------|
| 7001 | Orchestrator Encoder (FastAPI) | ✅ PRODUCTION | **Encoding for full pipeline** |
| 7002 | Orchestrator Decoder (FastAPI) | ✅ PRODUCTION | **Decoding from port 7001** |
| 8767 | GTR-T5 Encoder | ⚠️ Use with caution | Encoding ONLY (standalone, not for decode pipeline) |
| 8766 | Vec2Text Decoder | ❌ INCOMPATIBLE | DO NOT USE with any encoder |
| N/A | IsolatedVecTextVectOrchestrator | ✅ CORRECT | **Direct Python usage** |

---

## 🚨 CRITICAL: WIKIPEDIA DATA IS BACKWARD-BIASED (2025-11-02)

**⚠️ DO NOT TRAIN FORWARD LVM ON RAW WIKIPEDIA DATA**

**Data Analysis Results** (790,391 chunks analyzed):
- **Δ (Forward - Backward)**: **-0.0696** (backward is 7% stronger)
- **100% of articles**: Backward-biased (no exceptions)
- **Root cause**: Explanatory structure (later chunks reference earlier concepts)
- **Offset curve**: Monotonic increase toward recent past (k=-1 strongest)
- **Reversing doesn't help**: Δ_reversed = -0.0395 (still negative)

**Evidence**:
```
Forward (ctx[-1] → target_next):  0.3876 ± 0.1372
Backward (ctx[-1] → target_prev): 0.4572 ± 0.1769
Δ = -0.0696 ± 0.2064

Per-article: Mean Δ = -0.0766, Range [-0.1762, -0.0490]
Worst examples: Δ = -0.92 (chunk 1 after lead, cos(prev) = 0.98)
```

**Why Wikipedia is backward**:
- Lead sections introduce key terms (Einstein, relativity, quantum)
- Later sections reference these terms repeatedly ("As mentioned earlier...", "the Einstein...")
- Explanatory flow (general → specific → detail) not narrative flow (setup → payoff)
- Example: Apollo article chunk 5 says "As patron deity of Delphi" (refers to "deity" in chunk 0)

**Decision Gate**: Δ ≤ -0.05 → **STOP forward training on Wikipedia**

**See Complete Analysis**: `artifacts/lvm/wikipedia_temporal_analysis/REPORT.md`

**Options**:
1. ✅ **Switch to forward-flow data** (arXiv papers, tutorials, stories)
2. ✅ **LLM temporal rewiring** (rewrite chunks to add forward signals)
3. ✅ **Multi-scale hierarchy** (train on sentence/section scales where Δ > 0)
4. ✅ **Contrastive ranking** (rank next > prev instead of exact prediction)
5. ⚠️ **Bi-directional training** (exploit backward signal, schedule forward preference late)
6. ❌ **NOT: Train forward on raw Wikipedia** (will fail with negative margin)

**Sample Articles Extracted**: `artifacts/lvm/SAMPLE_ARTICLES_SUMMARY.md`
- Apollo (1,107 chunks), Russian Orthodox bell ringing (100), William West (50), Dennis Bock (20)
- Database: `psql -h localhost -U lnsp -d lnsp` (790,391 chunks)

---

## 📌 ACTIVE CHECKPOINT: Phase 2 Complete - Decision Point (2025-11-04)

**STATUS**: ✅ **PHASE 2 COMPLETE** - Critical Decision Point

**Results**:
- **Downloaded**: 3,715 arXiv papers (cs.CL, cs.LG, stat.ML, cs.AI)
- **Processed**: 619 papers (17% success rate - filter too aggressive)
- **Vectors**: 111,825 (768D GTR-T5 embeddings)
- **Sequences**: 108,730 (context_size=5)
- **Δ (Forward Bias)**: +0.0638 (6.38% - PASSING but 20% below target)

**Critical Finding**: V2 pre-cleaning filter has 82% false positive rate
- **ROOT CAUSE**: PDF text extraction creates single-line files (no newlines)
- Filter treats 88KB paper as ONE line → rejects as "ASCII art"
- See analysis: `artifacts/lvm/SKIPPED_PAPERS_FOR_FILTER_REVIEW.md`

**Decision Point**:
- **Option A**: Fix filter (6-9 hrs) → 1,500-2,000 papers, 270k-360k vectors, Δ may improve to +0.08-0.10
- **Option B**: Train P6b v2.3 now (8-12 hrs) → Use existing 111k vectors, Δ=+0.06
- **Recommendation**: Option B (data is clean, Δ is passing, faster to results)

**Next Step**: `./scripts/train_transformer_p6b_v23.sh --train-npz artifacts/lvm/arxiv_clean_sequences.npz`

**Complete Documentation**: `artifacts/lvm/SESSION_SUMMARY_2025_11_04_PHASE2_COMPLETE.md`

**Previous Checkpoint**: Wikipedia Ingestion (2025-10-18 - PAUSED)
- **Status**: ⚠️ **PAUSED - Data unsuitable for forward LVM training**
- **Data**: 790,391 concepts (500k articles) in PostgreSQL
- **Analysis**: Δ = -0.0696 (backward bias confirmed)
- **Decision**: DO NOT use Wikipedia for forward LVM training

**If resuming Wikipedia ingestion for OTHER purposes** (vecRAG, GraphRAG):
```bash
# Wikipedia is still useful for retrieval, just not forward LVM training
LNSP_TMD_MODE=hybrid \
LNSP_LLM_ENDPOINT="http://localhost:11434" \
LNSP_LLM_MODEL="llama3.1:8b" \
./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3432 --limit 3000
```

---

## 🎯 PRODUCTION RETRIEVAL CONFIGURATION (2025-10-24)

**STATUS**: ✅ Production Ready - Shard-Assist with ANN Tuning

**Performance**: 73.4% Contain@50, 50.2% R@5, 1.33ms P95

**See**: [docs/RETRIEVAL_OPTIMIZATION_RESULTS.md](docs/RETRIEVAL_OPTIMIZATION_RESULTS.md) for full details

### Quick Reference

**FAISS Configuration**:
```python
nprobe = 64                # ANN probe count (Pareto optimal)
K_global = 50              # Global IVF candidates
K_local = 20               # Per-article shard candidates
```

**Reranking Pipeline**:
```python
mmr_lambda = 0.7           # MMR diversity (FULL POOL, do NOT reduce!)
w_same_article = 0.05      # Same-article bonus
w_next_gap = 0.12          # Next-chunk gap bonus
tau = 3.0                  # Gap penalty temperature
directional_bonus = 0.03   # Directional alignment bonus
```

**Key Files**:
- Evaluation: `tools/eval_shard_assist.py`
- Article Shards: `artifacts/article_shards.pkl` (3.9GB)
- Production Results: `artifacts/lvm/eval_shard_assist_full_nprobe64.json`

**⚠️ DO NOT**:
- Reduce `mmr_lambda` from 0.7 (hurts R@10 by -10pp)
- Apply MMR to limited pool (use full candidate set)
- Use adaptive-K (doesn't help, adds complexity)
- Enable alignment head by default (hurts containment)

---

## 🎓 LVM TRAINING: P6b v2.3 "GOLDILOCKS" (2025-11-02)

**🔴 CRITICAL FINDING (Nov 2, 2025)**: Wikipedia data has inherent backward temporal bias (Δ = -0.0696). All approaches P1-P6b v2.2 failed because **the data itself teaches backward prediction**, not due to model architecture. Confirmed via comprehensive temporal flow analysis of 790,391 chunks.

**STATUS**: 🚀 **P6b v2.3 READY** - But **DO NOT train on raw Wikipedia** (use forward-flow data instead)

### Evolution: P1 → P6 → P6b v1 → v2.1 → v2.2

**P1-P5.1 ALL FAILED** (margin stayed negative):
- P1 Baseline: -0.167 (MSE follows dominant signal → backward)
- P2-P4 Directional: Collapsed or negative (λ too weak)
- P5.1 + Curriculum: -0.046 (landscape reshaping insufficient)

**P6 (NEXT token) ALSO FAILED**:
- Architecture: Removed identity path (cos(ctx[4], target_next) = 0.395)
- Result: R@5 = 70% ✅, margin = **-0.082** ❌ (worse!)
- **Proved**: Problem is data, not architecture

**P6b v1 COLLAPSED at Epoch 3**:
- Too aggressive ramp (4x pressure at epoch 3)
- "Two negatives" failure mode (large gap with both cosines negative)
- Death spiral: negative cosines → higher penalty → worse predictions
- See `artifacts/lvm/P6B_EPOCH3_COLLAPSE_ANALYSIS.md` for full post-mortem

**P6b v2.1 = 6-Layer Defense (✅ COMPLETED)**:
- Model: `artifacts/lvm/models/transformer_p6b_v21_20251102_182615/best_model.pt`
- Result: R@5 = 77% ✅, margin = **-0.047** ⚠️ (improved but still negative!)
- **Improved margin 43%** (-0.082 → -0.047) but didn't flip positive
- **Root cause**: Guardrails too conservative (ρ capped at 25%, directional loss too weak)
- **Verdict**: Stability proven ✅, but need stronger directional pressure

**P6b v2.2 = Controlled Stronger Pressure (❌ FAILED at Epoch 8)**:
- Model: `artifacts/lvm/models/transformer_p6b_v22_20251102_203637/best_model.pt`
- Result: Margin +0.002 at E8 (briefly positive!), but **FAKE WIN** - orthogonal escape
- Val cosine: 0.44 → 0.18 (60% collapse!)
- R@5: 100% → 12% (retrieval broke)
- **Failure mode**: Directional pressure too strong (ρ=0.35), overwhelmed MSE loss
- Model learned to predict vectors FAR from target (negative cosine to prev: -0.086)
- Passed only 1/5 5CAT gates (need 3/5)
- **Verdict**: Proved that training tricks can't overcome backward data bias

**P6b v2.3 = "Goldilocks" Balanced Pressure (✅ IMPLEMENTED, READY)**:
1. **Directional-when-confident gate** (CRITICAL) - Scale loss by cos(pred, target):
   - If cos < 0.30: scale=0 (directional OFF when misaligned)
   - If cos > 0.45: scale=1 (directional FULL when aligned)
   - Prevents orthogonal escape (v2.2's failure mode)
2. **Lower ρ targets** - 0.15 → 0.20 → 0.25 (not 0.35 like v2.2)
3. **Weaker penalties** - Back to v2.1 values (τ=0.10, β=1e-3, κ=1e-4)
4. **Lower λ_max** - 0.018 (not 0.03)
5. **Gentler margins** - 0.02-0.04 (not 0.06-0.07)
6. **All v2.1 guardrails kept** - Stability proven

### Quick Start
```bash
# ⚠️ DO NOT TRAIN ON WIKIPEDIA - Use forward-flow data instead!

# P6b v2.3 with directional-when-confident gate (RECOMMENDED for forward-flow data)
./scripts/train_transformer_p6b_v23.sh

# ❌ NOT: P6b v2.2 (failed with orthogonal escape)
# ❌ NOT: Train on raw Wikipedia (Δ = -0.0696, will fail)
```

### P6b v2.2 Implementation (✅ COMPLETE)

**Core Loss Functions** (`app/lvm/losses_directional.py`):
- `directional_margin_loss_v21()` - Scale-aware loss (α=0.7 mix of gap + ratio)
- `positive_floor_penalty()` - ReLU(τ - cos(pred, next))² with τ=0.12 (STRONGER)
- `norm_regularization()` - (||pred||₂ - 1)² penalty
- `orthogonality_penalty()` - (cos(pred, prev))² penalty (NEW)

**Training Integration** (`app/lvm/train_unified.py`):
- **ρ-controller**: Actively pushes ρ to target (0.15 → 0.25 → 0.35)
- Epoch-gated schedule with higher margins (0.06-0.07)
- Higher λ_max (0.03, was 0.02)
- Stronger pos_floor (τ=0.12, β=2e-3)
- All v2.1 guardrails retained (skip logic, safety caps)
- CLI flag: `--p6b-v22`

**Training Script** (`scripts/train_transformer_p6b_v22.sh`):
- 12 epochs, batch_size=32, lr=5e-4
- Automatic 5CAT validation at end
- Enhanced diagnostics (ρ vs ρ_target)

### Expected P6b v2.2 Results (12 epochs)

**Epoch-by-Epoch Prediction**:
- Epochs 1-2: Margin ≈ -0.04, ρ ≈ 0.15 (baseline)
- Epochs 3-4: Margin ≈ -0.02 to -0.01, ρ ≈ 0.25 (climbing)
- Epochs 5-6: **Margin flips positive** (0.00 → +0.01), ρ ≈ 0.35 🎉
- Epochs 7-9: Margin +0.02 to +0.04, ρ stable at 0.35
- Epochs 10-12: Margin +0.03 to +0.05 (stable positive)

**Final Model Targets**:
- ✅ Margin: +0.03 to +0.05 (POSITIVE!)
- ✅ R@5: ≥ 70% (high accuracy)
- ✅ Val cosine: ≥ 0.48 (good similarity)
- ✅ ρ: 0.35-0.50 (controlled by ρ-controller)
- ✅ Pass 3/5 5CAT gates minimum
- ✅ **Breaks backward curse!**

### P6 Data (Ready for P6b Training)

**P6 NEXT Token Architecture**: Predict target_next instead of target
- Training: 431,895 sequences (`artifacts/lvm/training_sequences_ctx5_p6_next_token.npz`)
- Validation: 18,360 sequences (`artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz`)
- OOD: 9,920 sequences (`artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz`)
- **Identity path removed**: cos(ctx[4], target_next) = 0.395 (vs ~0.8 for regular target)

**P6 Baseline Results** (10 epochs, NO directional loss):
- Model: `artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt`
- Val cosine: 0.511, R@5: 0.700 ✅
- Margin: -0.082 ❌ (proves data has backward bias)
- Use for: Baseline comparison for P6b

**Direction Diagnostics** (`tools/diagnose_p6_direction.py`):
- Forward (ctx[-1] → target_next): 0.3876
- Backward (ctx[-1] → target_prev): 0.4569
- **Δ = -0.0692** (backward is 7% stronger!)

### Current Production Model (Baseline)

**✅ P1 Baseline Transformer** (Deployed on port 9007)
- Model: `artifacts/lvm/models/transformer_baseline_p1/best_model.pt`
- Metrics: val_cos 0.550, margin -0.167 (backward bias)
- Access: http://localhost:9007/chat
- Use for: Baseline comparisons only (DO NOT use for production until P5.1 passes)

### Root Cause Analysis Complete (Nov 2, 2025)

**Problem Identified**: Wikipedia text has **inherent backward temporal structure**
- Forward (ctx[-1] → target_next): 0.3876
- Backward (ctx[-1] → target_prev): 0.4569
- **Δ = -0.0692** (backward is 7% stronger)

**Why Wikipedia is Backward**:
- Articles follow explanatory structure (lead → details)
- Later chunks reference previous concepts more than they preview future concepts
- Example: Chunk N mentions "Einstein" (from chunk 0) more than new concepts

**Proof**: P6 removed identity path (cos(ctx[4], target_next) = 0.395) but margin STILL negative (-0.082)

**Solution**: P6b = P6 Architecture + Directional Margin Loss
- P6 removes shortcuts
- Directional loss explicitly enforces `cos(pred, target_next) > cos(pred, target_prev) + margin`

**Complete Documentation**:
- Root cause paper: `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md`
- P6b v2.1 implementation guide: `artifacts/lvm/P6B_V21_IMPLEMENTATION.md` (500+ lines)
- P6b v1 collapse analysis: `artifacts/lvm/P6B_EPOCH3_COLLAPSE_ANALYSIS.md`
- Session summary: `artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md`
- Diagnostics tool: `tools/diagnose_p6_direction.py`

---

### Training Data & Validation Requirements

**Recommended Training Data**:
```bash
# Training: 438k sequences from articles 1-1499, 2000-3999, 4500-7671
artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz

# Validation: 18k sequences from articles 4000-4499
artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz

# OOD Test: 10k sequences from articles 1500-1999 (truly held-out)
artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz
```

**Key Learnings**:
- ❌ **NEVER train without validating data quality first** - use `diagnose_data_direction.py`
- ❌ **NEVER use `random_split()` for train/val splits** - causes data contamination
- ❌ **NEVER deploy without 5CAT validation** - must pass 3/5 gates minimum
- ✅ **ALWAYS use article-based splits** - no article overlap between train/val/OOD
- ✅ **ALWAYS integrate 5CAT testing during training** - detects backward bias early
- ✅ **ALWAYS verify OOD generalization** - val score alone is not enough
- ✅ **Require minimum coherence ≥ 0.40, signal ≥ +0.08**

**Retraining Commands**:
```bash
# Train with automatic 5CAT validation (RECOMMENDED)
./scripts/train_with_5cat_validation.sh transformer
```

**Old Models** (DO NOT USE):
- `artifacts/lvm/models_340k/*` - Backward bias (trained on low-quality data)
- `artifacts/lvm/models/transformer_directional_v3/` - Collapsed
- `artifacts/lvm/models/transformer_p{2,3,4}_*/` - Failed experiments

**See Full Documentation**:
- OOD Evaluation Fix: `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md`
- Training Session Report: `artifacts/lvm/TRAINING_SESSION_2025_11_01.md`
- Backward Bias Investigation: `artifacts/lvm/BACKWARD_BIAS_ROOT_CAUSE_REPORT.md`

---

## 📊 LVM DATA QUALITY REQUIREMENTS (2025-10-30 - MANDATORY)

**STATUS**: ✅ **ENFORCED** - All training data must pass quality gates

### Mandatory Pre-Training Validation

**BEFORE training ANY LVM model:**
```bash
./.venv/bin/python tools/tests/diagnose_data_direction.py \
  artifacts/lvm/YOUR_TRAINING_DATA.npz --n-samples 5000
```

### Quality Gates (ALL Must Pass)

| Metric | Minimum | Target | What It Measures |
|--------|---------|--------|------------------|
| **Coherence** | ≥ 0.40 | 0.45-0.50 | Adjacent context positions are similar |
| **Temporal Signal** | ≥ +0.08 | +0.10 to +0.15 | pos[4] much closer to target than pos[0] |
| **Temporal Order** | Monotonic | Strictly increasing | pos[0] < pos[1] < ... < pos[4] → target |

**Good data example**: 584k clean (coherence 0.46, signal +0.12, monotonic ✅)
**Bad data example**: 340k old (coherence 0.35, signal +0.01, non-monotonic ❌)

### If Data Fails Diagnostic

**DO NOT PROCEED WITH TRAINING!** Instead:
1. Check sequence creation script and chunk boundaries
2. Verify source vectors have proper article/chunk ordering
3. Regenerate using `tools/create_training_sequences_with_articles.py`
4. Document root cause and update scripts to prevent recurrence

### 5→1 Causal Alignment Test (5CAT)

**AFTER training, before deployment:**
```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/YOUR_MODEL/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

**Passing Criteria** (must pass 3/5 gates minimum):

| Gate | What It Tests | VAL Threshold | OOD Threshold |
|------|---------------|---------------|---------------|
| **A: Offset Sweep** | Predicts NEXT, not previous | Margin ≥ +0.12 | ≥ +0.10 |
| **B: Retrieval Rank** | Finds target in article | R@1≥60%, R@5≥95% | R@1≥55%, R@5≥92% |
| **C: Ablations** | Order matters | Shuffle delta ≤ -0.15 | ≤ -0.15 |
| **D: Rollout** | Multi-step coherence | Avg cos@H=5 ≥ 0.45 | ≥ 0.42 |
| **E: Bins Delta** | Generalization | abs(Val-OOD) ≤ 0.05 | ≤ 0.05 |

**🚨 CRITICAL**: If margin is **NEGATIVE**, model learned backward prediction! DO NOT DEPLOY!

### Pre-Training & Post-Training Checklists

**Before Training**:
- [ ] Run `diagnose_data_direction.py` on training data
- [ ] Verify coherence ≥ 0.40, temporal signal ≥ +0.08, monotonic order
- [ ] Use article-based splits (no article overlap in train/val/OOD)
- [ ] Document data source and creation method

**After Training**:
- [ ] Run full 5CAT test (5000 samples)
- [ ] Verify margin is POSITIVE (+0.10 minimum)
- [ ] Pass at least 3/5 gates
- [ ] Document 5CAT results and compare to baseline

**Only deploy models that pass both data quality and 5CAT validation!**

**RECOMMENDED**: Use integrated training script:
```bash
./scripts/train_with_5cat_validation.sh transformer  # Auto 5CAT every 5 epochs
```

---

## 🚨 CRITICAL RULES FOR DAILY OPERATIONS

1. **ALWAYS use REAL data** - Never use stub/placeholder data. Always use actual datasets from `data/` directory.

2. **🔴 NEVER USE ONTOLOGICAL DATASETS FOR LVM TRAINING** (Added Oct 11, 2025)
   - **Ontologies (WordNet, SWO, GO, DBpedia) are TAXONOMIC, NOT SEQUENTIAL**
   - They teach classification hierarchies ("dog → mammal → animal"), not narrative flow
   - **For LVM training, use ONLY sequential document data:**
     - ✅ Wikipedia articles, textbooks, scientific papers, programming tutorials, stories
     - ❌ NEVER WordNet, SWO/GO, or DBpedia ontology chains
   - **Why**: Autoregressive LVMs predict next vector from context. They need temporal/causal relationships, not IS-A hierarchies.
   - **Validation**: Use `tools/test_sequential_coherence.py` to verify dataset suitability
   - **See**: `docs/LVM_TRAINING_CRITICAL_FACTS.md` for detailed explanation

3. **Ontologies ARE useful for GraphRAG, NOT for LVM training**
   - ✅ Use ontologies for: vecRAG retrieval, knowledge graphs, Neo4j relationships
   - ❌ DO NOT use ontologies for: training autoregressive/generative models

4. **ALWAYS verify dataset_source labels** - Training data must use sequential sources (not `ontology-*`)

5. **ALWAYS call faiss_db.save()** - FAISS vectors must be persisted after ingestion

6. **ALWAYS use REAL LLM** - Never fall back to stub extraction. Use Ollama with Llama 3.1:
   - Install: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull model: `ollama pull llama3.1:8b`
   - Start: `ollama serve` (keep running)
   - Verify: `curl http://localhost:11434/api/tags`
   - See `docs/howto/how_to_access_local_AI.md` for full setup

7. **🔴 ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION** (Added Oct 7, 2025)
   - **Every concept MUST have a unique ID** (UUID/CPE ID) that links:
     - PostgreSQL `cpe_entry` table (concept text, CPESH negatives, metadata)
     - Neo4j `Concept` nodes (graph relationships)
     - FAISS NPZ file (768D/784D vectors at index position)
     - Training data chains (ordered sequences for LVM)
   - **NPZ files MUST include**: `concept_texts`, `cpe_ids`, `vectors` arrays
   - **Why this matters**: Without IDs, cannot correlate data across stores → training/inference impossible!

8. **ALWAYS use REAL embeddings** - Use Vec2Text-Compatible GTR-T5 Encoder:
   - **🚨 CRITICAL**: NEVER use `sentence-transformers` directly for vec2text workflows!
   - **✅ CORRECT**: Use `IsolatedVecTextVectOrchestrator` from `app/vect_text_vect/vec_text_vect_isolated.py`
   - **Why**: sentence-transformers produces INCOMPATIBLE vectors (9.9x worse quality)
   - **Test**: Run `tools/compare_encoders.py` to verify encoder compatibility
   - **See**: `docs/how_to_use_jxe_and_ielab.md` for real examples

9. **Never run training without explicit permission.**

10. **Vec2Text usage**: Follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage.
    - Devices: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
    - Steps: Use `--steps 1` by default; increase only when asked.

11. **CPESH data**: Always generate complete CPESH using LLM, never empty arrays.

12. **🔴 macOS OpenMP Crash Fix** (Added Oct 21, 2025)
    - **Problem**: Duplicate OpenMP libraries (PyTorch + FAISS both load `libomp.dylib`)
    - **Solution**: `export KMP_DUPLICATE_LIB_OK=TRUE` (add to ALL training scripts on macOS)
    - **Applies to**: CPU training only (MPS/GPU doesn't use OpenMP)
    - **See**: `CRASH_ROOT_CAUSE.md` for full diagnostic details

---

## ✅ EXPECTED BEHAVIORS (As-Expected, Not Bugs)

1. **Missing CPESH metadata in Wikipedia ingestion** (Added Oct 18, 2025)
   - **Behavior**: Wikipedia concepts may have empty `probe_question` and `expected_answer` fields
   - **Why**: Wikipedia ingestion currently focuses on vector generation for LVM training
   - **Expected**: Concepts have vectors in `cpe_vectors` table and NPZ files, but may lack CPESH metadata
   - **Usage**: These vectors are sufficient for LVM training (vector-to-vector prediction)
   - **Future**: CPESH backfill can be added later if needed for vecRAG applications
   - **Verification**: Check `cpe_vectors` table has matching records, not `probe_question` fields

---

## 📍 CURRENT STATUS (2025-11-02)

**Production Data**:
- 339,615 Wikipedia concepts (articles 1-3,431) with vectors in PostgreSQL
- Vectors: 663MB NPZ file (`artifacts/wikipedia_500k_corrected_vectors.npz`)
- CPESH metadata: Not populated (expected for LVM training data)
- Ingested: Oct 15-18, 2025

**LVM Models**:
- ✅ **P1 Baseline Transformer**: Deployed on port 9007 (http://localhost:9007/chat)
  - Metrics: val_cos 0.550, margin -0.167 (backward bias)
  - Use for: Stable inference, baseline comparisons only
- ✅ **P6b v2.1**: COMPLETED (margin still negative but improved 43%)
  - Model: `artifacts/lvm/models/transformer_p6b_v21_20251102_182615/best_model.pt`
  - Results: val_cos 0.488, R@5 0.769, margin -0.047 (was -0.082)
  - Verdict: Stability proven ✅, guardrails too conservative ⚠️
- 🚀 **P6b v2.2**: READY TO TRAIN (controlled stronger pressure)
  - Script: `./scripts/train_transformer_p6b_v22.sh`
  - Improvements: ρ-controller (0.35 target), stronger anchors, orthogonality penalty
  - Expected: Margin flip positive at epochs 5-6, final +0.03 to +0.05
- ❌ **All previous approaches (V3, P2-P4, P5.1, P6, P6b v1)**: Failed with backward bias or collapsed
- 🔍 **ROOT CAUSE IDENTIFIED**: Wikipedia data has inherent backward bias (Δ = -0.069)

**Components**:
- Full Pipeline: Text→Vec→LVM→Vec→Text working end-to-end (~10s total, vec2text = 97% bottleneck)
- Vec2Text: Use `IsolatedVecTextVectOrchestrator` with `--vec2text-backend isolated`
- CPESH: Full implementation with real LLM generation (Ollama + Llama 3.1:8b)
- n8n MCP: Configured and tested (`claude mcp list` to verify)

**Recent Updates (Nov 2, 2025)**:
- ✅ P6b v2.1 completed: margin -0.047 (improved 43% but still negative)
- ✅ P6b v2.2 implemented: ρ-controller, stronger anchors, orthogonality penalty
- ✅ All components tested and verified working
- 🚀 **Next Steps**: Train P6b v2.2, verify margin flips positive at epochs 5-6

## 🤖 REAL COMPONENT SETUP

### Local LLM Setup (Ollama + Llama 3.1)
```bash
# Quick setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve &

# Test LLM is working
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'

# Environment variables for LNSP integration
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"
```

### Real Embeddings Setup (Vec2Text-Compatible GTR-T5 768D)
```bash
# 🚨 CRITICAL: DO NOT use sentence-transformers directly!
# ❌ WRONG (produces incompatible vectors):
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('sentence-transformers/gtr-t5-base')  # DON'T DO THIS!

# ✅ CORRECT: Use Vec2Text Orchestrator
python -c "
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
orchestrator = IsolatedVecTextVectOrchestrator()
print('✓ Vec2text-compatible encoder loaded')
"

# Test embedding generation (CORRECT method)
python -c "
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
orchestrator = IsolatedVecTextVectOrchestrator()
vectors = orchestrator.encode_texts(['Hello world'])
print('Generated vector shape:', vectors.shape)
print('✓ Vec2text-compatible vectors (will decode correctly)')
"

# Test compatibility (recommended)
./.venv/bin/python tools/compare_encoders.py
# Expected: CORRECT encoder = 0.89 cosine, WRONG encoder = 0.09 cosine
```

### FastAPI Service Management
```bash
# 🚨 CRITICAL: ALWAYS restart services before ingestion runs

# Start all services (Episode, Semantic, GTR-T5, Ingest)
./scripts/start_all_fastapi_services.sh

# Stop all services (clean shutdown)
./scripts/stop_all_fastapi_services.sh

# Check service health
curl -s http://localhost:8900/health  # Episode Chunker
curl -s http://localhost:8001/health  # Semantic Chunker
curl -s http://localhost:8767/health  # GTR-T5 Embeddings
curl -s http://localhost:8004/health  # Ingest API
```

**Best Practice for Long Ingestion Runs:**
```bash
# Stop old → wait 5s → start fresh → wait 10s → run ingestion
./scripts/stop_all_fastapi_services.sh && sleep 5 && \
./scripts/start_all_fastapi_services.sh && sleep 10 && \
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3432 --limit 3000
```

### macOS OpenMP Fix (CRITICAL for Training)
```bash
# 🚨 ALWAYS add this to training scripts on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Example training script
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
python tools/train_model.py ...
```

**Why**: PyTorch + FAISS both load `libomp.dylib`, macOS kills process without this flag
**When**: ✅ CPU training on macOS | ❌ NOT needed for MPS/GPU or Linux

### Ontology Data Ingestion (No FactoidWiki)
```bash
# CRITICAL: Do NOT use FactoidWiki. Use ontology sources only (SWO/GO/DBpedia/etc.)

# 1) Ensure local LLM is configured
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# 2) Ingest ontologies atomically (PostgreSQL + Neo4j + FAISS)
./scripts/ingest_ontologies.sh

# 3) Verify synchronization
./scripts/verify_data_sync.sh

# 4) (Optional) Add 6-degree shortcuts to Neo4j
./scripts/generate_6deg_shortcuts.sh
```


## 📂 KEY COMMANDS

### n8n Integration
```bash
# Setup n8n MCP server in Claude Code
claude mcp add n8n-local -- npx -y n8n-mcp --n8n-url=http://localhost:5678

# Check MCP connection status
claude mcp list

# Start n8n server
N8N_SECURE_COOKIE=false n8n start

# Import workflows
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json

# Test webhook integration
python3 n8n_workflows/test_webhook_simple.py
python3 n8n_workflows/test_batch_via_webhook.py
```

### Vec2Text Testing
```bash
# Test vec2text encoding/decoding (CORRECT method)
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "What is AI?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Key parameters:
# --vec2text-backend isolated (required)
# --subscribers jxe,ielab (test both decoders)
# --steps 1 (default for speed, use 5 for better quality)
# Environment variables enforce CPU usage and project venv
```

## 🏗️ REPOSITORY POINTERS
- **Core runtime**: `app/`
  - Orchestrators: `app/agents/`
  - Models/training: `app/mamba/`, `app/nemotron_vmmoe/`
  - Vec2Text: `app/vect_text_vect/`
  - Utilities: `app/utils/`
- **CLIs and pipelines**: `app/cli/`, `app/pipeline/` (if present)
- **Tests**: `tests/`
- **Docs**: `docs/how_to_use_jxe_and_ielab.md`

## 🔍 VERIFICATION COMMANDS

### Component Status Check
```bash
# Quick system status check
echo "=== LNSP Component Status ==="
echo "Ollama LLM:  " $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "PostgreSQL:  " $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Neo4j:       " $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "GTR-T5:      " $(python -c "from src.vectorizer import EmbeddingBackend; EmbeddingBackend()" >/dev/null 2>&1 && echo "✓" || echo "✗")

# Verify specific components
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | grep llama3.1  # LLM
psql lnsp -c "SELECT jsonb_array_length(concept_vec) as vector_dims FROM cpe_vectors LIMIT 1;"  # Vectors
```

## 💡 DEVELOPMENT GUIDELINES
- **ALWAYS verify real components before starting work** - Run status check above
- **NO STUB FUNCTIONS** - If LLM/embeddings fail, fix the service, don't fall back to stubs
- Python 3.11+ with venv (`python3 -m venv venv && source venv/bin/activate`)
- Install with `python -m pip install -r requirements.txt`
- Lint with `ruff check app tests scripts`
- Run smoke tests: `pytest tests/lnsp_vec2text_cli_main_test.py -k smoke`
- Keep changes aligned with vec2text isolated backend unless otherwise specified

## 📚 KEY DOCUMENTATION

### 🗺️ Data Architecture & Storage (Start Here!)
- **📍 Database Locations**: `docs/DATABASE_LOCATIONS.md`
  - Complete reference for ALL databases, vector stores, and data locations
  - **ACTIVE status indicators** for every component (✅/⚠️/🗑️)
  - Current data volumes: 80,636 concepts, 500k vectors
  - Environment variables, connection strings, verification commands
  - **Use this to find where data lives and what's currently active**

- **🧠 LVM Data Map**: `docs/LVM_DATA_MAP.md`
  - Comprehensive LVM training data, models, and inference pipeline
  - All 4 trained models (AMN, LSTM⭐, GRU, Transformer) with benchmarks
  - Full text→vec→LVM→vec→text pipeline explanation
  - Performance metrics (0.49-2.68ms LVM, ~10s total with vec2text)
  - **Use this for all LVM training and inference work**

- **🔄 Data Flow Diagram**: `docs/DATA_FLOW_DIAGRAM.md`
  - Visual ASCII diagrams showing complete system architecture
  - Data flow from Wikipedia → PostgreSQL → FAISS → LVM → Inference
  - Latency breakdown by component (vec2text = 97% bottleneck!)
  - Critical data correlations (CPE ID linking)
  - **Use this to understand how data flows through the system**

### Component Setup & Usage
- **LLM setup**: `docs/howto/how_to_access_local_AI.md`
- **Vec2Text usage**: `docs/how_to_use_jxe_and_ielab.md`
- **CPESH generation**: `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`
- **Known-good procedures**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`

### Quick Reference
- **What's currently active?** → `docs/DATABASE_LOCATIONS.md` (Quick Reference Table at top)
- **Which LVM model to use?** → `docs/LVM_DATA_MAP.md` (LSTM recommended for production)
- **How does data flow?** → `docs/DATA_FLOW_DIAGRAM.md` (Visual diagrams)
- **LVM Performance?** → `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` (Detailed benchmarks)