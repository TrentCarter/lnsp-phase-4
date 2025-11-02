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

## 📌 ACTIVE CHECKPOINT: Wikipedia Ingestion (2025-10-18 - Updated)

**STATUS**: Ready to continue ingestion with improved checkpoint system

- **Current data**: 339,615 concepts (articles 1-3,431)
- **Next batch**: Articles 3,432+ (checkpoint system now active!)
- **Improvements**: Auto-save every 100 articles, `--resume` flag, graceful shutdown
- **Estimated time**: ~30-40 hours for 3,000 more articles

To continue ingestion with checkpoints:
```bash
# Start fresh batch (saves progress every 100 articles)
LNSP_TMD_MODE=hybrid \
LNSP_LLM_ENDPOINT="http://localhost:11434" \
LNSP_LLM_MODEL="llama3.1:8b" \
./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3432 \
  --limit 3000 \
  > logs/wikipedia_ingestion_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Or resume from checkpoint (if crashed)
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --resume
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

## 🎓 LVM TRAINING & OOD EVALUATION (2025-11-01 - P4 FAILED)

**STATUS**: 🔴 **P4 ROLLOUT FAILED - BACKWARD BIAS UNRESOLVED** (2025-11-01)

### November 1, 2025 Training Session - Complete Results

**12-hour systematic investigation** across 5 approaches (V3, P1, P2, P3, P4). All directional loss approaches failed.

**Session Summary**: `artifacts/lvm/TRAINING_SESSION_2025_11_01.md` (comprehensive documentation)

#### Results Overview

| Approach | Strategy | Val Cosine | Margin (5CAT) | 5CAT Gates | Outcome |
|----------|----------|------------|---------------|------------|---------|
| **V3** | Strong guards (λ=0.01) | 0.354 | -0.132 | 0/5 | ❌ Collapsed at epoch 4 |
| **P1** | Pure MSE baseline | 0.550 | ~0.0* | Unknown | ✅ Stable (needs 5CAT validation) |
| **P2** | Residual architecture | 0.472 | -0.534 | Unknown | ❌ Made copying easier |
| **P3** | Tiny guards (λ=0.002) | 0.526 | -0.064 | Unknown | ⚠️ 51% improvement, still biased |
| **P4** | Rollout + adaptive | 0.540 (epoch 3) → 0.338 (epoch 4) | **-0.149** | **2/5** | ❌ **SAME COLLAPSE AS V3** |

*P1 margin based on training metric, not 5CAT validation

#### Key Learnings

**What Failed** (ALL directional loss approaches):
1. **V3 Strong Guards**: Guards 10x stronger than MSE → catastrophic collapse (val_cos 0.540 → 0.354 at epoch 4)
2. **P2 Residual**: `ŷ = norm(u + α·Δ)` → model learns Δ≈0 to copy last frame (margin -0.534)
3. **P3 Tiny Guards**: Guards too weak (λ=0.002) to overcome MSE patterns (margin -0.064, only 51% improvement)
4. **P4 Rollout + Adaptive Guards**: **SAME collapse as V3** (val_cos 0.540 → 0.338 at epoch 4, margin -0.149)

**Critical Finding** (P4 5CAT Results):
- **Epoch 3 model** (pure MSE, BEFORE rollout/guards): val_cos 0.540 ✅ BUT margin **-0.149** ❌
- **Backward bias exists in BASE MSE training**, NOT caused by directional losses!
- R@1: 1.04% (should be 60%+), R@5: 22.12% (should be 95%+) → retrieval destroyed
- Peak at k=-1 (previous) instead of k=+1 (next) → model predicts backward

**Root Cause Hypothesis**:
- Training data may have inherent backward bias (chunk[i-1] more similar to context than chunk[i+1])
- OR MSE needs more epochs to converge to neutral (P1: 20 epochs → margin 0.0, P4: 3 epochs → margin -0.149)
- OR fundamental issue with sequence generation or evaluation methodology

**Production Status**:
- ✅ **P1 Baseline**: Deployed on port 9007 (http://localhost:9007/chat)
  - Model: `artifacts/lvm/models/transformer_baseline_p1/best_model.pt`
  - Val cosine: 0.550, Margin: ~0.0* (neutral, needs 5CAT validation)
  - Use for: Stable inference, baseline comparisons
  - *Margin based on training metric, not 5CAT
- ❌ **P4 Rollout**: FAILED - Same collapse as V3
  - Model: `artifacts/lvm/models/transformer_p4_rollout/best_model.pt` (epoch 3, for analysis)
  - Val cosine: 0.540 (epoch 3) → 0.338 (epoch 4 collapse)
  - 5CAT: Margin -0.149, R@1 1.04%, 2/5 gates passed
  - **DO NOT USE** - Backward bias unresolved

**Complete Documentation**:
- Session Overview: `artifacts/lvm/TRAINING_SESSION_2025_11_01.md`
- P4 Failure Report: `artifacts/lvm/P4_FAILURE_REPORT.md` ← **READ THIS FIRST**
- P4 Approach (original): `artifacts/lvm/P4_ROLLOUT_APPROACH.md`
- Training Scripts: `scripts/train_transformer_p{1,2,3,4}_*.sh`
- P1 Chat Launcher: `scripts/launch_p1_chat.py`

**CRITICAL Next Steps** (MUST DO before more training):
1. ✅ **Run 5CAT on P1** to establish true baseline (does P1 also have backward bias?)
   ```bash
   ./.venv/bin/python tools/tests/test_5to1_alignment.py \
     --model artifacts/lvm/models/transformer_baseline_p1/best_model.pt \
     --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
     --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
     --articles-npz artifacts/wikipedia_584k_fresh.npz \
     --device mps --max-samples 5000
   ```
2. ✅ **Validate training data** for inherent backward bias:
   ```bash
   ./.venv/bin/python tools/tests/diagnose_data_direction.py \
     artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
     --n-samples 5000
   ```
3. ⚠️ **DO NOT attempt more directional loss approaches** until root cause is understood
4. ⚠️ **Consider fundamentally different architectures** (autoregressive, contrastive, etc.)

---

### Critical Discovery (2025-10-30 Evening) - ONGOING INVESTIGATION

**Issue**: ALL production models (ports 9001-9006) predict the **PREVIOUS** vector instead of **NEXT** vector
- Transformer Optimized: margin **-0.134** (should be +0.12)
- LSTM: margin **-0.149**
- GRU: margin **-0.124**
- Transformer: margin **-0.054**
- AMN: margin **-0.002** (random)

**Root Cause**: Models were trained on low-quality 340k dataset:
- Internal coherence: **0.353** (should be ~0.47)
- Temporal signal: **+0.015** (should be +0.12)
- Nearly flat, incoherent sequences

**Resolution**: Retrain on clean 584k dataset with 5CAT validation

### Retraining Commands

```bash
# Train individual model with 5CAT validation (RECOMMENDED - test first)
./scripts/train_with_5cat_validation.sh transformer

# Or train all 4 models sequentially (~6-8 hours total)
./scripts/retrain_all_production_models.sh
```

**Expected Results with Clean Data**:
- ✅ Positive margins: +0.10 to +0.18
- ✅ Strong rollout: 0.50-0.55
- ✅ Val cosine: 0.56-0.62
- ✅ NO backward bias

### Key Learnings

**Data Quality**:
- ❌ **NEVER train without validating data quality first**
- ✅ **ALWAYS run diagnostic tool on training data**:
  ```bash
  ./.venv/bin/python tools/tests/diagnose_data_direction.py \
    artifacts/lvm/training_sequences_NEW.npz --n-samples 5000
  ```
- ✅ **Require minimum coherence ≥ 0.40, signal ≥ +0.08**

**Training Process**:
- ❌ **NEVER use `random_split()` for train/val splits** - causes data contamination
- ✅ **ALWAYS use article-based splits** - ensures no article appears in both train and val
- ✅ **ALWAYS integrate 5CAT testing during training** - detects backward bias early
- ✅ **ALWAYS verify OOD generalization** - val score alone is not enough

**Model Validation**:
- ❌ **NEVER deploy without 5CAT validation**
- ✅ **ALWAYS run full 5CAT test before production**:
  ```bash
  ./.venv/bin/python tools/tests/test_5to1_alignment.py \
    --model path/to/model.pt \
    --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
    --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
    --articles-npz artifacts/wikipedia_584k_fresh.npz \
    --device mps --max-samples 5000
  ```
- ✅ **Require passing all 5 gates** (or 3/5 minimum)

**Production Models** (2025-11-01 Update):
```bash
# OLD models with backward bias (DO NOT USE)
artifacts/lvm/models_340k/transformer/best_model.pt  ❌ margin -0.054
artifacts/lvm/models_340k/gru/best_model.pt          ❌ margin -0.124
artifacts/lvm/models_340k/lstm/best_model.pt         ❌ margin -0.149
artifacts/lvm/models/transformer_optimized_*/best_model.pt  ❌ margin -0.134
artifacts/lvm/models/amn_790k_production_*/best_model.pt    ❌ random

# V3 directional guards
artifacts/lvm/models/transformer_directional_v3/best_model.pt  ❌ FAILED (collapsed)
  ↳ Result: margin -0.132, val_cos 0.354 (collapsed at epoch 4)
  ↳ Root cause: Guards too strong (λ=0.01), 10x stronger than MSE

# P1 Baseline (CURRENTLY DEPLOYED)
artifacts/lvm/models/transformer_baseline_p1/best_model.pt  ✅ PORT 9007
  ↳ Status: Deployed at http://localhost:9007/chat
  ↳ Metrics: val_cos 0.550, margin 0.0 (neutral, no backward bias)
  ↳ Launcher: ./.venv/bin/python scripts/launch_p1_chat.py
  ↳ Use for: Stable baseline, inference testing

# P2 Residual
artifacts/lvm/models/transformer_residual_p2/best_model.pt  ❌ FAILED (worse copying)
  ↳ Result: margin -0.534, val_cos 0.472
  ↳ Root cause: Residual arch (ŷ=norm(u+α·Δ)) made copying easier (Δ≈0)

# P3 Tiny Guards
artifacts/lvm/models/transformer_p3_tiny_guards/best_model.pt  ⚠️ PARTIAL SUCCESS
  ↳ Result: margin -0.064, val_cos 0.526 (51% improvement over V3)
  ↳ Issue: Guards too weak (λ=0.002) to overcome entrenched patterns

# P4 Rollout + Adaptive Guards (FAILED - 2025-11-01)
artifacts/lvm/models/transformer_p4_rollout/best_model.pt  ❌ FAILED (same collapse as V3)
  ↳ Result: margin -0.149 (5CAT), val_cos 0.540 (epoch 3) → 0.338 (epoch 4)
  ↳ 5CAT: 2/5 gates passed, R@1 1.04%, R@5 22.12%
  ↳ Root cause: Backward bias EXISTS in epoch 3 (pure MSE, BEFORE rollout/guards)
  ↳ Collapse: Same epoch-4 pattern as V3 (rollout+guards too aggressive)
  ↳ DO NOT USE - Preserved for analysis only
```

**See Full Reports**:
- P4 Failure: `artifacts/lvm/P4_FAILURE_REPORT.md`
- Backward Bias Investigation: `artifacts/lvm/BACKWARD_BIAS_ROOT_CAUSE_REPORT.md`

**Training Data** (RECOMMENDED):
```bash
# Training: 438k sequences from articles 1-1499, 2000-3999, 4500-7671
artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz

# Validation: 18k sequences from articles 4000-4499
artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz

# OOD Test: 10k sequences from articles 1500-1999 (truly held-out)
artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz
```

**What Was Fixed**:
1. **Root Cause**: `random_split()` mixed articles across train/val → inflated val scores
2. **Fix**: Article-based splits → no article overlap between train/val/OOD
3. **Result**: OOD now matches Val (proves true generalization)

**Architecture Notes**:
- ✅ AMN: Works perfectly for sequential Wikipedia prediction (OOD=0.5622)
- ❌ AMN: Incompatible with chat "repeat-pad" mode (use LSTM/GRU/Transformer for chat)
- ✅ All architectures benefit from article-based splits

**See**: `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md` for full details

---

## 📊 LVM DATA QUALITY REQUIREMENTS (2025-10-30 - MANDATORY)

**STATUS**: ✅ **ENFORCED** - All training data must pass quality gates

### Mandatory Pre-Training Validation

**BEFORE training ANY LVM model, run the data diagnostic tool:**

```bash
./.venv/bin/python tools/tests/diagnose_data_direction.py \
  artifacts/lvm/YOUR_TRAINING_DATA.npz \
  --n-samples 5000
```

### Quality Gates (ALL Must Pass)

| Metric | Minimum | Target | What It Measures |
|--------|---------|--------|------------------|
| **Coherence** | ≥ 0.40 | 0.45-0.50 | Adjacent context positions are similar |
| **Temporal Signal** | ≥ +0.08 | +0.10 to +0.15 | pos[4] much closer to target than pos[0] |
| **Temporal Order** | Monotonic | Strictly increasing | pos[0] < pos[1] < ... < pos[4] → target |

**Coherence**: Mean cosine similarity between adjacent context positions (e.g., pos[0] ↔ pos[1])
**Temporal Signal**: Difference between last-to-target vs first-to-target similarity
**Temporal Order**: Position-to-target similarity should monotonically increase

### Example Good Data (584k Clean)

```
🔍 Test 1: Position-to-Target Similarity
   pos[0] → target: 0.3399
   pos[1] → target: 0.3515
   pos[2] → target: 0.3649
   pos[3] → target: 0.3869
   pos[4] → target: 0.4569  ← +0.117 improvement ✅

   ✅ CORRECT: Similarity increases toward target

🔍 Test 2: First vs Last Position
   Difference: +0.1171  ← Strong signal ✅

🔍 Test 3: Internal Context Coherence
   Mean coherence: 0.4569  ← Good coherence ✅
```

### Example Bad Data (340k Old)

```
🔍 Test 1: Position-to-Target Similarity
   pos[0] → target: 0.3383
   pos[4] → target: 0.3532  ← Only +0.015 improvement ❌

   ⚠️  WARNING: Non-monotonic pattern

🔍 Test 2: First vs Last Position
   Difference: +0.0148  ← Weak signal ❌

🔍 Test 3: Internal Context Coherence
   Mean coherence: 0.3532  ← Low coherence ❌
```

### If Data Fails Diagnostic

**DO NOT PROCEED WITH TRAINING!** Instead:

1. **Investigate Data Pipeline**
   - Check sequence creation script
   - Verify chunk boundaries don't cross articles inappropriately
   - Ensure vectors aren't shuffled during loading

2. **Check Source Data**
   - Verify source vectors have proper article/chunk ordering
   - Run diagnostic on source NPZ: `diagnose_data_direction.py SOURCE.npz`
   - Look for gaps or discontinuities in article sequences

3. **Regenerate Training Data**
   - Use `tools/create_training_sequences_with_articles.py`
   - Verify parameters (context_len=5, stride=1)
   - Re-run diagnostic on new data

4. **Document Root Cause**
   - Add findings to investigation log
   - Update data creation script if needed
   - Prevent recurrence

### 5→1 Causal Alignment Test (5CAT)

**AFTER training, before deployment, run 5CAT:**

```bash
./.venv/bin/python tools/tests/test_5to1_alignment.py \
  --model artifacts/lvm/models/YOUR_MODEL/best_model.pt \
  --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
  --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
  --articles-npz artifacts/wikipedia_584k_fresh.npz \
  --device mps --max-samples 5000
```

**Passing Criteria** (must pass 3/5 minimum):

| Gate | Metric | VAL Threshold | OOD Threshold | What It Tests |
|------|--------|---------------|---------------|---------------|
| **A: Offset Sweep** | Margin(+1) | ≥ +0.12 | ≥ +0.10 | Predicts NEXT, not previous |
| **B: Retrieval Rank** | R@1 / R@5 / MRR | ≥60% / ≥95% / ≥80% | ≥55% / ≥92% / ≥75% | Finds target in article |
| **C: Ablations** | Shuffle delta | ≤ -0.15 | ≤ -0.15 | Order matters |
| **D: Rollout** | Avg cos@H=5 | ≥ 0.45 | ≥ 0.42 | Multi-step coherence |
| **E: Bins Delta** | abs(Val - OOD) | ≤ 0.05 | ≤ 0.05 | Generalization |

**Critical Alert**: If margin is **NEGATIVE**, model learned backward prediction! DO NOT DEPLOY!

### Training with 5CAT Integration

**RECOMMENDED**: Use integrated training script that runs 5CAT every 5 epochs:

```bash
# Train with automatic 5CAT validation
./scripts/train_with_5cat_validation.sh transformer

# Script will:
# - Test data quality before starting
# - Run 5CAT every 5 epochs
# - Alert if backward bias detected
# - Early stop if margin < -0.10
# - Save best 5CAT model separately
```

### Data Quality Checklist

Before training ANY LVM model:

- [ ] ✅ Run `diagnose_data_direction.py` on training data
- [ ] ✅ Verify coherence ≥ 0.40 (target: 0.45-0.50)
- [ ] ✅ Verify temporal signal ≥ +0.08 (target: +0.10 to +0.15)
- [ ] ✅ Verify monotonic increasing position-to-target similarity
- [ ] ✅ Document data source and creation method
- [ ] ✅ Use article-based splits (no article overlap in train/val/OOD)

After training:

- [ ] ✅ Run full 5CAT test (5000 samples)
- [ ] ✅ Verify margin is POSITIVE (+0.10 minimum)
- [ ] ✅ Pass at least 3/5 gates
- [ ] ✅ Document 5CAT results in model metadata
- [ ] ✅ Compare to baseline models

**Only deploy models that pass both data quality and 5CAT validation!**

---

## 🚨 CRITICAL RULES FOR DAILY OPERATIONS

1. **ALWAYS use REAL data** - Never use stub/placeholder data. Always use actual datasets from `data/` directory.
2. **🔴 CRITICAL: NEVER USE ONTOLOGICAL DATASETS FOR LVM TRAINING** (Added Oct 11, 2025)
   - **Ontologies (WordNet, SWO, GO, DBpedia) are TAXONOMIC, NOT SEQUENTIAL**
   - They teach classification hierarchies ("dog → mammal → animal"), not narrative flow
   - **For LVM training, use ONLY sequential document data:**
     - ✅ **Wikipedia articles** (narrative progression)
     - ✅ **Textbooks** (sequential instruction: "First... → Next... → Finally...")
     - ✅ **Scientific papers** (temporal flow: "Methods → Results → Conclusions")
     - ✅ **Programming tutorials** (step-by-step procedures)
     - ✅ **Stories/narratives** (causal/temporal relationships)
     - ❌ **NEVER WordNet** (taxonomic hierarchies)
     - ❌ **NEVER SWO/GO** (ontological categories)
     - ❌ **NEVER DBpedia ontology chains** (classification structures)
   - **Why this matters**: Autoregressive LVMs predict next vector from context. They need temporal/causal relationships, not IS-A hierarchies.
   - **Validation**: Use `tools/test_sequential_coherence.py` to verify dataset suitability before training
   - **See**: `docs/LVM_TRAINING_CRITICAL_FACTS.md` for detailed explanation
3. **Ontologies ARE useful for GraphRAG, NOT for LVM training**
   - ✅ Use ontologies for: vecRAG retrieval, knowledge graphs, Neo4j relationships
   - ❌ DO NOT use ontologies for: training autoregressive/generative models
4. **ALWAYS verify dataset_source labels** - Training data must use sequential sources (not `ontology-*`)
5. **ALWAYS call faiss_db.save()** - FAISS vectors must be persisted after ingestion (see Oct 4 fix)
6. **ALWAYS use REAL LLM** - Never fall back to stub extraction. Use Ollama with Llama 3.1:
   - Install: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull model: `ollama pull llama3.1:8b`
   - Start: `ollama serve` (keep running)
   - Verify: `curl http://localhost:11434/api/tags`
   - See `docs/howto/how_to_access_local_AI.md` for full setup
6. **🔴 CRITICAL: ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION** (Added Oct 7, 2025)
   - **Every concept MUST have a unique ID** (UUID/CPE ID) that links:
     - PostgreSQL `cpe_entry` table (concept text, CPESH negatives, metadata)
     - Neo4j `Concept` nodes (graph relationships)
     - FAISS NPZ file (768D/784D vectors at index position)
     - Training data chains (ordered sequences for LVM)
   - **NPZ files MUST include**:
     - `concept_texts`: Array of concept strings (for lookup)
     - `cpe_ids`: Array of UUIDs (for database correlation)
     - `vectors`: 768D or 784D arrays (for training/inference)
   - **Why this matters**:
     - vecRAG search: Query → FAISS index → CPE ID → concept text
     - LVM training: Chain concepts → match text → get vector index → training sequences
     - Inference: LVM output vector → FAISS nearest neighbor → CPE ID → final text
   - **Without IDs**: Cannot correlate data across stores → training/inference impossible!
3. **ALWAYS use REAL embeddings** - Use Vec2Text-Compatible GTR-T5 Encoder:
   - **🚨 CRITICAL**: NEVER use `sentence-transformers` directly for vec2text workflows!
   - **✅ CORRECT**: Use `IsolatedVecTextVectOrchestrator` from `app/vect_text_vect/vec_text_vect_isolated.py`
   - **Why**: sentence-transformers produces INCOMPATIBLE vectors (9.9x worse quality - see Oct 16 test)
   - **Proof**: See `docs/how_to_use_jxe_and_ielab.md` (top section, Oct 16 2025) for real examples
   - **Test**: Run `tools/compare_encoders.py` to verify encoder compatibility
   - Generates true 768-dimensional dense vectors that work with vec2text decoders
   - See `models/` directory for cached model files
4. **Never run training without explicit permission.**
5. **Vec2Text usage**: follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage.
6. **Devices**: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
7. **Steps**: Use `--steps 1` for vec2text by default; increase only when asked.
8. **CPESH data**: Always generate complete CPESH (Concept-Probe-Expected-SoftNegatives-HardNegatives) using LLM, never empty arrays.
9. **🔴 CRITICAL: macOS OpenMP Crash Fix** (Added Oct 21, 2025)
   - **Problem**: Duplicate OpenMP libraries (PyTorch + FAISS both load `libomp.dylib`)
   - **Symptom**: "Abort trap: 6" crashes at random batches during training
   - **Root cause**: macOS doesn't allow multiple OpenMP runtimes in same process
   - **Solution**: `export KMP_DUPLICATE_LIB_OK=TRUE` (add to ALL training scripts)
   - **Why this works**: Tells OpenMP to ignore duplicate runtime copies
   - **Is it safe?**: YES for single-process training (our use case)
   - **Applies to**: CPU training only (MPS/GPU doesn't use OpenMP)
   - **See**: `CRASH_ROOT_CAUSE.md` for full diagnostic details
   - **Verification**: Run `bash test_fix.sh` to confirm fix works

<!-- Audio notifications section removed to keep repo guidance focused and neutral. -->

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

## 📍 CURRENT STATUS (2025-11-01 - Updated)
- **Production Data**: 339,615 Wikipedia concepts (articles 1-3,431) with vectors in PostgreSQL
  - Vectors: 663MB NPZ file (`artifacts/wikipedia_500k_corrected_vectors.npz`)
  - CPESH metadata: Not populated (expected - see "Expected Behaviors" above)
  - Ingested: Oct 15-18, 2025 (complete fresh dataset)
- **LVM Models**: P1 Baseline Transformer deployed, ALL directional approaches FAILED
  - ✅ **P1 Baseline**: Deployed on port 9007 (val_cos 0.550, margin ~0.0*, needs 5CAT validation)
  - ❌ **P4 Rollout**: FAILED - Same collapse as V3 (margin -0.149, R@1 1.04%, 2/5 gates)
  - ❌ **V3/P2/P3**: All failed with backward bias (see `P4_FAILURE_REPORT.md`)
  - ❌ **Old Models** (ports 9001-9006): Backward bias, do NOT use
  - *P1 margin from training metric, NOT 5CAT (needs validation)
- **CRITICAL Finding**: Backward bias exists in **PURE MSE** (P4 epoch 3), NOT caused by directional losses
- **Full Pipeline**: Text→Vec→LVM→Vec→Text working end-to-end (~10s total, vec2text = 97% bottleneck)
- **CPESH Integration**: Full CPESH (Concept-Probe-Expected-SoftNegatives-HardNegatives) implemented with real LLM generation
- **Vec2Text**: Use `app/vect_text_vect/vec_text_vect_isolated.py` with `--vec2text-backend isolated`
- **n8n MCP**: Configured and tested. Use `claude mcp list` to verify connection
- **Local LLM**: Ollama + Llama 3.1:8b running for real CPESH generation
- **Recent Updates (Nov 1, 2025 - 12-hour session)**:
  - ✅ 5 training approaches tested (V3, P1, P2, P3, P4)
  - ✅ P1 Baseline deployed on port 9007 (stable, needs 5CAT validation)
  - ❌ P4 Rollout FAILED - Same epoch-4 collapse as V3
  - ✅ 5CAT validation completed on P4 (margin -0.149, R@1 1.04%, 2/5 gates)
  - 🔍 **CRITICAL**: Backward bias found in epoch 3 (pure MSE, BEFORE directional losses)
  - ✅ Comprehensive failure analysis (`artifacts/lvm/P4_FAILURE_REPORT.md`)
  - ✅ Model loader fixed to handle both old and new architectures
  - ✅ 5CAT validation framework proven effective at detecting backward bias
  - ⚠️ **Next**: Validate P1 with 5CAT, diagnose training data for inherent bias
- **Previous Updates (Oct 16, 2025)**:
  - ✅ 4 LVM models trained with MSE loss (Wikipedia 80k sequences)
  - ✅ Comprehensive benchmarking completed (see `docs/LVM_DATA_MAP.md`)
  - ✅ Complete data map documentation created (3 new docs)
  - ✅ Full pipeline validated with text examples (ROUGE/BLEU scoring)
- **Previous Fixes (Oct 4, 2025)**:
  - ✅ Fixed `dataset_source` labeling bug (ontology data now labeled correctly)
  - ✅ Fixed FAISS save() call (NPZ files now created automatically)
  - ✅ Updated validation script (checks content, not just labels)

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

### FastAPI Service Management (Added Oct 18, 2025)
```bash
# 🚨 CRITICAL: ALWAYS restart services before ingestion runs
# Why: Clears memory, resets connections, prevents resource leaks

# Start all required services (Episode, Semantic, GTR-T5, Ingest)
./scripts/start_all_fastapi_services.sh

# Check service health
curl -s http://localhost:8900/health  # Episode Chunker
curl -s http://localhost:8001/health  # Semantic Chunker
curl -s http://localhost:8767/health  # GTR-T5 Embeddings
curl -s http://localhost:8004/health  # Ingest API

# Stop all services (clean shutdown)
./scripts/stop_all_fastapi_services.sh

# Service logs (if needed)
tail -f /tmp/lnsp_api_logs/*.log
```

**Best Practice for Long Ingestion Runs:**
```bash
# 1. Stop old services (clear memory)
./scripts/stop_all_fastapi_services.sh

# 2. Wait 5 seconds for clean shutdown
sleep 5

# 3. Start fresh services
./scripts/start_all_fastapi_services.sh

# 4. Wait 10 seconds for initialization
sleep 10

# 5. Run ingestion
LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py \
  --input data/datasets/wikipedia/wikipedia_500k.jsonl \
  --skip-offset 3432 \
  --limit 3000
```

### macOS OpenMP Fix (CRITICAL for Training)
```bash
# 🚨 ALWAYS add this to training scripts on macOS
# Fixes "Abort trap: 6" crashes caused by PyTorch + FAISS OpenMP conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# Example: Add to top of any training launch script
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
python tools/train_model.py ...

# Verify fix works
bash test_fix.sh
```

**Why needed:**
- PyTorch loads `libomp.dylib` (OpenMP runtime)
- FAISS also loads `libomp.dylib`
- macOS kills process when multiple OpenMP runtimes detected
- Setting `KMP_DUPLICATE_LIB_OK=TRUE` tells OpenMP to allow duplicates

**When needed:**
- ✅ **Always** for CPU training on macOS (especially with FAISS)
- ❌ **NOT needed** for MPS/GPU training (MPS doesn't use OpenMP)
- ❌ **NOT needed** on Linux (handles multiple OpenMP better)

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

### FastAPI Service Management (NEW - 2025-10-18)
```bash
# Start all required services for ingestion pipeline
./scripts/start_all_fastapi_services.sh

# Stop all services (always do this before starting a new ingestion run!)
./scripts/stop_all_fastapi_services.sh

# Check individual service health
curl -s http://localhost:8900/health  # Episode Chunker
curl -s http://localhost:8001/health  # Semantic Chunker
curl -s http://localhost:8767/health  # GTR-T5 Embeddings
curl -s http://localhost:8004/health  # Ingest API

# View service logs
tail -f /tmp/lnsp_api_logs/*.log
```

### n8n Integration Commands (NEW - 2025-09-19)
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

### General Commands
```bash
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "What is AI?" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "One day, a little" \
  --subscribers jxe,ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Individual decoder checks (optional)
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "girl named Lily found" \
  --subscribers ielab \
  --vec2text-backend isolated \
  --output-format json \
  --steps 1

# Key parameters
# --vec2text-backend isolated (required)
# --subscribers jxe,ielab to test both decoders
# --steps 1 for speed (use 5 for better quality when requested)
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

### Check All Real Components Are Working
```bash
# 1. Verify Ollama LLM is running
curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | grep llama3.1

# 2. Verify GTR-T5 embeddings
python -c "from src.vectorizer import EmbeddingBackend; eb = EmbeddingBackend(); print('✓ GTR-T5 embeddings working')"

# 3. Verify CPESH data has real negatives (not empty arrays)
psql lnsp -c "SELECT count(*) as items_with_negatives FROM cpe_entry WHERE jsonb_array_length(soft_negatives) > 0 AND jsonb_array_length(hard_negatives) > 0;"

# 4. Verify real vector dimensions
psql lnsp -c "SELECT jsonb_array_length(concept_vec) as vector_dims FROM cpe_vectors LIMIT 1;"

# 5. Test complete CPESH extraction
python -c "
from src.prompt_extractor import extract_cpe_from_text
import os
os.environ['LNSP_LLM_ENDPOINT'] = 'http://localhost:11434'
os.environ['LNSP_LLM_MODEL'] = 'llama3.1:8b'
result = extract_cpe_from_text('The Eiffel Tower was built in 1889.')
print('✓ LLM extraction working')
print('Soft negatives:', len(result.get('soft_negatives', [])))
print('Hard negatives:', len(result.get('hard_negatives', [])))
"
```

### Component Status Check
```bash
# Complete system check
echo "=== LNSP Real Component Status ==="
echo "1. Ollama LLM:" $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running")
echo "2. PostgreSQL:" $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓ Connected" || echo "✗ Not connected")
echo "3. Neo4j:" $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓ Connected" || echo "✗ Not connected")
echo "4. GTR-T5:" $(python -c "from src.vectorizer import EmbeddingBackend; EmbeddingBackend()" >/dev/null 2>&1 && echo "✓ Available" || echo "✗ Not available")
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