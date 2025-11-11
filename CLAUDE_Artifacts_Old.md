# CLAUDE_Artifacts_Old.md

**Historical Reference Archive** - Detailed records from abandoned LVM training experiments

**Purpose**: This file contains detailed implementation guides and failure analyses from the AR-LVM training period (Oct-Nov 2025). These are preserved for historical reference but are NOT part of active operational guidance.

**Note**: For current operational guidance, see CLAUDE.md and LNSP_LONG_TERM_MEMORY.md

---

## 📌 P7 "Directional Ranker" - Full Failure Report (2025-11-04 Late Evening)

**STATUS**: ❌ **P7 BASELINE TRAINING FAILED** - Model learned backward prediction despite all defenses

### Training Results (10 epochs, arXiv data)
- **Training margin**: +0.12 (positive, learning forward ✓)
- **Validation margin**: **-0.067** (NEGATIVE, predicting backward ✗)
- cos(pred, next): 0.271 (low similarity to target)
- cos(pred, prev): 0.338 (HIGH similarity to previous chunk!)
- cos(pred, anchor): 0.430 (moderate context alignment)

### Critical Finding
Train/val mismatch - model learned forward on training but predicts backward on validation!

### Epoch 3 Collapse
- When teacher warmup ended (epoch 3), all metrics dropped 33-39%:
- cos_anchor: 0.588 → 0.391 (predictions drifted from context subspace)
- cos_next: 0.396 → 0.241 (lost alignment with target)
- Raw model predictions (q_raw) lost ~60% of context alignment
- Semantic anchoring (λ=0.8) couldn't prevent drift

### Root Cause Analysis
1. ✅ **Data is forward-biased** (Δ = +6.3% in validation set) - data quality is NOT the issue
2. ❌ **InfoNCE ranking loss** dominated margin loss (w_rank=1.0 > w_margin=0.5)
3. ❌ **Semantic anchoring** created conflicting gradients between raw predictions and anchored output
4. ❌ **Teacher pull** - model collapsed immediately when warmup ended at epoch 3
5. ⚠️ **Train/val distribution mismatch** - positive train margin but negative val margin suggests overfitting

### What ALL Defenses Failed
- InfoNCE ranking loss (supposed to prevent escape)
- Prev-repel margin loss (supposed to enforce forward directionality)
- Semantic anchoring λ=0.8 (supposed to keep predictions in context subspace)
- Directional gating (supposed to filter weak sequences)
- Teacher pull warmup (model collapsed when it ended)

### Files Created
- `app/lvm/losses_ranking.py` (430 lines) - P7 loss functions
- `app/lvm/models_p7_ranker.py` (330 lines) - TransformerP7Ranker + LSTMP7Ranker
- `app/lvm/train_p7_ranker.py` (470 lines, JSON bug fixed)
- `scripts/train_p7_ranker.sh` - Training interface
- `artifacts/lvm/P7_BASELINE_FAILURE_REPORT.md` - Complete analysis
- Model: `artifacts/lvm/models/p7_ranker_c5_m0.07_l0.8_20251104_222516/best_model.pt` (DO NOT USE)

### Next Steps Considered (Never Executed)
1. Investigate train/val mismatch - Check if distributions differ (5 min analysis)
2. Increase margin loss weight - w_margin: 0.5→1.5, w_rank: 1.0→0.5 (~10 hrs CPU)
3. Stronger semantic anchoring - λ: 0.8→0.6, add anchor loss (~10 hrs CPU)
4. Pure margin training - Disable InfoNCE entirely, use only margin + MSE (~10 hrs CPU)
5. **⚠️ Abandon autoregressive LVM** - After P1-P7 all failing

**Decision**: Abandoned after narrative delta test proved fundamental limitation

---

## 🎓 P6b v2.3 "GOLDILOCKS" - Full Implementation Guide (Never Trained)

**STATUS**: 🚀 **READY BUT NEVER TRAINED** - Superseded by abandonment decision

### Design Philosophy
P6b v2.3 attempted to balance between v2.1 (too conservative) and v2.2 (too aggressive) by using directional-when-confident gating.

### Architecture Changes from v2.2

**1. Directional-when-confident gate** (CRITICAL):
```python
# Scale loss by cos(pred, target)
confidence = cos(pred, target)
if confidence < 0.30:
    scale = 0  # directional OFF when misaligned
elif confidence > 0.45:
    scale = 1  # directional FULL when aligned
else:
    scale = (confidence - 0.30) / 0.15  # linear ramp

directional_loss = scale * directional_margin_loss(pred, next, prev)
```

**2. Lower ρ targets**:
- Epochs 1-2: ρ_target = 0.15 (baseline)
- Epochs 3-6: ρ_target = 0.20 (gentle climb)
- Epochs 7-12: ρ_target = 0.25 (not 0.35 like v2.2)

**3. Weaker penalties** (back to v2.1 values):
- τ = 0.10 (positive floor threshold)
- β = 1e-3 (positive floor penalty weight)
- κ = 1e-4 (orthogonality penalty weight)

**4. Lower λ_max**:
- 0.018 (was 0.03 in v2.2)

**5. Gentler margins**:
- Epochs 1-2: margin = 0.02
- Epochs 3-6: margin = 0.03
- Epochs 7-12: margin = 0.04 (not 0.06-0.07 like v2.2)

### Expected Results (12 epochs)

**Epoch-by-Epoch Prediction**:
- Epochs 1-2: Margin ≈ -0.04, ρ ≈ 0.15 (baseline)
- Epochs 3-4: Margin ≈ -0.02 to -0.01, ρ ≈ 0.20 (climbing)
- Epochs 5-6: **Margin flips positive** (0.00 → +0.01), ρ ≈ 0.25
- Epochs 7-9: Margin +0.02 to +0.04, ρ stable at 0.25
- Epochs 10-12: Margin +0.03 to +0.05 (stable positive)

**Final Model Targets**:
- ✅ Margin: +0.03 to +0.05 (POSITIVE!)
- ✅ R@5: ≥ 70% (high accuracy)
- ✅ Val cosine: ≥ 0.48 (good similarity)
- ✅ ρ: 0.25-0.35 (controlled by ρ-controller)
- ✅ Pass 3/5 5CAT gates minimum

### Training Script
```bash
# ⚠️ DO NOT TRAIN ON WIKIPEDIA - Use forward-flow data instead!
./scripts/train_transformer_p6b_v23.sh
```

**Reason Never Trained**: Narrative delta test (Nov 4) proved GTR-T5 lacks temporal signal at fundamental level

---

## 🎓 P6b v2.2 Implementation Details (Failed at Epoch 8)

### Core Loss Functions (`app/lvm/losses_directional.py`)
- `directional_margin_loss_v21()` - Scale-aware loss (α=0.7 mix of gap + ratio)
- `positive_floor_penalty()` - ReLU(τ - cos(pred, next))² with τ=0.12 (STRONGER)
- `norm_regularization()` - (||pred||₂ - 1)² penalty
- `orthogonality_penalty()` - (cos(pred, prev))² penalty (NEW)

### Training Integration (`app/lvm/train_unified.py`)
- **ρ-controller**: Actively pushes ρ to target (0.15 → 0.25 → 0.35)
- Epoch-gated schedule with higher margins (0.06-0.07)
- Higher λ_max (0.03, was 0.02)
- Stronger pos_floor (τ=0.12, β=2e-3)
- All v2.1 guardrails retained (skip logic, safety caps)
- CLI flag: `--p6b-v22`

### Training Script
```bash
./scripts/train_transformer_p6b_v22.sh
# 12 epochs, batch_size=32, lr=5e-4
# Automatic 5CAT validation at end
# Enhanced diagnostics (ρ vs ρ_target)
```

### Failure Mode (Epoch 8)
- Model: `artifacts/lvm/models/transformer_p6b_v22_20251102_203637/best_model.pt`
- Result: Margin +0.002 at E8 (briefly positive!), but **FAKE WIN** - orthogonal escape
- Val cosine: 0.44 → 0.18 (60% collapse!)
- R@5: 100% → 12% (retrieval broke)
- **Failure mode**: Directional pressure too strong (ρ=0.35), overwhelmed MSE loss
- Model learned to predict vectors FAR from target (negative cosine to prev: -0.086)
- Passed only 1/5 5CAT gates (need 3/5)
- **Verdict**: Proved that training tricks can't overcome backward data bias

---

## 🎓 P6b v2.1 "Six-Layer Defense" - Complete Results

### Implementation (✅ COMPLETED)
- Model: `artifacts/lvm/models/transformer_p6b_v21_20251102_182615/best_model.pt`
- Result: R@5 = 77% ✅, margin = **-0.047** ⚠️ (improved but still negative!)
- **Improved margin 43%** (-0.082 → -0.047) but didn't flip positive
- **Root cause**: Guardrails too conservative (ρ capped at 25%, directional loss too weak)
- **Verdict**: Stability proven ✅, but need stronger directional pressure

### Six Defense Layers
1. **Margin loss**: cos(pred, next) - cos(pred, prev) ≥ margin
2. **Positive floor**: cos(pred, next) ≥ τ (prevent negative cosines)
3. **Norm regularization**: ||pred||₂ ≈ 1 (stay on unit hypersphere)
4. **Directional gating**: Skip low-quality sequences (signal < 0.08)
5. **Epoch scheduling**: Gradual margin increase (0.02 → 0.04 → 0.06)
6. **Safety caps**: ρ ≤ 0.25 (prevent over-reliance on directional loss)

### Results
- Val cosine: 0.488 (vs 0.511 for P6 baseline)
- R@5: 0.769 (vs 0.700 for P6)
- Margin: -0.047 (vs -0.082 for P6) - **43% improvement!**
- Training stable, no collapse

### Why Margin Still Negative
- Guardrails too conservative (ρ capped at 25%)
- Directional loss weight (λ_max=0.02) too weak to overcome Wikipedia backward bias
- Safety caps prevented aggressive optimization

---

## 📊 P6 Data Details (Ready for Training, Never Used)

### P6 NEXT Token Architecture
Predict target_next instead of target to remove identity path.

**Data Files**:
- Training: 431,895 sequences (`artifacts/lvm/training_sequences_ctx5_p6_next_token.npz`)
- Validation: 18,360 sequences (`artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz`)
- OOD: 9,920 sequences (`artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz`)
- **Identity path removed**: cos(ctx[4], target_next) = 0.395 (vs ~0.8 for regular target)

### P6 Baseline Results (10 epochs, NO directional loss)
- Model: `artifacts/lvm/models/transformer_p6_20251102_131816/best_model.pt`
- Val cosine: 0.511, R@5: 0.700 ✅
- Margin: -0.082 ❌ (proves data has backward bias)
- Use for: Baseline comparison for P6b

### Direction Diagnostics (`tools/diagnose_p6_direction.py`)
- Forward (ctx[-1] → target_next): 0.3876
- Backward (ctx[-1] → target_prev): 0.4569
- **Δ = -0.0692** (backward is 7% stronger!)

---

## 📊 LVM DATA QUALITY REQUIREMENTS (Historical Reference)

**Note**: These requirements were developed for LVM training, now abandoned. Preserved for potential Q-tower ranker work.

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

### Passing Criteria (must pass 3/5 gates minimum)

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

---

## 📊 Training Data Requirements (Historical)

### Recommended Training Data (Never Used After Abandonment)
```bash
# Training: 438k sequences from articles 1-1499, 2000-3999, 4500-7671
artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz

# Validation: 18k sequences from articles 4000-4499
artifacts/lvm/validation_sequences_ctx5_articles4000-4499.npz

# OOD Test: 10k sequences from articles 1500-1999 (truly held-out)
artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz
```

### Key Learnings
- ❌ **NEVER train without validating data quality first** - use `diagnose_data_direction.py`
- ❌ **NEVER use `random_split()` for train/val splits** - causes data contamination
- ❌ **NEVER deploy without 5CAT validation** - must pass 3/5 gates minimum
- ✅ **ALWAYS use article-based splits** - no article overlap between train/val/OOD
- ✅ **ALWAYS integrate 5CAT testing during training** - detects backward bias early
- ✅ **ALWAYS verify OOD generalization** - val score alone is not enough
- ✅ **Require minimum coherence ≥ 0.40, signal ≥ +0.08**

### Old Models (DO NOT USE)
- `artifacts/lvm/models_340k/*` - Backward bias (trained on low-quality data)
- `artifacts/lvm/models/transformer_directional_v3/` - Collapsed
- `artifacts/lvm/models/transformer_p{2,3,4}_*/` - Failed experiments
- `artifacts/lvm/models/transformer_p6_*/` - Baseline (backward bias)
- `artifacts/lvm/models/transformer_p6b_v21_*/` - Improved but still negative margin
- `artifacts/lvm/models/transformer_p6b_v22_*/` - Orthogonal escape failure
- `artifacts/lvm/models/p7_ranker_*/` - Train/val mismatch failure

---

## 📚 Related Documentation (Historical)

### Full Analysis Documents
- `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` - Decisive test proving abandonment
- `artifacts/lvm/P8_PILOT_FAILURE_REPORT.md` - P8 constrained mixture failure
- `artifacts/lvm/P7_BASELINE_FAILURE_REPORT.md` - P7 ranker failure analysis
- `artifacts/lvm/P6B_V21_IMPLEMENTATION.md` - Six-layer defense implementation (500+ lines)
- `artifacts/lvm/P6B_EPOCH3_COLLAPSE_ANALYSIS.md` - P6b v1 collapse post-mortem
- `docs/WIKIPEDIA_BACKWARD_BIAS_ROOT_CAUSE.md` - Why Wikipedia is backward
- `artifacts/lvm/SESSION_SUMMARY_2025_11_02_BACKWARD_BIAS.md` - Complete session notes
- `artifacts/lvm/OOD_EVALUATION_FIX_COMPLETE_SUMMARY.md` - Data quality improvements
- `artifacts/lvm/TRAINING_SESSION_2025_11_01.md` - Training experiments log
- `artifacts/lvm/BACKWARD_BIAS_ROOT_CAUSE_REPORT.md` - Full investigation

### Tools Developed
- `tools/diagnose_p6_direction.py` - Direction diagnostics
- `tools/narrative_delta_check.py` - Narrative delta validation
- `tools/tests/diagnose_data_direction.py` - Data quality diagnostics
- `tools/tests/test_5to1_alignment.py` - 5CAT validation suite
- `tools/create_training_sequences_with_articles.py` - Proper data generation

---

## 🗂️ CLAUDE.md SECTIONS ARCHIVED (2025-11-11)

**Purpose**: These sections were removed from CLAUDE.md to reduce token usage (~40k → ~25k tokens). All details are preserved here and in dedicated documentation files.

**Date**: November 11, 2025
**Reason**: Token optimization while preserving complete information through doc links

---

### 1. Encoder/Decoder Full Configuration

**Original Location**: CLAUDE.md lines 17-121

**Production Services**: Ports 7001 (Encode) and 7002 (Decode)

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

**Direct Python Usage (IsolatedVecTextVectOrchestrator):**
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

**WRONG: Using port 8767 encoder + port 8766 decoder**
```python
# ❌ DON'T DO THIS - Port 8766 decoder is NOT compatible with port 8767 encoder!
encode_resp = requests.post("http://localhost:8767/embed", json={"texts": [text]})
vector = encode_resp.json()["embeddings"][0]

decode_resp = requests.post("http://localhost:8766/decode", json={"vectors": [vector]})
# Result: Gibberish output with cosine ~0.05 (nearly orthogonal)
```

**Why This Matters:**
- Port 8767 + Port 8766 give **cosine similarity ~0.05** (gibberish)
- Port 7001 + Port 7002 give **80-100% keyword matches** (meaningful output)
- Orchestrator encode + decode gives **meaningful output** with actual keyword matches
- **ONLY use the orchestrator** (ports 7001/7002 or direct Python) - do NOT mix ports 8767/8766

**CPU vs MPS Performance:**
- ✅ **CPU (ports 7001/7002)**: 1,288ms per decode, 0.78 req/sec
- ⚠️ **MPS (ports 7003/7004)**: 3,779ms per decode, 0.26 req/sec
- **Production recommendation**: Use CPU services (7001/7002). Even with 12 CPU cores at 100%, it's still 2.93x faster than MPS.

**Port Reference:**
| Port | Service | Status | Use For |
|------|---------|--------|---------|
| 7001 | Orchestrator Encoder (FastAPI) | ✅ PRODUCTION | **Encoding for full pipeline** |
| 7002 | Orchestrator Decoder (FastAPI) | ✅ PRODUCTION | **Decoding from port 7001** |
| 8767 | GTR-T5 Encoder | ⚠️ Use with caution | Encoding ONLY (standalone, not for decode pipeline) |
| 8766 | Vec2Text Decoder | ❌ INCOMPATIBLE | DO NOT USE with any encoder |
| N/A | IsolatedVecTextVectOrchestrator | ✅ CORRECT | **Direct Python usage** |

**See Full Docs**: `docs/how_to_use_jxe_and_ielab.md`

---

### 2. Production Retrieval Full Configuration

**Original Location**: CLAUDE.md lines 152-188

**STATUS**: ✅ Production Ready - Shard-Assist with ANN Tuning
**Performance**: 73.4% Contain@50, 50.2% R@5, 1.33ms P95

**FAISS Configuration:**
```python
nprobe = 64                # ANN probe count (Pareto optimal)
K_global = 50              # Global IVF candidates
K_local = 20               # Per-article shard candidates
```

**Reranking Pipeline:**
```python
mmr_lambda = 0.7           # MMR diversity (FULL POOL, do NOT reduce!)
w_same_article = 0.05      # Same-article bonus
w_next_gap = 0.12          # Next-chunk gap bonus
tau = 3.0                  # Gap penalty temperature
directional_bonus = 0.03   # Directional alignment bonus
```

**Key Files:**
- Evaluation: `tools/eval_shard_assist.py`
- Article Shards: `artifacts/article_shards.pkl` (3.9GB)
- Production Results: `artifacts/lvm/eval_shard_assist_full_nprobe64.json`

**⚠️ DO NOT:**
- Reduce `mmr_lambda` from 0.7 (hurts R@10 by -10pp)
- Apply MMR to limited pool (use full candidate set)
- Use adaptive-K (doesn't help, adds complexity)
- Enable alignment head by default (hurts containment)

**See Full Docs**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md`

---

### 3. Critical Rules Detailed Explanations

**Original Location**: CLAUDE.md lines 191-230

**Rule 1: ALWAYS use REAL data**
- Never use stub/placeholder data
- Always use actual datasets from `data/` directory
- Verify data exists before starting operations

**Rule 2: ALWAYS call faiss_db.save()**
- FAISS vectors must be persisted after ingestion
- Without save(), vectors are lost on restart
- Verify `.index` file created in `artifacts/`

**Rule 3: ALWAYS use REAL LLM**
- Never fall back to stub extraction
- Use Ollama with Llama 3.1:8b
- Install: `curl -fsSL https://ollama.ai/install.sh | sh`
- Pull model: `ollama pull llama3.1:8b`
- Start: `ollama serve` (keep running)
- Verify: `curl http://localhost:11434/api/tags`
- **See**: `docs/howto/how_to_access_local_AI.md`

**Rule 4: ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION**
- Every concept MUST have a unique ID (UUID/CPE ID) that links:
  - PostgreSQL `cpe_entry` table (concept text, CPESH negatives, metadata)
  - Neo4j `Concept` nodes (graph relationships)
  - FAISS NPZ file (768D/784D vectors at index position)
- NPZ files MUST include: `concept_texts`, `cpe_ids`, `vectors` arrays
- Without IDs, cannot correlate data across stores → retrieval impossible!
- **See**: `docs/DATA_CORRELATION_GUIDE.md`

**Rule 5: ALWAYS use REAL embeddings**
- Use Vec2Text-Compatible GTR-T5 Encoder
- **🚨 CRITICAL**: NEVER use `sentence-transformers` directly for vec2text workflows!
- **✅ CORRECT**: Use `IsolatedVecTextVectOrchestrator` from `app/vect_text_vect/vec_text_vect_isolated.py`
- **Why**: sentence-transformers produces INCOMPATIBLE vectors (9.9x worse quality)
- **Test**: Run `tools/compare_encoders.py` to verify encoder compatibility
- **See**: `docs/how_to_use_jxe_and_ielab.md`

**Rule 6: Vec2Text usage**
- Follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage
- Devices: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
- Steps: Use `--steps 1` by default; increase only when asked.

**Rule 7: CPESH data**
- Always generate complete CPESH using LLM
- Never use empty arrays
- Verify LLM is running before ingestion

**Rule 8: macOS OpenMP Crash Fix**
- **Problem**: Duplicate OpenMP libraries (PyTorch + FAISS both load `libomp.dylib`)
- **Solution**: `export KMP_DUPLICATE_LIB_OK=TRUE` (add to ALL training scripts on macOS)
- **Applies to**: CPU training only (MPS/GPU doesn't use OpenMP)
- **See**: `docs/MACOS_OPENMP_FIX.md`

---

### 4. Detailed Status History

**Original Location**: CLAUDE.md lines 233-267

**Production Data (Oct 2025):**
- 339,615 Wikipedia concepts (articles 1-3,431) with vectors in PostgreSQL
- Vectors: 663MB NPZ file (`artifacts/wikipedia_500k_corrected_vectors.npz`)
- Ingested: Oct 15-18, 2025

**Production Retrieval:**
- ✅ **FAISS vecRAG**: 73.4% Contain@50, 50.2% R@5, 1.33ms P95 (Production ready)
- ✅ **Reranking pipeline**: Shard-assist with ANN tuning (nprobe=64)
- ✅ **Paragraph-only retrieval**: Tested sentence-aware alternatives (P9), no improvement

**Components:**
- Vec2Text: Use `IsolatedVecTextVectOrchestrator` with `--vec2text-backend isolated`
- Encoder/Decoder: Ports 7001/7002 (CPU, 2.93x faster than MPS)
- CPESH: Full implementation with real LLM generation (Ollama + Llama 3.1:8b)
- n8n MCP: Configured and tested (`claude mcp list` to verify)
- **✨ PLMS Tier 1**: Project Lifecycle Management System
- **✨ P0 End-to-End Integration**: Gateway → PAS Root → Aider-LCO → Aider CLI (Nov 10)
- **✨ DirEng + PEX**: Two-tier AI interface architecture (Nov 7)

**Recent Updates (Nov 4-11, 2025):**
- ✅ **AR-LVM abandoned**: Narrative delta test (Δ=0.0004) proved GTR-T5 lacks temporal signal
- ✅ **Wikipedia analysis**: Backward-biased (Δ=-0.0696), still useful for retrieval
- ✅ **P9 sentence retrieval**: Tested, no improvement over paragraph-only
- ✅ **PLMS Tier 1 shipped**: Multi-run support, Bayesian calibration, risk visualization (Nov 6)
- ✅ **Integration gaps closed**: 10 gaps (auth, secrets, sandboxing, etc.) - Nov 7
- ✅ **DirEng designed**: Human-facing interface agent (like Claude Code) - Nov 7
- ✅ **PEX contract**: Project orchestrator with strict safety rules - Nov 7
- ✅ **P0 End-to-End Integration**: Gateway + PAS Root + Aider-LCO complete (Nov 10)
- ✅ **Communication logging**: Flat .txt logs with LLM metadata tracking (Nov 10)
- ✅ **HMI Sequencer enhancements**: Zoom (100x), scrollbars, color scheme (Nov 11)
- ✅ **Slash commands**: Created `/wrap-up` for session documentation (Nov 11)
- 🎯 **Current focus**: Test P0 stack, then start Phase 1 (LightRAG Code Index)
- 🔍 **Optional future work**: Q-tower ranker for retrieved candidates

**See**: `docs/DATABASE_LOCATIONS.md` for current data volumes

---

### 5. Component Setup Full Examples

**Original Location**: CLAUDE.md lines 269-373

**Local LLM Setup (Ollama + Llama 3.1):**
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

**Real Embeddings Setup (Vec2Text-Compatible GTR-T5 768D):**
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

**FastAPI Service Management:**
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

**macOS OpenMP Fix (CRITICAL for Training):**
```bash
# 🚨 ALWAYS add this to training scripts on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Example training script
#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
python tools/train_model.py ...
```

**Ontology Data Ingestion (No FactoidWiki):**
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

**See**:
- `docs/howto/how_to_access_local_AI.md` - LLM setup
- `docs/how_to_use_jxe_and_ielab.md` - Vec2Text usage
- `docs/MACOS_OPENMP_FIX.md` - macOS OpenMP fix

---

### 6. PLMS Full API Documentation

**Original Location**: CLAUDE.md lines 376-490

**Status**: ✅ Shipped (Nov 6, 2025)
**Version**: V1 Tier 1
**PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`

**What is PLMS?**
Production-grade project orchestration system that estimates token costs, duration, and resource allocation for PAS-executed projects. Includes multi-run support (baseline/rehearsal/replay), Bayesian calibration, and risk visualization.

**Quick Start:**
```bash
# 1. Apply database migration
sqlite3 artifacts/registry/registry.db < migrations/2025_11_06_plms_v1_tier1.sql

# 2. Verify migration
sqlite3 artifacts/registry/registry.db ".schema project_runs" | grep run_kind

# 3. Import PLMS modules
./.venv/bin/python -c "from services.plms.api.projects import router; print('✓ PLMS ready')"

# 4. Run test vectors (requires API server running on port 6100)
export PLMS_API_BASE_URL=http://localhost:6100
bash tests/api/plms_test_vectors.sh
```

**Key Features (Tier 1):**
1. **Multi-run Support**: `run_kind` enum (baseline/rehearsal/replay/hotfix)
2. **Idempotent API**: Requires `Idempotency-Key` header for safe retries
3. **Rehearsal Mode**: 1% canary testing before full execution (`?rehearsal_pct=0.01`)
4. **Credible Intervals**: 90% Bayesian CIs for token/duration/cost estimates
5. **Lane-Specific KPIs**: Beyond Echo-Loop (test pass rate, schema diff, BLEU, etc.)
6. **Active Learning**: Lane override feedback (`lane_overrides` table)
7. **Budget Runway**: Time-to-depletion + projected overrun visualization
8. **Risk Heatmap**: Lane × phase risk matrix (MAE, CI width)
9. **Estimation Drift**: Sparkline charts showing MAE trends

**API Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/projects/{id}/start` | POST | Start execution (requires `Idempotency-Key`) |
| `/api/projects/{id}/simulate` | POST | Rehearsal mode with `?rehearsal_pct=0.01` |
| `/api/projects/{id}/metrics` | GET | Get estimates (add `?with_ci=1` for credible intervals) |
| `/api/projects/{id}/lane-overrides` | GET | Active learning feedback for lane classifier |
| `/api/projects/{id}/budget-runway` | GET | Budget depletion time + projected overrun |
| `/api/projects/{id}/risk-heatmap` | GET | Lane × phase risk scores |
| `/api/projects/{id}/estimation-drift` | GET | MAE trend sparklines |

**Files & Locations:**
- Code: `services/plms/api/projects.py`, `services/plms/kpi_validators.py`, `services/plms/calibration.py`
- Database: `migrations/2025_11_06_plms_v1_tier1.sql`, `artifacts/registry/registry.db`
- Documentation: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` (70KB), `docs/HMI_JSON_CONTRACTS_PLMS.md`
- Tests: `tests/api/plms_test_vectors.sh`

**Integration TODOs:**
1. Database Integration: Replace `db_*` stub functions with actual Registry DB queries
2. PAS Integration: Replace `pas_submit_jobcard()` with actual PAS Architect submission
3. Auth Middleware: Replace `get_current_user()` with JWT/session auth
4. Idempotency Cache: Swap `_IDEMP_CACHE` (in-memory) to Redis for production
5. Calibration Webhooks: Wire `update_priors_after_run()` to PAS completion events
6. KPI Validators: Add actual DB queries

**Example Usage:**
```python
import requests

# Start execution (idempotent)
response = requests.post(
    "http://localhost:6100/api/projects/42/start",
    headers={"Idempotency-Key": "unique-uuid-here"},
    json={"run_kind": "baseline"}
)
print(response.json())
# {"run_id": "abc123", "replay_passport": {...}, "status": "submitted"}

# Simulate with 1% rehearsal
response = requests.post(
    "http://localhost:6100/api/projects/42/simulate?rehearsal_pct=0.01"
)
print(response.json())
# {"rehearsal_tokens": 150, "projected_tokens": 15000, "risk_score": 0.12}

# Get estimates with credible intervals
response = requests.get(
    "http://localhost:6100/api/projects/42/metrics?with_ci=1"
)
print(response.json())
# {"tokens_mean": 15000, "tokens_ci_lower": 13200, "tokens_ci_upper": 16800, ...}
```

**See Full Docs**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`

---

### 7. P0 Integration Full Guide

**Original Location**: CLAUDE.md lines 493-574

**Status**: ✅ Production Ready
**Doc**: `docs/P0_END_TO_END_INTEGRATION.md`

**What is P0 Integration?**
The complete production scaffold connecting user requests to filesystem/git operations through a safe, auditable pipeline.

**Key Insight: "Verdict = Aider Re-skinned"**

Verdict (your CLI) wraps Aider (open-source AI pair programmer) with safety guardrails.

```
Verdict CLI (user interface)
  ↓
Gateway (port 6120 - entry point)
  ↓
PAS Root (port 6100 - orchestration, no AI)
  ↓
Aider-LCO RPC (port 6130 - safety wrapper)
  ↓
Aider CLI (external tool - real AI editing)
  ↓
Git/Filesystem (with allowlists enforced)
```

**Quick Start:**
```bash
# 1. Install Aider (one-time)
pipx install aider-chat
export ANTHROPIC_API_KEY=your_key_here  # or OPENAI_API_KEY

# 2. Start all services
bash scripts/run_stack.sh

# Expected output:
# [Aider-LCO] ✓ Started on http://127.0.0.1:6130
# [PAS Root]  ✓ Started on http://127.0.0.1:6100
# [Gateway]   ✓ Started on http://127.0.0.1:6120

# 3. Test via CLI
./bin/verdict health
./bin/verdict send \
  --title "Add docstrings" \
  --goal "Add docstrings to all functions in services/gateway/app.py" \
  --entry-file "services/gateway/app.py"

# 4. Check status
./bin/verdict status --run-id <uuid>

# 5. View artifacts
cat artifacts/runs/<uuid>/aider_stdout.txt
```

**Safety Layers:**
| Layer | What It Does | Example Block |
|-------|--------------|---------------|
| **FS Allowlist** | Only workspace files | ❌ `/etc/passwd`, `~/.ssh/`, `.env` |
| **CMD Allowlist** | Only safe commands | ❌ `rm -rf`, `git push --force` |
| **Env Isolation** | Redact secrets | ❌ `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| **Timeout** | Kill runaway processes | ⏱️ 900s default (15 min) |

**Components Implemented:**
- Gateway: `services/gateway/app.py` (port 6120)
- PAS Root: `services/pas/root/app.py` (port 6100)
- Aider-LCO RPC: `services/tools/aider_rpc/app.py` (port 6130)
- Verdict CLI: `tools/verdict_cli_p0.py`, `bin/verdict`
- Config: `configs/pas/*.yaml` (allowlists)
- Launcher: `scripts/run_stack.sh`

**See Full Docs**: `docs/P0_END_TO_END_INTEGRATION.md`

---

### 8. Communication Logging Full Examples

**Original Location**: CLAUDE.md lines 577-673

**Status**: ✅ Production Ready

Complete parent-child communication logging for PAS with flat `.txt` logs, LLM metadata tracking, and real-time parsing.

**Quick Start:**
```bash
# View all logs for today (colored output)
./tools/parse_comms_log.py

# Filter by run ID
./tools/parse_comms_log.py --run-id abc123-def456

# Filter by LLM model
./tools/parse_comms_log.py --llm claude
./tools/parse_comms_log.py --llm qwen

# Watch logs in real-time (tail -f mode)
./tools/parse_comms_log.py --tail

# Export to JSON
./tools/parse_comms_log.py --format json > logs.json
```

**Log Format:**
```
timestamp|from|to|type|message|llm_model|run_id|status|progress|metadata
```

**Example:**
```txt
2025-11-10T18:31:37.429Z|Gateway|PAS Root|CMD|Submit Prime Directive: Add docstrings|-|test-run-001|-|-|-
2025-11-10T18:31:45.123Z|Aider-LCO|PAS Root|HEARTBEAT|Processing file 3 of 5|ollama/qwen2.5-coder:7b-instruct|test-run-001|running|0.60|%7B%22files_done%22%3A3%7D
2025-11-10T18:32:10.456Z|Aider-LCO|PAS Root|RESPONSE|Completed successfully|ollama/qwen2.5-coder:7b-instruct|test-run-001|completed|1.0|%7B%22rc%22%3A0%7D
```

**Log Locations:**
- Global logs: `artifacts/logs/pas_comms_<date>.txt` (daily rotation)
- Per-run logs: `artifacts/runs/<run-id>/comms.txt`

**Developer Usage:**
```python
from services.common.comms_logger import get_logger

logger = get_logger()

# Log a command
logger.log_cmd(
    from_agent="PAS Root",
    to_agent="Aider-LCO",
    message="Execute Prime Directive",
    llm_model="ollama/qwen2.5-coder:7b-instruct",
    run_id="abc123"
)

# Log a heartbeat
logger.log_heartbeat(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Processing file 3 of 5",
    llm_model="ollama/qwen2.5-coder:7b-instruct",
    run_id="abc123",
    status="running",
    progress=0.6,
    metadata={"files_done": 3}
)
```

**Schema Updates:**
- ✅ `contracts/heartbeat.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `children_agents`
- ✅ `schemas/heartbeat.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`
- ✅ `contracts/status_update.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `progress`
- ✅ `schemas/status_update.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `progress`

**See Full Docs**: `docs/COMMS_LOGGING_GUIDE.md`, `docs/FLAT_LOG_FORMAT.md`

---

### 9. Two-Tier AI Interface Full Details

**Original Location**: CLAUDE.md lines 676-730

**Critical**: You are now part of a **two-tier architecture** for human↔AI interaction.

**Tier 1: DirEng (Director of Engineering AI) - YOUR ROLE**
- **Identity**: Human-facing conversational assistant (like Claude Code)
- **User Interface**: Natural language ("Where is X?", "Implement feature Y")
- **Scope**: Exploration, small edits (1-3 files, <5 min), local operations
- **Tools**: Direct FS/git/shell access (with approval for risky ops)
- **Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (400+ lines)

**When to Handle Directly:**
- "Where is X defined?" → Use LightRAG `rag.where_defined()`
- "Fix typo in file Y" → Apply patch directly
- "Run tests" → Execute `pytest` and show results
- "Show me how Z works" → Read code, explain with snippets

**When to Delegate to PEX (Tier 2):**
- "Implement feature X" (multi-file, multi-step)
- "Estimate how long this will take" (needs PLMS)
- "Run full test suite and fix all errors" (budget tracking)
- User wants rehearsal mode, KPI validation, or budget runway

**Tier 2: PEX (Project Executive) - THE ORCHESTRATOR**
- **Identity**: Project-facing orchestration layer
- **Interface**: Structured API (JSON, not natural language)
- **Scope**: Multi-task projects, budget tracking, KPI validation
- **Tools**: Sandboxed executors, allowlists, PLMS/PAS/Vector-Ops
- **Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md` (204 lines)

**Architecture Diagram:**
```
You (Human)
    ↕ Natural language
DirEng (Tier 1) ← YOU ARE HERE
    ↕ Delegation (when task is large)
PEX (Tier 2)
    ↕ Orchestration
PLMS + PAS + Vector-Ops
```

**Key Documents:**
- DirEng Contract: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` ⭐ READ THIS
- PEX Contract: `docs/contracts/PEX_SYSTEM_PROMPT.md`
- Architecture: `docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md` (500+ lines)
- Integration Plan: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md`

**Implementation Status:**
- ✅ **Contracts**: DirEng + PEX complete (Nov 7)
- ✅ **PAS Stub**: 12 endpoints operational (port 6200)
- ✅ **VP CLI**: 7 commands working (delegates to PAS)
- ⏳ **DirEng REPL**: To be implemented (Phase 3, Weeks 3-4)
- ⏳ **LightRAG**: To be implemented (Phase 1, Weeks 1-2)
- ⏳ **Full PAS**: To be implemented (Phase 4, Weeks 5-8)

---

### 10. Key Commands Full Examples

**Original Location**: CLAUDE.md lines 733-771

**n8n Integration:**
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

**Vec2Text Testing:**
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

**See Full Docs**: `docs/QUICK_COMMANDS.md`

---

### 11. Project Structure and Guidelines

**Original Location**: CLAUDE.md lines 773-840

**Repository Pointers:**
- **Core runtime**: `app/`
  - Orchestrators: `app/agents/`
  - Models/training: `app/mamba/`, `app/nemotron_vmmoe/`
  - Vec2Text: `app/vect_text_vect/`
  - Utilities: `app/utils/`
- **CLIs and pipelines**: `app/cli/`, `app/pipeline/` (if present)
- **Tests**: `tests/`
- **Docs**: `docs/how_to_use_jxe_and_ielab.md`

**Verification Commands:**
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

**Development Guidelines:**
- **ALWAYS verify real components before starting work** - Run status check above
- **NO STUB FUNCTIONS** - If LLM/embeddings fail, fix the service, don't fall back to stubs
- Python 3.11+ with venv (`python3 -m venv venv && source venv/bin/activate`)
- Install with `python -m pip install -r requirements.txt`
- Lint with `ruff check app tests scripts`
- Run smoke tests: `pytest tests/lnsp_vec2text_cli_main_test.py -k smoke`
- Keep changes aligned with vec2text isolated backend unless otherwise specified

**Key Documentation:**
- **📍 Database Locations**: `docs/DATABASE_LOCATIONS.md` - Complete reference for ALL databases, vector stores, and data locations
- **🔄 Data Flow Diagram**: `docs/DATA_FLOW_DIAGRAM.md` - Visual ASCII diagrams showing complete system architecture
- **vecRAG optimization**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md`
- **Known-good procedures**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`
- **LLM setup**: `docs/howto/how_to_access_local_AI.md`
- **Vec2Text usage**: `docs/how_to_use_jxe_and_ielab.md`
- **CPESH generation**: `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`

**Archived (Historical Reference):**
- **LVM Data Map**: `docs/LVM_DATA_MAP.md` (AR-LVM abandoned Nov 2025)
- **LVM Performance**: `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` (Historical)

---

**END OF CLAUDE.md SECTIONS ARCHIVED (2025-11-11)**

_All sections preserved here are now referenced in trimmed CLAUDE.md via doc links._
_Estimated token savings: ~10,500 tokens (40k → ~29k)_

---

**END OF HISTORICAL ARCHIVE**

_This file contains detailed records from Oct-Nov 2025 LVM training experiments._
_All approaches (P1-P8) ultimately failed due to fundamental GTR-T5 limitations._
_Decision made Nov 4, 2025: Pivot to retrieval-only vecRAG._
