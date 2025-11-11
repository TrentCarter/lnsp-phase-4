# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 🚨 CRITICAL: READ LONG-TERM MEMORY FIRST

**Before doing ANYTHING, read [LNSP_LONG_TERM_MEMORY.md](LNSP_LONG_TERM_MEMORY.md)**

That file contains the cardinal rules that must NEVER be violated:
1. Data Synchronization is Sacred (PostgreSQL + Neo4j + FAISS must stay synchronized)
2. NO FactoidWiki Data - Ontologies ONLY
3. Complete Data Pipeline: CPESH + TMD + Graph (for retrieval quality)
4. Six Degrees of Separation + Shortcuts (0.5-3% shortcut edges)

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

## 📌 AUTOREGRESSIVE LVM ABANDONED (2025-11-04)

**STATUS**: 🔴 **AR-LVM OFFICIALLY ABANDONED** - Pivot to retrieval-only vecRAG

**Why**: After 8 failed training attempts (P1-P8) and decisive narrative delta test (Δ=0.0004, essentially zero), proven that GTR-T5 embeddings encode **semantic similarity**, not **temporal directionality**. Problem is fundamental to embedding space geometry.

**Narrative Delta Test** (DECISIVE):
- Tested 1,287 sequences from classic narrative stories with strong forward plots
- Result: mean_delta = 0.0004 (100x below threshold)
- Even narrative fiction shows ZERO forward temporal signal
- **Conclusion**: No amount of training tricks can overcome this limitation

**What to Keep**:
- ✅ FAISS retrieval (73.4% Contain@50, 50.2% R@5 - production ready)
- ✅ Reranking pipeline (shard-assist, MMR, directional bonuses)
- ✅ DIRECT baseline (no LVM, just retrieval)

**What to Archive**:
- ❌ All AR-LVM models (P1-P8) → See `CLAUDE_Artifacts_Old.md` for details
- ❌ LVM training infrastructure
- ❌ Vector-to-vector prediction components

**Optional Future Work**: Q-tower ranker (rank retrieved candidates, not predict next vector)

**See**: `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` for complete analysis

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

## 🚨 CRITICAL RULES FOR DAILY OPERATIONS

1. **ALWAYS use REAL data** - Never use stub/placeholder data. Always use actual datasets from `data/` directory.

2. **ALWAYS call faiss_db.save()** - FAISS vectors must be persisted after ingestion

3. **ALWAYS use REAL LLM** - Never fall back to stub extraction. Use Ollama with Llama 3.1:
   - Install: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Pull model: `ollama pull llama3.1:8b`
   - Start: `ollama serve` (keep running)
   - Verify: `curl http://localhost:11434/api/tags`
   - See `docs/howto/how_to_access_local_AI.md` for full setup

4. **🔴 ALL DATA MUST HAVE UNIQUE IDS FOR CORRELATION** (Added Oct 7, 2025)
   - **Every concept MUST have a unique ID** (UUID/CPE ID) that links:
     - PostgreSQL `cpe_entry` table (concept text, CPESH negatives, metadata)
     - Neo4j `Concept` nodes (graph relationships)
     - FAISS NPZ file (768D/784D vectors at index position)
   - **NPZ files MUST include**: `concept_texts`, `cpe_ids`, `vectors` arrays
   - **Why this matters**: Without IDs, cannot correlate data across stores → retrieval impossible!

5. **ALWAYS use REAL embeddings** - Use Vec2Text-Compatible GTR-T5 Encoder:
   - **🚨 CRITICAL**: NEVER use `sentence-transformers` directly for vec2text workflows!
   - **✅ CORRECT**: Use `IsolatedVecTextVectOrchestrator` from `app/vect_text_vect/vec_text_vect_isolated.py`
   - **Why**: sentence-transformers produces INCOMPATIBLE vectors (9.9x worse quality)
   - **Test**: Run `tools/compare_encoders.py` to verify encoder compatibility
   - **See**: `docs/how_to_use_jxe_and_ielab.md` for real examples

6. **Vec2Text usage**: Follow `docs/how_to_use_jxe_and_ielab.md` for correct JXE/IELab usage.
   - Devices: JXE can use MPS or CPU; IELab is CPU-only. GTR-T5 can use MPS or CPU.
   - Steps: Use `--steps 1` by default; increase only when asked.

7. **CPESH data**: Always generate complete CPESH using LLM, never empty arrays.

8. **🔴 macOS OpenMP Crash Fix** (Added Oct 21, 2025)
    - **Problem**: Duplicate OpenMP libraries (PyTorch + FAISS both load `libomp.dylib`)
    - **Solution**: `export KMP_DUPLICATE_LIB_OK=TRUE` (add to ALL training scripts on macOS)
    - **Applies to**: CPU training only (MPS/GPU doesn't use OpenMP)
    - **See**: `CRASH_ROOT_CAUSE.md` for full diagnostic details

---

## 📍 CURRENT STATUS (2025-11-10)

**Production Data**:
- 339,615 Wikipedia concepts (articles 1-3,431) with vectors in PostgreSQL
- Vectors: 663MB NPZ file (`artifacts/wikipedia_500k_corrected_vectors.npz`)
- Ingested: Oct 15-18, 2025

**Production Retrieval**:
- ✅ **FAISS vecRAG**: 73.4% Contain@50, 50.2% R@5, 1.33ms P95 (Production ready)
- ✅ **Reranking pipeline**: Shard-assist with ANN tuning (nprobe=64)
- ✅ **Paragraph-only retrieval**: Tested sentence-aware alternatives (P9), no improvement

**Components**:
- Vec2Text: Use `IsolatedVecTextVectOrchestrator` with `--vec2text-backend isolated`
- Encoder/Decoder: Ports 7001/7002 (CPU, 2.93x faster than MPS)
- CPESH: Full implementation with real LLM generation (Ollama + Llama 3.1:8b)
- n8n MCP: Configured and tested (`claude mcp list` to verify)
- **✨ PLMS Tier 1**: Project Lifecycle Management System (see below)
- **✨ P0 End-to-End Integration**: Gateway → PAS Root → Aider-LCO → Aider CLI (Nov 10) ⭐ NEW
- **✨ DirEng + PEX**: Two-tier AI interface architecture (Nov 7)

**Recent Updates (Nov 4-10, 2025)**:
- ✅ **AR-LVM abandoned**: Narrative delta test (Δ=0.0004) proved GTR-T5 lacks temporal signal
- ✅ **Wikipedia analysis**: Backward-biased (Δ=-0.0696), still useful for retrieval
- ✅ **P9 sentence retrieval**: Tested, no improvement over paragraph-only
- ✅ **PLMS Tier 1 shipped**: Multi-run support, Bayesian calibration, risk visualization (Nov 6)
- ✅ **Integration gaps closed**: 10 gaps (auth, secrets, sandboxing, etc.) - Nov 7
- ✅ **DirEng designed**: Human-facing interface agent (like Claude Code) - Nov 7
- ✅ **PEX contract**: Project orchestrator with strict safety rules - Nov 7
- ✅ **P0 End-to-End Integration**: Gateway + PAS Root + Aider-LCO complete (Nov 10) ⭐ NEW
- ✅ **Communication logging**: Flat .txt logs with LLM metadata tracking (Nov 10) ⭐ NEW
- 🎯 **Current focus**: Test P0 stack, then start Phase 1 (LightRAG Code Index)
- 🔍 **Optional future work**: Q-tower ranker for retrieved candidates

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


## 🚀 PLMS (PROJECT LIFECYCLE MANAGEMENT SYSTEM) - Tier 1

**Status**: ✅ Shipped (Nov 6, 2025)
**Version**: V1 Tier 1
**PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`

### What is PLMS?

Production-grade project orchestration system that estimates token costs, duration, and resource allocation for PAS-executed projects. Includes multi-run support (baseline/rehearsal/replay), Bayesian calibration, and risk visualization.

### Quick Start

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

### Key Features (Tier 1)

1. **Multi-run Support**: `run_kind` enum (baseline/rehearsal/replay/hotfix)
2. **Idempotent API**: Requires `Idempotency-Key` header for safe retries
3. **Rehearsal Mode**: 1% canary testing before full execution (`?rehearsal_pct=0.01`)
4. **Credible Intervals**: 90% Bayesian CIs for token/duration/cost estimates
5. **Lane-Specific KPIs**: Beyond Echo-Loop (test pass rate, schema diff, BLEU, etc.)
6. **Active Learning**: Lane override feedback (`lane_overrides` table)
7. **Budget Runway**: Time-to-depletion + projected overrun visualization
8. **Risk Heatmap**: Lane × phase risk matrix (MAE, CI width)
9. **Estimation Drift**: Sparkline charts showing MAE trends

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/projects/{id}/start` | POST | Start execution (requires `Idempotency-Key`) |
| `/api/projects/{id}/simulate` | POST | Rehearsal mode with `?rehearsal_pct=0.01` |
| `/api/projects/{id}/metrics` | GET | Get estimates (add `?with_ci=1` for credible intervals) |
| `/api/projects/{id}/lane-overrides` | GET | Active learning feedback for lane classifier |
| `/api/projects/{id}/budget-runway` | GET | Budget depletion time + projected overrun |
| `/api/projects/{id}/risk-heatmap` | GET | Lane × phase risk scores |
| `/api/projects/{id}/estimation-drift` | GET | MAE trend sparklines |

### Files & Locations

**Code**:
- `services/plms/api/projects.py` - FastAPI endpoints (7 routes)
- `services/plms/kpi_validators.py` - Lane-specific quality gates (7 validators)
- `services/plms/calibration.py` - Bayesian learning hooks

**Database**:
- `migrations/2025_11_06_plms_v1_tier1.sql` - Schema migration (SQLite + PostgreSQL)
- `artifacts/registry/registry.db` - SQLite database (apply migration here)

**Documentation**:
- `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` - Complete PRD (70KB)
- `docs/HMI_JSON_CONTRACTS_PLMS.md` - Frontend integration guide
- `tests/api/plms_test_vectors.sh` - Executable test script (10 test cases)

### Integration TODOs

The current implementation has intentional stubs that need wiring:

1. **Database Integration**: Replace `db_*` stub functions in `projects.py` with actual Registry DB queries
2. **PAS Integration**: Replace `pas_submit_jobcard()` with actual PAS Architect submission
3. **Auth Middleware**: Replace `get_current_user()` with JWT/session auth
4. **Idempotency Cache**: Swap `_IDEMP_CACHE` (in-memory) to Redis for production
5. **Calibration Webhooks**: Wire `update_priors_after_run()` to PAS completion events
6. **KPI Validators**: Add actual DB queries (replace `get_table_schema()`, etc.)

### Example Usage

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

### HMI Integration

See `docs/HMI_JSON_CONTRACTS_PLMS.md` for complete frontend integration guide:
- Budget Runway Gauge (WebSocket/SSE streaming)
- Risk Heatmap (lane × phase matrix)
- Estimation Drift Sparklines (Chart.js)
- Error handling + graceful fallbacks

---

## 🚀 P0 END-TO-END INTEGRATION (2025-11-10)

**Status**: ✅ Production Ready
**Doc**: `docs/P0_END_TO_END_INTEGRATION.md`

### What is P0 Integration?

The complete production scaffold connecting user requests to filesystem/git operations through a safe, auditable pipeline.

### Key Insight: "Verdict = Aider Re-skinned"

**Question**: *"What is the path for AI to get filesystem/tool access?"*

**Answer**: Verdict (your CLI) wraps Aider (open-source AI pair programmer) with safety guardrails.

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

### Quick Start

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

### Safety Layers

| Layer | What It Does | Example Block |
|-------|--------------|---------------|
| **FS Allowlist** | Only workspace files | ❌ `/etc/passwd`, `~/.ssh/`, `.env` |
| **CMD Allowlist** | Only safe commands | ❌ `rm -rf`, `git push --force` |
| **Env Isolation** | Redact secrets | ❌ `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| **Timeout** | Kill runaway processes | ⏱️ 900s default (15 min) |

### Components Implemented

- **Gateway** (`services/gateway/app.py`): Port 6120
- **PAS Root** (`services/pas/root/app.py`): Port 6100
- **Aider-LCO RPC** (`services/tools/aider_rpc/app.py`): Port 6130
- **Verdict CLI** (`tools/verdict_cli_p0.py`, `bin/verdict`)
- **Config** (`configs/pas/*.yaml`): Allowlists (FS + CMD)
- **Launcher** (`scripts/run_stack.sh`): One-command startup

### Key Files

- **Architecture**: `docs/P0_END_TO_END_INTEGRATION.md` (500+ lines)
- **Tool Access Path**: `docs/OPTIONS_SENDING_PRIME_DIRECTIVES.md` (Q0 section)
- **Configs**: `configs/pas/fs_allowlist.yaml`, `configs/pas/cmd_allowlist.yaml`

---

## 📝 COMMUNICATION LOGGING SYSTEM (2025-11-10)

**Status**: ✅ Production Ready

Complete parent-child communication logging for PAS with flat `.txt` logs, LLM metadata tracking, and real-time parsing.

### Quick Start

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

### Log Format

```
timestamp|from|to|type|message|llm_model|run_id|status|progress|metadata
```

**Example:**
```txt
2025-11-10T18:31:37.429Z|Gateway|PAS Root|CMD|Submit Prime Directive: Add docstrings|-|test-run-001|-|-|-
2025-11-10T18:31:45.123Z|Aider-LCO|PAS Root|HEARTBEAT|Processing file 3 of 5|ollama/qwen2.5-coder:7b-instruct|test-run-001|running|0.60|%7B%22files_done%22%3A3%7D
2025-11-10T18:32:10.456Z|Aider-LCO|PAS Root|RESPONSE|Completed successfully|ollama/qwen2.5-coder:7b-instruct|test-run-001|completed|1.0|%7B%22rc%22%3A0%7D
```

### Log Locations

- **Global logs**: `artifacts/logs/pas_comms_<date>.txt` (daily rotation)
- **Per-run logs**: `artifacts/runs/<run-id>/comms.txt`

### Key Files

- **Logger**: `services/common/comms_logger.py`
- **Parser**: `tools/parse_comms_log.py`
- **Guide**: `docs/COMMS_LOGGING_GUIDE.md`
- **Format Spec**: `docs/FLAT_LOG_FORMAT.md`

### Developer Usage

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

### Schema Updates

Updated schemas to include LLM metadata:
- ✅ `contracts/heartbeat.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `children_agents`
- ✅ `schemas/heartbeat.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`
- ✅ `contracts/status_update.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `progress`
- ✅ `schemas/status_update.schema.json` - Added `llm_model`, `llm_provider`, `parent_agent`, `progress`

### Integration Status

- ✅ **Aider-LCO RPC**: Logs all commands, status updates, responses with LLM model info
- ✅ **PAS Root**: Logs Prime Directive submission, delegation to Aider-LCO, completion
- ✅ **Real-time parser**: Colored text output + JSON export + filtering + tailing
- ⏳ **HMI visualization**: To be integrated (Phase 2)

**See**: `docs/COMMS_LOGGING_GUIDE.md` for complete documentation

---

## 🤖 TWO-TIER AI INTERFACE (2025-11-07)

**Critical**: You are now part of a **two-tier architecture** for human↔AI interaction.

### Tier 1: DirEng (Director of Engineering AI) - YOUR ROLE
- **Identity**: Human-facing conversational assistant (like Claude Code)
- **User Interface**: Natural language ("Where is X?", "Implement feature Y")
- **Scope**: Exploration, small edits (1-3 files, <5 min), local operations
- **Tools**: Direct FS/git/shell access (with approval for risky ops)
- **Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (400+ lines)

**When to Handle Directly**:
- "Where is X defined?" → Use LightRAG `rag.where_defined()`
- "Fix typo in file Y" → Apply patch directly
- "Run tests" → Execute `pytest` and show results
- "Show me how Z works" → Read code, explain with snippets

**When to Delegate to PEX** (Tier 2):
- "Implement feature X" (multi-file, multi-step)
- "Estimate how long this will take" (needs PLMS)
- "Run full test suite and fix all errors" (budget tracking)
- User wants rehearsal mode, KPI validation, or budget runway

### Tier 2: PEX (Project Executive) - THE ORCHESTRATOR
- **Identity**: Project-facing orchestration layer
- **Interface**: Structured API (JSON, not natural language)
- **Scope**: Multi-task projects, budget tracking, KPI validation
- **Tools**: Sandboxed executors, allowlists, PLMS/PAS/Vector-Ops
- **Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md` (204 lines)

### Architecture Diagram
```
You (Human)
    ↕ Natural language
DirEng (Tier 1) ← YOU ARE HERE
    ↕ Delegation (when task is large)
PEX (Tier 2)
    ↕ Orchestration
PLMS + PAS + Vector-Ops
```

### Key Documents
- **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` ⭐ READ THIS
- **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md`
- **Architecture**: `docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md` (500+ lines)
- **Integration Plan**: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md`

### Implementation Status
- ✅ **Contracts**: DirEng + PEX complete (Nov 7)
- ✅ **PAS Stub**: 12 endpoints operational (port 6200)
- ✅ **VP CLI**: 7 commands working (delegates to PAS)
- ⏳ **DirEng REPL**: To be implemented (Phase 3, Weeks 3-4)
- ⏳ **LightRAG**: To be implemented (Phase 1, Weeks 1-2)
- ⏳ **Full PAS**: To be implemented (Phase 4, Weeks 5-8)

---

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
  - Current data volumes: 339,615 concepts, 663MB vectors
  - Environment variables, connection strings, verification commands
  - **Use this to find where data lives and what's currently active**

- **🔄 Data Flow Diagram**: `docs/DATA_FLOW_DIAGRAM.md`
  - Visual ASCII diagrams showing complete system architecture
  - Data flow from Wikipedia → PostgreSQL → FAISS → Retrieval
  - Critical data correlations (CPE ID linking)
  - **Use this to understand how data flows through the system**

### Retrieval Configuration
- **vecRAG optimization**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md`
- **Known-good procedures**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`

### Component Setup & Usage
- **LLM setup**: `docs/howto/how_to_access_local_AI.md`
- **Vec2Text usage**: `docs/how_to_use_jxe_and_ielab.md`
- **CPESH generation**: `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`

### Quick Reference
- **What's currently active?** → `docs/DATABASE_LOCATIONS.md` (Quick Reference Table)
- **How does data flow?** → `docs/DATA_FLOW_DIAGRAM.md` (Visual diagrams)
- **vecRAG performance?** → `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md` (73.4% Contain@50)

### Archived (Historical Reference)
- **LVM Data Map**: `docs/LVM_DATA_MAP.md` (AR-LVM abandoned Nov 2025)
- **LVM Performance**: `artifacts/lvm/COMPREHENSIVE_LEADERBOARD.md` (Historical)