# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## 🚀 SESSION STARTUP

**IMPORTANT**: When starting a new conversation, ALWAYS run `/restore` to load the last session context.
This helps you quickly understand what was accomplished and what needs work.

- Use `/restore` for concise summary (recommended)
- Use `/restore --git` for detailed git status

---

## 🚨 CRITICAL: READ LONG-TERM MEMORY FIRST

**Before doing ANYTHING, read [LNSP_LONG_TERM_MEMORY.md](LNSP_LONG_TERM_MEMORY.md)**

That file contains the cardinal rules that must NEVER be violated:
1. Data Synchronization is Sacred (PostgreSQL + Neo4j + FAISS must stay synchronized)
2. NO FactoidWiki Data - Ontologies ONLY
3. Complete Data Pipeline: CPESH + TMD + Graph (for retrieval quality)
4. Six Degrees of Separation + Shortcuts (0.5-3% shortcut edges)

---

## 🔴 CRITICAL: ENCODER/DECODER

**Production**: Ports 7001 (encode) + 7002 (decode) - CPU is 2.93x faster than MPS
**Usage**: `IsolatedVecTextVectOrchestrator` from `app/vect_text_vect/vec_text_vect_isolated.py`
**⚠️ NEVER**: Use `sentence-transformers` directly (9.9x worse quality)

**Start Services:**
```bash
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app --host 127.0.0.1 --port 7001 &
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app --host 127.0.0.1 --port 7002 &
```

**See**: `docs/how_to_use_jxe_and_ielab.md` | `CLAUDE_Artifacts_Old.md` (Section 1)

---

## 📌 AR-LVM ABANDONED

**Status**: 🔴 Abandoned (Nov 4, 2025) - Pivot to retrieval-only vecRAG

After 8 failed training attempts (P1-P8) and decisive narrative delta test (Δ=0.0004), proven that GTR-T5 embeddings encode semantic similarity, not temporal directionality.

**See**: `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` | `CLAUDE_Artifacts_Old.md` (all P1-P8 details)

---

## 🎯 PRODUCTION RETRIEVAL

**Performance**: 73.4% Contain@50, 50.2% R@5, 1.33ms P95 (Production Ready)
**Config**: Shard-Assist with ANN tuning (nprobe=64)

**See**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md` | `CLAUDE_Artifacts_Old.md` (Section 2)

---

## 🚨 CRITICAL RULES FOR DAILY OPERATIONS

1. **ALWAYS use REAL data** - Never use stub/placeholder data
2. **ALWAYS call faiss_db.save()** - FAISS vectors must be persisted
3. **ALWAYS use REAL LLM** - Ollama + Llama 3.1:8b (`docs/howto/how_to_access_local_AI.md`)
4. **ALL DATA MUST HAVE UNIQUE IDS** - CPE IDs link PostgreSQL + Neo4j + FAISS (`docs/DATA_CORRELATION_GUIDE.md`)
5. **ALWAYS use REAL embeddings** - Vec2Text-compatible GTR-T5 (`docs/how_to_use_jxe_and_ielab.md`)
6. **Vec2Text usage** - Follow `docs/how_to_use_jxe_and_ielab.md` (devices, steps)
7. **CPESH data** - Always generate complete CPESH using LLM
8. **macOS OpenMP fix** - `export KMP_DUPLICATE_LIB_OK=TRUE` for CPU training (`docs/MACOS_OPENMP_FIX.md`)

**See**: `CLAUDE_Artifacts_Old.md` (Section 3) for detailed explanations

---

## 📍 CURRENT STATUS (2025-11-11)

**Focus**: Test P0 stack, then start Phase 1 (LightRAG Code Index)

**Production Data**: 339,615 Wikipedia concepts | 663MB NPZ | `docs/DATABASE_LOCATIONS.md`

**Recent Milestones** (Last 2 Weeks):
- ✅ **PLMS Tier 1**: Multi-run support, Bayesian calibration (Nov 6) → `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`
- ✅ **P0 Integration**: Gateway → PAS Root → Aider-LCO complete (Nov 10) → `docs/P0_END_TO_END_INTEGRATION.md`
- ✅ **Communication Logging**: Flat .txt logs with LLM metadata (Nov 10) → `docs/COMMS_LOGGING_GUIDE.md`
- ✅ **HMI Sequencer**: 100x zoom, scrollbars, color scheme (Nov 11)
- ✅ **Slash Commands**: `/wrap-up` and `/restore` for session management (Nov 11)
- ✅ **CLAUDE.md Optimization**: 72.9% token reduction (3,894 → 1,056 words) with zero information loss (Nov 11)
- ✅ **Slash Command Enhancement**: `/wrap-up` now auto-commits and pushes session documentation (Nov 11)
- ✅ **Task Intake System**: `/pas-task` conversational interface for submitting tasks to P0 stack (Nov 11) → `.claude/commands/pas-task.md`

**See**: `CLAUDE_Artifacts_Old.md` (Section 4) for complete history

---

## 🚀 KEY SYSTEMS (Production Ready)

### PLMS (Project Lifecycle Management)
**Status**: ✅ Shipped (Nov 6, 2025) | **Port**: 6100
**What**: Token cost estimation, multi-run support, Bayesian calibration, risk visualization
**See**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` | `CLAUDE_Artifacts_Old.md` (Section 6)

### P0 End-to-End Integration
**Status**: ✅ Production Ready (Nov 10, 2025) | **Ports**: 6120 (Gateway), 6100 (PAS), 6130 (Aider-LCO)
**What**: Verdict CLI → Gateway → PAS Root → Aider-LCO → Aider CLI (filesystem/git with safety)
**Quick Start**: `bash scripts/run_stack.sh` then `./bin/verdict health`
**See**: `docs/P0_END_TO_END_INTEGRATION.md` | `CLAUDE_Artifacts_Old.md` (Section 7)

### Communication Logging
**Status**: ✅ Production Ready (Nov 10, 2025)
**What**: Parent-child logging with LLM metadata tracking
**Quick Start**: `./tools/parse_comms_log.py` (view logs) | `./tools/parse_comms_log.py --tail` (real-time)
**See**: `docs/COMMS_LOGGING_GUIDE.md` | `CLAUDE_Artifacts_Old.md` (Section 8)

### DirEng (Your Role - Two-Tier AI)
**Status**: ✅ Active | **Role**: Tier 1 - Human-facing conversational assistant
**Scope**: Exploration, small edits (1-3 files, <5 min), local operations
**Delegate to PEX when**: Multi-file projects, budget tracking, KPI validation needed
**See**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` | `CLAUDE_Artifacts_Old.md` (Section 9)

### Task Intake System (`/pas-task`)
**Status**: ✅ Production Ready (Nov 11, 2025) | **Type**: Slash Command
**What**: Conversational interface for submitting tasks to P0 stack - DirEng acts as requirements analyst, gathers structured information, formats Prime Directive, submits via Verdict CLI, tracks status
**Quick Start**: `/pas-task` (interactive mode) or `./bin/verdict send` (direct CLI)
**See**: `.claude/commands/pas-task.md`

---

## 📂 QUICK COMMANDS

**Full Command Reference**: `docs/QUICK_COMMANDS.md` | `CLAUDE_Artifacts_Old.md` (Section 10)

**All ports and service mappings**: `docs/SERVICE_PORTS.md` - Complete port mapping for Multi-Tier PAS architecture

```bash
# Service Management
./scripts/start_all_fastapi_services.sh  # Start all services
./scripts/stop_all_fastapi_services.sh   # Stop all services

# LLM (Ollama)
ollama serve &                           # Start Ollama
export LNSP_LLM_ENDPOINT="http://localhost:11434"
export LNSP_LLM_MODEL="llama3.1:8b"

# Vec2Text Testing
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "Test" --subscribers jxe,ielab --vec2text-backend isolated \
  --output-format json --steps 1

# P0 Stack
bash scripts/run_stack.sh                # Start Gateway + PAS + Aider-LCO
./bin/verdict health                     # Check status
./bin/verdict send --title "Task" --goal "Description" --entry-file "file.py"

# Task Intake (Conversational)
# Use /pas-task slash command - DirEng will guide you through structured questions

# Database Check
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;"  # PostgreSQL
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN COUNT(c)"  # Neo4j

# System Status
echo "Ollama: " $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "PostgreSQL: " $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Neo4j: " $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
```

---

## 📚 KEY DOCUMENTATION

**Start Here**:
- **📍 Database Locations**: `docs/DATABASE_LOCATIONS.md` - All databases, vectors, data locations
- **🔄 Data Flow**: `docs/DATA_FLOW_DIAGRAM.md` - System architecture diagrams
- **🎯 vecRAG Config**: `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md` - Production configuration

**Component Setup**:
- **LLM Setup**: `docs/howto/how_to_access_local_AI.md`
- **Vec2Text Usage**: `docs/how_to_use_jxe_and_ielab.md`
- **Data Correlation**: `docs/DATA_CORRELATION_GUIDE.md` (Rule 4: Unique IDs)
- **macOS OpenMP Fix**: `docs/MACOS_OPENMP_FIX.md` (Rule 8: CPU training)
- **Quick Commands**: `docs/QUICK_COMMANDS.md` (Consolidated reference)

**System Integration**:
- **PLMS**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md` (70KB complete PRD)
- **P0 Integration**: `docs/P0_END_TO_END_INTEGRATION.md` (500+ lines)
- **Communication Logging**: `docs/COMMS_LOGGING_GUIDE.md`
- **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (400+ lines)
- **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md` (204 lines)

**Data Ingestion**:
- **Known-Good Procedures**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`
- **CPESH Generation**: `docs/design_documents/prompt_template_lightRAG_TMD_CPE.md`

**Archive (Historical)**:
- **LVM Training Experiments**: `CLAUDE_Artifacts_Old.md` (P1-P8, Oct-Nov 2025)
- **CLAUDE.md Archived Sections**: `CLAUDE_Artifacts_Old.md` (Sections 1-11, full details)

---

## 🔍 VERIFICATION COMMANDS

```bash
# Quick system status check
echo "=== LNSP Component Status ==="
echo "Ollama LLM:  " $(curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "PostgreSQL:  " $(psql lnsp -c "SELECT 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Neo4j:       " $(cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Encoder:     " $(curl -s http://localhost:7001/health >/dev/null 2>&1 && echo "✓" || echo "✗")
echo "Decoder:     " $(curl -s http://localhost:7002/health >/dev/null 2>&1 && echo "✓" || echo "✗")
```

**See**: `docs/DATABASE_LOCATIONS.md` for detailed verification commands

---

## 💡 DEVELOPMENT GUIDELINES

- **ALWAYS verify real components before starting work** (run status check above)
- **NO STUB FUNCTIONS** - Fix services, don't fall back to stubs
- Python 3.11+ with venv (`.venv/bin/python`)
- Install: `python -m pip install -r requirements.txt`
- Lint: `ruff check app tests scripts`
- Tests: `LNSP_TEST_MODE=1 ./.venv/bin/pytest tests -m "not heavy" -v`

**See**: `CLAUDE_Artifacts_Old.md` (Section 11) for complete guidelines

---

**Note**: For detailed examples, code blocks, and complete configurations, see:
- `CLAUDE_Artifacts_Old.md` (Sections 1-11) - All archived details
- Linked documentation files above - Component-specific guides
