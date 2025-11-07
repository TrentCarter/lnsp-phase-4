# Project Status — lnsp-phase-4 Repository

**Last Updated:** 2025-11-06
**Repository:** lnsp-phase-4

---

## 🚨 IMPORTANT: This Repository Contains TWO Projects

This repository houses **two independent AI projects** that were developed sequentially. Understanding which project is active is critical for navigation.

---

## 📊 Quick Status

| Project | Status | Completion | Last Activity | Current Work? |
|---------|--------|------------|---------------|---------------|
| **vecRAG** | ⏸️ Paused | Production Ready | Nov 4-5, 2025 | ❌ NO |
| **PAS (Agent Swarm)** | ✅ Active | 57% (4/7 phases) | Nov 6, 2025 | ✅ **YES** |

---

## 1️⃣ vecRAG - Vector Retrieval System (PAUSED)

### Overview
A retrieval-augmented generation (RAG) system using vec2text for invertible embeddings.

### Status: ⏸️ **PAUSED** (Production Ready)
- **Performance:** 73.4% Contain@50, 50.2% R@5, 1.33ms P95
- **Data:** 339,615 Wikipedia concepts with vectors
- **Decision:** AR-LVM (autoregressive vector-to-vector) abandoned after proving GTR-T5 lacks temporal directionality
- **Outcome:** Retrieval-only vecRAG works well, no need for vector prediction

### Key Files (vecRAG)
```
CLAUDE.md                                   - Complete vecRAG instructions
docs/RETRIEVAL_OPTIMIZATION_RESULTS.md      - Performance results
docs/how_to_use_jxe_and_ielab.md           - Vec2text usage guide
src/                                        - Source code (retrieval, ingest, etc.)
tools/                                      - Ingestion and evaluation tools
artifacts/lvm/                              - Models and evaluation data
artifacts/wikipedia_500k_corrected_vectors.npz - 663MB vector file
```

### When to Work on vecRAG
- If you need to improve retrieval quality
- If you want to add new data sources
- If you need to tune FAISS parameters
- **Otherwise:** Don't touch this project

### Quick Commands (vecRAG - NOT CURRENT WORK)
```bash
# Start vecRAG services (if needed)
./scripts/start_all_fastapi_services.sh

# Run retrieval evaluation
./.venv/bin/python tools/eval_shard_assist.py

# View vecRAG documentation
cat CLAUDE.md
cat docs/RETRIEVAL_OPTIMIZATION_RESULTS.md
```

---

## 2️⃣ PAS - Polyglot Agent Swarm + HMI (ACTIVE) ⭐

### Overview
A multi-agent coordination system with service discovery, resource management, and real-time HMI dashboard.

### Status: ✅ **ACTIVE** (57% Complete, 4/7 Phases)
- **Services Running:** 8/11 (73%)
- **Tests Passing:** 77/77 (100%)
- **Code Written:** ~5,800 lines
- **Started:** Nov 6, 2025
- **Current Phase:** Phase 4 (Claude Sub-Agents) next

### Architecture (PAS)

**Phase 0: Core Infrastructure** ✅ Complete
- Registry (6121) - Service registration & discovery
- Heartbeat Monitor (6109) - Health checks & alerts

**Phase 1: Management Agents** ✅ Complete
- Resource Manager (6104) - CPU/memory/GPU allocation
- Token Governor (6105) - Context tracking & breach detection

**Phase 2: HMI Dashboard** ✅ Complete
- Event Stream (6102) - WebSocket server for real-time updates
- Flask HMI (6101) - Web dashboard with D3.js tree visualization

**Phase 3: Gateway & Routing** ✅ Complete
- Provider Router (6103) - AI provider selection (cost/latency/balanced)
- Gateway (6120) - Central routing hub with cost tracking

**Phase 4: Claude Sub-Agents** ⏳ Next
- 42 agent definitions (.claude/agents/)
- Agent hierarchy structure
- Registry integration

**Phase 5: Local LLM Services** 🔜 Planned
- llama-3.1-8b, TinyLlama, TLC Classifier

**Phase 6: External API Adapters** 🔜 Planned
- OpenAI, Anthropic, Gemini, Grok adapters

### Key Files (PAS)
```
PROGRESS.md                                 - Overall progress tracker
NEXT_STEPS.md                               - Resume guide for Phase 4
services/                                   - All PAS services (8 services)
contracts/                                  - JSON schemas (6 contracts)
scripts/                                    - Start/stop/test scripts
docs/SESSION_SUMMARY_*_PAS_*.md            - Phase completion summaries
docs/PRDs/PRD_Polyglot_Agent_Swarm.md     - Requirements document
docs/PRDs/PRD_IMPLEMENTATION_PHASES.md     - Implementation plan
```

### Quick Commands (PAS - CURRENT WORK)
```bash
# Start all PAS services (8 services)
./scripts/start_all_pas_services.sh

# Run all PAS tests (77 tests)
./scripts/test_phase0.sh
./scripts/test_phase1.sh
./scripts/test_phase2.sh
./scripts/test_phase3.sh

# Open HMI Dashboard (with cost tracking)
open http://localhost:6101

# Check service status
curl http://localhost:6121/health  # Registry
curl http://localhost:6109/health  # Heartbeat Monitor
curl http://localhost:6104/health  # Resource Manager
curl http://localhost:6105/health  # Token Governor
curl http://localhost:6102/health  # Event Stream
curl http://localhost:6101/health  # Flask HMI
curl http://localhost:6103/health  # Provider Router
curl http://localhost:6120/health  # Gateway

# View cost metrics
curl "http://localhost:6120/metrics?window=minute" | jq .

# View cost receipts
cat artifacts/costs/*.jsonl | jq .

# Stop all PAS services
./scripts/stop_all_pas_services.sh
```

---

## 🎯 What Should I Work On?

### Work on PAS (Polyglot Agent Swarm) ✅
**Current focus:** Phase 4 - Claude Sub-Agents

**Next steps:**
1. Define 42 agent specifications in `.claude/agents/`
2. Create agent hierarchy structure (Tree Supervisor → Coordinators → Workers)
3. Build agent registration script
4. Integration tests for agent discovery
5. Agent invocation tests

### Do NOT Work on vecRAG ❌
**Unless:** You have a specific reason to improve retrieval quality or add new data sources.

**vecRAG is complete and working.** Focus on PAS.

---

## 📂 Directory Structure

```
lnsp-phase-4/
│
├── PROJECT_STATUS.md          ⭐ THIS FILE (project overview)
├── PROGRESS.md                📊 PAS progress tracker
├── NEXT_STEPS.md              📝 Resume guide (PAS Phase 4)
├── CLAUDE.md                  📖 vecRAG instructions (PAUSED project)
│
├── services/                  🎯 PAS services (ACTIVE)
│   ├── registry/
│   ├── heartbeat_monitor/
│   ├── resource_manager/
│   ├── token_governor/
│   ├── event_stream/
│   ├── webui/                 (HMI dashboard)
│   ├── provider_router/       (NEW: Phase 3)
│   └── gateway/               (NEW: Phase 3)
│
├── src/                       📦 vecRAG source code (PAUSED)
│   ├── faiss_index.py
│   ├── vectorizer.py
│   └── ...
│
├── tools/                     🔧 vecRAG tools (PAUSED)
│   ├── eval_shard_assist.py
│   ├── ingest_wikipedia_pipeline.py
│   └── ...
│
├── artifacts/
│   ├── lvm/                   📊 vecRAG models & data (PAUSED)
│   ├── registry/              🗄️ PAS registry database (ACTIVE)
│   ├── resource_manager/      🗄️ PAS resource DB (ACTIVE)
│   ├── token_governor/        🗄️ PAS token DB (ACTIVE)
│   ├── provider_router/       🗄️ PAS provider DB (ACTIVE)
│   └── costs/                 💰 PAS cost receipts (ACTIVE)
│
├── scripts/                   🚀 Start/stop/test scripts
│   ├── start_all_pas_services.sh      (PAS)
│   ├── stop_all_pas_services.sh       (PAS)
│   ├── test_phase*.sh                 (PAS)
│   └── start_all_fastapi_services.sh  (vecRAG)
│
├── contracts/                 📋 JSON schemas (PAS)
│   ├── service_registration.schema.json
│   ├── provider_registration.schema.json
│   └── ...
│
└── docs/                      📚 Documentation
    ├── SESSION_SUMMARY_*_PAS_*.md     (PAS summaries)
    ├── RETRIEVAL_OPTIMIZATION_*.md    (vecRAG results)
    ├── PRDs/                          (Requirements)
    └── ...
```

---

## ⚡ Quick Decision Guide

**If you're asked to:**
- "Improve retrieval" → vecRAG (PAUSED project)
- "Add agents" → PAS (ACTIVE project) ✅
- "Update dashboard" → PAS (ACTIVE project) ✅
- "Optimize FAISS" → vecRAG (PAUSED project)
- "Add cost tracking" → PAS (ACTIVE project) ✅
- "Fix vec2text" → vecRAG (PAUSED project)
- "Implement Phase 4" → PAS (ACTIVE project) ✅

**When in doubt:** Work on **PAS (Polyglot Agent Swarm)**. That's the active project.

---

## 📊 Progress Summary

### vecRAG (PAUSED)
- ✅ Wikipedia ingestion (339,615 concepts)
- ✅ FAISS retrieval (73.4% Contain@50)
- ✅ Shard-assist with ANN tuning
- ✅ Production configuration documented
- ⏸️ AR-LVM abandoned (Δ=0.0004, no temporal signal)
- ⏸️ **Paused at production-ready state**

### PAS (ACTIVE)
- ✅ Phase 0: Core Infrastructure (Registry, Heartbeat Monitor)
- ✅ Phase 1: Management Agents (Resource Manager, Token Governor)
- ✅ Phase 2: HMI Dashboard (Event Stream, Flask Web UI)
- ✅ Phase 3: Gateway & Routing (Provider Router, Gateway with cost tracking)
- ⏳ Phase 4: Claude Sub-Agents (Next)
- 🔜 Phase 5: Local LLM Services
- 🔜 Phase 6: External API Adapters

**Overall:** 57% complete (4/7 phases)

---

## 🔗 Key Documentation Links

### PAS (ACTIVE) - Read These
- `PROGRESS.md` - Progress tracker
- `NEXT_STEPS.md` - Resume guide for Phase 4
- `docs/SESSION_SUMMARY_2025_11_06_PAS_PHASE03_COMPLETE.md` - Latest phase summary
- `docs/PRDs/PRD_Polyglot_Agent_Swarm.md` - Requirements
- `docs/PRDs/PRD_IMPLEMENTATION_PHASES.md` - Implementation plan

### vecRAG (PAUSED) - Reference Only
- `CLAUDE.md` - Complete vecRAG instructions
- `docs/RETRIEVAL_OPTIMIZATION_RESULTS.md` - Performance results
- `docs/how_to_use_jxe_and_ielab.md` - Vec2text guide
- `artifacts/lvm/NARRATIVE_DELTA_TEST_FINAL.md` - Why AR-LVM was abandoned

---

## 🎯 Current Objective

**Build Phase 4 of PAS: Claude Sub-Agents**

1. Define 42 agent specifications
2. Create agent hierarchy (Tree Supervisor → Coordinators → Workers)
3. Registry integration
4. Agent invocation tests

**Goal:** Enable coordinated multi-agent workflows with cost tracking and HMI visibility.

---

**Questions?**
- Working on PAS? → See `PROGRESS.md` and `NEXT_STEPS.md`
- Working on vecRAG? → See `CLAUDE.md` (but ask first - it's paused!)
- Unsure which project? → **Work on PAS** (it's the active one)

**Ready to continue with PAS Phase 4!** 🚀
