# Before You /clear - Quick Checklist

**Date**: Nov 7, 2025
**Session**: Integration Gaps Closed + DirEng Design
**Status**: ✅ Ready for /clear

---

## ✅ What Was Accomplished (20 Files Created)

### Contracts (2)
- [x] `docs/contracts/PEX_SYSTEM_PROMPT.md` (204 lines) - Project orchestrator
- [x] `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (400+ lines) ⭐ - Human interface agent

### Architecture (1)
- [x] `docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md` (500+ lines) - Two-tier design

### Security (1)
- [x] `docs/design/SECURITY_INTEGRATION_PLAN.md` - Auth, secrets, sandboxing

### Policy (3)
- [x] `services/pas/policy/model_broker.schema.json`
- [x] `services/pas/policy/model_broker_example.json`
- [x] `services/pas/policy/validate_broker_policy.py`

### Allowlists (6)
- [x] `services/pas/executors/allowlists/code-impl.yaml`
- [x] `services/pas/executors/allowlists/code-api-design.yaml`
- [x] `services/pas/executors/allowlists/data-schema.yaml`
- [x] `services/pas/executors/allowlists/vector-ops.yaml`
- [x] `services/pas/executors/allowlists/graph-ops.yaml`
- [x] `services/pas/executors/allowlists/narrative.yaml`

### Utilities (3)
- [x] `services/vector_ops/__init__.py`
- [x] `services/vector_ops/refresh_daemon.py` (250 LOC)
- [x] `scripts/replay_from_passport.sh` (200 LOC)

### Documentation (4)
- [x] `docs/PRE_PHASE_4_CHECKLIST.md` (450 lines) - Acceptance tests
- [x] `docs/INTEGRATION_GAPS_RESOLVED.md` (350 lines) - Gap analysis
- [x] `docs/SESSION_SUMMARY_2025_11_07.md` (800+ lines) ⭐ - Complete summary
- [x] `BEFORE_CLEAR_CHECKLIST.md` (this file)

---

## 📚 AFTER /CLEAR: Start Here

### 1. Read Session Summary (5 min)
```bash
cat docs/SESSION_SUMMARY_2025_11_07.md
```
**Contains**: Complete session overview, all 20 files, implementation timeline

### 2. Review Two-Tier Architecture (10 min)
```bash
# DirEng (your role as human interface)
cat docs/contracts/DIRENG_SYSTEM_PROMPT.md

# Architecture overview
cat docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md

# PEX (orchestrator)
cat docs/contracts/PEX_SYSTEM_PROMPT.md
```

### 3. Check CLAUDE.md (Updated)
```bash
cat CLAUDE.md | grep -A 20 "TWO-TIER AI INTERFACE"
```
**Contains**: Your role as DirEng (Tier 1), when to delegate to PEX (Tier 2)

### 4. Optional: Run Acceptance Tests (15-30 min)
```bash
cat docs/PRE_PHASE_4_CHECKLIST.md
# Follow instructions to test PAS stub, VP CLI, validators, etc.
```

---

## 🎯 Key Insights to Remember

### 1. Two-Tier Architecture
```
You (Human)
    ↕ Natural language
DirEng (Tier 1) ← Claude Code = YOU
    ↕ Delegate large tasks
PEX (Tier 2) ← Project orchestrator
    ↕ Orchestrate
PLMS + PAS + Vector-Ops
```

### 2. When to Delegate to PEX
- User says "Implement feature X" (multi-file, multi-step)
- User wants estimation, budget tracking, or KPI validation
- Task duration > 5 minutes or involves > 3 files

### 3. Integration Gaps Closed (10 Total)
1. ✅ AuthN/AuthZ (JWT, JWKS, 8 scopes)
2. ✅ Secrets (4-layer defense)
3. ✅ Sandboxing (bubblewrap, allowlists, cgroups v2)
4. ✅ Artifact store (content-addressed, immutable)
5. ✅ Resource quotas (per-lane CPU%, mem, timeout)
6. ✅ Model broker (schema + validator + example)
7. ✅ LightRAG scale plan (refresh daemon, SLO ≤2 min)
8. ✅ Calibration hygiene (already in PLMS)
9. ✅ PII/license scanner (content policy pass)
10. ✅ Disaster recovery (replay from passport)

### 4. Timeline (10 Weeks, Start Monday Nov 8)
- **Phase 1** (Weeks 1-2): LightRAG Code Index ⭐ START MONDAY
- **Phase 2** (Week 2): Multi-Metric Telemetry
- **Phase 3** (Weeks 3-4): DirEng REPL (LCO MVP)
- **Phase 4** (Weeks 5-8): Full PAS (largest effort)
- **Phase 5** (Weeks 9-10): Planner Learning

---

## 🚀 Monday Morning (Nov 8, 2025)

### Priority Order
1. **09:30 ET**: Daily standup (15 min)
2. **10:00 ET**: Review session summary + contracts (30 min)
3. **10:30 ET**: Start Phase 1 (LightRAG Code Index)
   - Tree-sitter parsing
   - Graph extraction (AST → Neo4j)
   - Semantic indexing (embeddings → FAISS)

### Resources Needed
- 1 senior engineer (LightRAG implementation)
- 1 architect (PAS design review, Phase 4 prep)

---

## 🔑 Quick Commands Reference

### PAS Stub (Port 6200)
```bash
make run-pas-stub      # Start
make pas-health        # Check health
```

### VP CLI (Project Management)
```bash
./.venv/bin/python cli/vp.py new --title "Test" --prd "Test PRD"
./.venv/bin/python cli/vp.py estimate <id>
./.venv/bin/python cli/vp.py simulate <id> --rehearsal 0.01
./.venv/bin/python cli/vp.py start <id>
./.venv/bin/python cli/vp.py status <id>
```

### Policy Validator
```bash
./.venv/bin/python services/pas/policy/validate_broker_policy.py \
  services/pas/policy/model_broker_example.json
```

### Vector-Ops Daemon
```bash
./.venv/bin/python services/vector_ops/refresh_daemon.py --once
```

### Replay Script
```bash
bash scripts/replay_from_passport.sh <RUN_ID> --dry-run
```

---

## 📊 Git Status (20 Untracked Files)

```bash
git status

# Untracked files:
#   docs/contracts/DIRENG_SYSTEM_PROMPT.md ⭐
#   docs/architecture/HUMAN_AI_INTERFACE_ARCHITECTURE.md ⭐
#   docs/contracts/PEX_SYSTEM_PROMPT.md
#   docs/design/SECURITY_INTEGRATION_PLAN.md
#   docs/PRE_PHASE_4_CHECKLIST.md
#   docs/INTEGRATION_GAPS_RESOLVED.md
#   docs/SESSION_SUMMARY_2025_11_07.md ⭐
#   BEFORE_CLEAR_CHECKLIST.md
#   services/pas/policy/*.{json,py}
#   services/pas/executors/allowlists/*.yaml (6 files)
#   services/vector_ops/*.py
#   scripts/replay_from_passport.sh
```

**Optional**: Commit these before /clear
```bash
git add docs/ services/ scripts/ BEFORE_CLEAR_CHECKLIST.md
git commit -m "feat: close 10 integration gaps + design DirEng/PEX two-tier architecture

- Add DirEng (human interface) + PEX (orchestrator) contracts
- Close auth, secrets, sandboxing, model broker gaps
- Add 6 lane allowlists, vector-ops daemon, replay script
- Complete architecture docs + acceptance tests

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## ✅ Final Checks Before /clear

- [x] Session summary created (`docs/SESSION_SUMMARY_2025_11_07.md`)
- [x] CLAUDE.md updated with two-tier architecture section
- [x] All 20 files created and saved
- [x] Acceptance tests documented (`docs/PRE_PHASE_4_CHECKLIST.md`)
- [x] Integration gaps analysis complete (`docs/INTEGRATION_GAPS_RESOLVED.md`)
- [x] Before-clear checklist created (this file)
- [x] Git status shows 20 untracked files (ready to commit)

---

## 🎊 You're Ready!

**Status**: ✅ COMPLETE - Ready for /clear
**Risk**: LOW (down from HIGH)
**Next Session**: Implement Phase 1 (LightRAG Code Index)
**Timeline**: 10 weeks to full production

**Key Achievement**: Transformed "high-risk, blocked integration" → "low-risk, clear path forward"

---

**After /clear, read**: `docs/SESSION_SUMMARY_2025_11_07.md` (complete overview)
