# PRD: Manager & Programmer FastAPI Service Upgrade

**Date:** 2025-11-12
**Status:** Active Development
**Priority:** P0 (Critical Architecture Fix)
**Estimated Effort:** 2-3 days

---

## Executive Summary

Upgrade Managers (Tier 4) and Programmers (Tier 5) from lightweight metadata entities to full FastAPI services with LLM capabilities, matching the architecture of Directors (Tier 3). This enables true parallelization, LLM intelligence at all tiers, and makes the WebUI LLM model selection functional.

**Current State:** Managers are metadata-only entities, single shared Aider RPC service (port 6130)
**Target State:** Multiple Manager FastAPI services (ports 6141-6150), multiple Programmer FastAPI services (ports 6151-6199)
**Key Benefit:** 49 parallel Programmers instead of 1 bottlenecked RPC service

---

## Problem Statement

### Design Drift Identified (2025-11-12)

The current implementation has drifted from design goals:

1. **Managers are metadata-only** (not FastAPI services like Directors)
   - No HTTP endpoints
   - No LLM capabilities
   - Communication via file-based queues + heartbeat
   - Cannot intelligently break down tasks

2. **Single shared Programmer service** (Aider-LCO on port 6130)
   - Bottleneck: Only 1 code execution at a time
   - No parallelization
   - No LLM switching (always uses Qwen 2.5 Coder 7B)

3. **WebUI Settings Are Decorative**
   - LLM Model Selection page has Manager/Programmer dropdowns
   - These dropdowns don't work - no LLM configuration exists
   - User expectation mismatch

### Root Cause

P0 scaffold took a shortcut to get something working quickly. This was pragmatic for proving the concept, but now we need to implement the full architecture.

---

## Goals

### Primary Goals

1. **Create Manager FastAPI Services** (7 instances, ports 6141-6147)
   - Each Manager is a full HTTP service like Directors
   - LLM-powered task breakdown (Gemini 2.5 Flash by default)
   - Independent operation, parallel execution
   - Report to parent Directors via HTTP

2. **Create Programmer FastAPI Services** (10 initial instances, ports 6151-6160)
   - Each Programmer is a full HTTP service
   - Wraps Aider CLI with LLM configuration
   - Multiple LLM options (Qwen, Claude, GPT-4, DeepSeek)
   - Parallel code execution (10 simultaneous jobs)

3. **Make WebUI LLM Selection Functional**
   - Manager Primary/Backup model dropdowns work
   - Programmer Primary/Backup model dropdowns work
   - Settings persist to configs

4. **Enable True Parallelization**
   - Multiple Managers work simultaneously (up to 7 parallel)
   - Multiple Programmers work simultaneously (up to 49 parallel)
   - Director can delegate to multiple Managers at once

### Success Metrics

- ✅ All Managers are HTTP services on ports 6141-6147
- ✅ All Programmers are HTTP services on ports 6151-6160 (initial 10)
- ✅ E2E test shows parallel execution (>1 Programmer working simultaneously)
- ✅ WebUI LLM dropdowns persist and configure services correctly
- ✅ Performance: 5x speedup on multi-file tasks (serial → parallel)

---

## Architecture

### Port Allocation

**Managers (Tier 4):** Ports 6141-6150 (10 slots)
```
6141 - Manager-Code-01    (Code lane, primary)
6142 - Manager-Code-02    (Code lane, secondary)
6143 - Manager-Code-03    (Code lane, tertiary)
6144 - Manager-Models-01  (Models lane)
6145 - Manager-Data-01    (Data lane)
6146 - Manager-DevSecOps-01 (DevSecOps lane)
6147 - Manager-Docs-01    (Docs lane)
6148-6150 - Reserved for future expansion
```

**Programmers (Tier 5):** Ports 6151-6199 (49 slots)
```
6151-6155 - Programmer-Qwen-001 through 005 (Qwen 2.5 Coder 7B, Ollama)
6156-6157 - Programmer-Claude-001, 002 (Claude Sonnet 4, Anthropic API)
6158      - Programmer-GPT-001 (GPT-4, OpenAI API)
6159-6160 - Programmer-DeepSeek-001, 002 (DeepSeek Coder V3, Ollama)
6161-6199 - Reserved for future expansion (39 slots)
```

### Service Architecture

Each Manager/Programmer follows the same pattern as Directors:

```
services/pas/manager_{lane}_{num}/
├── app.py              # FastAPI service
├── config.yaml         # LLM config, port, metadata
├── decomposer.py       # (Managers only) Task breakdown
└── executor.py         # (Programmers only) Aider CLI wrapper
```

### Communication Flow

**Before (P0 Scaffold):**
```
Director (6111) → ManagerFactory (metadata) → Aider RPC (6130) → Code
                   ↓ (file queue)              ↑ (single bottleneck)
```

**After (Full Architecture):**
```
Director (6111) → Manager-Code-01 (6141) → Programmer-Qwen-001 (6151) → Code
                ↘ Manager-Code-02 (6142) → Programmer-Qwen-002 (6152) → Code
                 ↘ Manager-Code-03 (6143) → Programmer-Qwen-003 (6153) → Code
                   ↓ (HTTP)                   ↑ (parallel execution)
```

---

## Implementation Plan

### Phase 1: Manager FastAPI Services (Day 1)

**1.1 Create Manager Service Template**
- Copy Director structure to `services/pas/manager_code_01/`
- Update service name, port (6141), tier (4)
- Keep decomposer logic (already exists in Director decomposer.py)
- Add HTTP endpoints: `/health`, `/submit`, `/status/{job_id}`

**1.2 Create Manager Startup Script**
```bash
scripts/start_all_managers.sh
```
- Start 7 Manager services (ports 6141-6147)
- Load LLM config from configs/pas/manager_*.yaml
- Register with Service Registry (6121)
- Register with Heartbeat Monitor (6109)

**1.3 Update Director Delegation**
- Modify `services/pas/director_code/app.py:delegate_to_managers()`
- Change from ManagerFactory → HTTP POST to Manager services
- Load balance across Manager-Code-01, 02, 03 (round-robin)
- Monitor via HTTP GET `/status/{job_id}`

**Files to Modify:**
- NEW: `services/pas/manager_code_01/app.py` (template)
- NEW: `configs/pas/manager_code_01.yaml`
- NEW: `scripts/start_all_managers.sh`
- MODIFY: `services/pas/director_code/app.py` (HTTP delegation)

### Phase 2: Programmer FastAPI Services (Day 2)

**2.1 Create Programmer Service Template**
- Copy Aider-LCO structure to `services/tools/programmer_qwen_001/`
- Update service name, port (6151), tier (5)
- Keep Aider CLI wrapper logic
- Add LLM configuration loading
- Add HTTP endpoints: `/health`, `/execute`, `/status/{task_id}`

**2.2 Create Programmer Startup Script**
```bash
scripts/start_all_programmers.sh
```
- Start 10 Programmer services (ports 6151-6160)
- Each loads its own LLM config (Qwen, Claude, GPT, DeepSeek)
- Register with Service Registry (6121)
- Register with Heartbeat Monitor (6109)

**2.3 Create Programmer Pool Manager**
- NEW: `services/common/programmer_pool.py`
- Track available Programmers (idle, busy, offline)
- Load balancing: Select least-busy Programmer
- Health monitoring: Mark offline Programmers

**2.4 Update Manager Delegation**
- Modify Manager services to use Programmer Pool
- Change from Aider RPC → HTTP POST to Programmer services
- Parallel execution: Submit to multiple Programmers simultaneously
- Monitor via HTTP GET `/status/{task_id}`

**Files to Create:**
- NEW: `services/tools/programmer_qwen_001/app.py` (template)
- NEW: `configs/pas/programmer_qwen_001.yaml`
- NEW: `scripts/start_all_programmers.sh`
- NEW: `services/common/programmer_pool.py`
- MODIFY: Manager services (HTTP delegation)

### Phase 3: WebUI Integration (Day 3)

**3.1 Add LLM Model Persistence**
- MODIFY: `services/webui/templates/base.html`
- Make Manager LLM dropdowns functional
- Make Programmer LLM dropdowns functional
- Persist to `configs/pas/manager_*.yaml` and `configs/pas/programmer_*.yaml`

**3.2 Update Model Pool Integration**
- Connect WebUI dropdowns to Model Pool (8050)
- Display available models from Model Pool
- Show model status (active, idle, offline)

**3.3 Add Manager/Programmer Monitoring**
- Add Manager status cards to HMI Dashboard
- Add Programmer status cards to HMI Dashboard
- Show parallel execution in Tree View
- Update Sequencer to show Manager/Programmer tiers

**Files to Modify:**
- MODIFY: `services/webui/templates/base.html` (LLM dropdowns)
- MODIFY: `services/webui/app.py` (API endpoints)

### Phase 4: Testing & Validation (Day 3)

**4.1 Update E2E Test**
- MODIFY: `test_manager_e2e.py`
- Test parallel execution (submit 3 jobs, expect 3 Managers + 3 Programmers)
- Verify LLM switching (Qwen vs Claude vs GPT)
- Measure speedup (serial vs parallel)

**4.2 Performance Testing**
- Test with 10 simultaneous jobs (should use 10 Programmers)
- Test with 20 simultaneous jobs (should queue, use 10 Programmers)
- Measure P95 latency
- Measure throughput (jobs/minute)

**4.3 Integration Testing**
- Full stack: Gateway → PAS → Architect → Director → Manager → Programmer
- Test all 5 lanes (Code, Models, Data, DevSecOps, Docs)
- Test LLM fallback (primary fails → backup)
- Test HHMRS (timeout → restart → escalate)

---

## Configuration Files

### Manager Config Template

`configs/pas/manager_code_01.yaml`:
```yaml
service:
  name: "Manager-Code-01"
  port: 6141
  host: "127.0.0.1"
  tier: 4
  lane: "Code"

agent_metadata:
  agent_id: "Mgr-Code-01"
  role: "manager"
  parent: "Dir-Code"
  grandparent: "Architect"
  tier: "manager"

llm:
  primary:
    provider: "google"
    model: "gemini-2.5-flash"
    temperature: 0.3
    max_tokens: 4096
  backup:
    provider: "anthropic"
    model: "claude-haiku-4"
    temperature: 0.3
    max_tokens: 4096

resources:
  max_concurrent_tasks: 5
  max_programmers: 5
  timeout_s: 600
```

### Programmer Config Template

`configs/pas/programmer_qwen_001.yaml`:
```yaml
service:
  name: "Programmer-Qwen-001"
  port: 6151
  host: "127.0.0.1"
  tier: 5

agent_metadata:
  agent_id: "Prog-Qwen-001"
  role: "programmer"
  parent: "Mgr-Code-01"
  grandparent: "Dir-Code"
  tier: "programmer"
  tool: "aider"

llm:
  primary:
    provider: "ollama"
    model: "qwen2.5-coder:7b-instruct"
    base_url: "http://localhost:11434"
    temperature: 0.2
    max_tokens: 8192
  backup:
    provider: "openai"
    model: "gpt-5-codex"
    temperature: 0.2
    max_tokens: 8192

aider:
  timeout_s: 300
  max_retries: 3
  fs_allowlist: "configs/pas/fs_allowlist.yaml"
  cmd_allowlist: "configs/pas/cmd_allowlist.yaml"
```

---

## Migration Strategy

### Backward Compatibility

1. **Keep Aider-LCO (6130) running temporarily**
   - Deprecate but don't remove
   - Fallback if all Programmers are busy
   - Remove after 1 week of stable operation

2. **Graceful Migration**
   - Phase 1: Managers only (Directors still use Aider RPC)
   - Phase 2: Programmers (Directors use new Programmer services)
   - Phase 3: Remove Aider RPC dependency

3. **Rollback Plan**
   - If critical bugs: Revert to P0 scaffold
   - Keep `services/common/manager_pool/` (old implementation)
   - Keep `services/tools/aider_rpc/` (old implementation)
   - Feature flag: `LNSP_USE_LEGACY_MANAGERS=1`

---

## Testing Plan

### Unit Tests

- `tests/test_manager_service.py` - Manager HTTP endpoints
- `tests/test_programmer_service.py` - Programmer HTTP endpoints
- `tests/test_programmer_pool.py` - Pool load balancing

### Integration Tests

- `tests/test_manager_programmer_e2e.py` - Full flow
- `tests/test_parallel_execution.py` - Multiple Programmers
- `tests/test_llm_switching.py` - Primary/backup fallback

### Performance Tests

- `tests/test_parallelization_speedup.py` - Serial vs parallel
- `tests/test_programmer_pool_saturation.py` - 10+ jobs

---

## Risks & Mitigations

| **Risk** | **Impact** | **Mitigation** |
|----------|-----------|----------------|
| Port exhaustion (49 Programmers) | High | Start with 10, scale on demand |
| LLM API costs (Claude/GPT) | Medium | Default to Ollama (free), API opt-in |
| Programmer starvation (all busy) | Medium | Queue jobs, show wait time in HMI |
| Config file explosion (7 Managers + 10 Programmers) | Low | Template generation script |
| Service startup time (17 new services) | Low | Parallel startup, health checks |

---

## Success Criteria

### Phase 1 Complete (Managers)
- ✅ 7 Manager services running (ports 6141-6147)
- ✅ Directors delegate via HTTP (not ManagerFactory)
- ✅ Manager health checks pass
- ✅ E2E test passes (Director → Manager → Aider RPC)

### Phase 2 Complete (Programmers)
- ✅ 10 Programmer services running (ports 6151-6160)
- ✅ Managers delegate via HTTP (not Aider RPC)
- ✅ Programmer Pool load balancing works
- ✅ E2E test shows parallel execution (>1 Programmer)

### Phase 3 Complete (WebUI)
- ✅ LLM dropdowns functional (persist to configs)
- ✅ HMI shows Manager/Programmer status
- ✅ Tree View shows parallelization

### Phase 4 Complete (Testing)
- ✅ All tests pass (unit, integration, performance)
- ✅ 5x speedup on multi-file tasks
- ✅ P95 latency < 30s (simple tasks)
- ✅ Throughput > 10 jobs/minute

---

## Rollout Plan

### Week 1 (2025-11-12 to 2025-11-18)
- **Day 1-2:** Phase 1 (Managers)
- **Day 3-4:** Phase 2 (Programmers)
- **Day 5:** Phase 3 (WebUI)
- **Day 6:** Phase 4 (Testing)
- **Day 7:** Documentation, wrap-up

### Week 2 (2025-11-19 to 2025-11-25)
- Monitor production usage
- Scale Programmers if needed (up to 49)
- Deprecate Aider-LCO (6130)
- Remove legacy ManagerFactory code

---

## Open Questions

1. **Manager count per lane?**
   - Code: 3 Managers (high volume)
   - Others: 1 Manager each (lower volume)
   - Scale dynamically?

2. **Programmer distribution?**
   - 5x Qwen (free, fast)
   - 2x Claude (expensive, high quality)
   - 1x GPT (expensive, high quality)
   - 2x DeepSeek (free, experimental)
   - Adjust based on usage?

3. **LLM fallback strategy?**
   - Primary fails → Backup immediately?
   - Primary fails → Retry 3x, then Backup?
   - Track success rates per model?

4. **Programmer auto-scaling?**
   - Start with 10, scale to 49 based on queue depth?
   - Shut down idle Programmers to save resources?

---

## Appendix: File Structure

```
services/
├── pas/
│   ├── manager_code_01/
│   │   ├── app.py
│   │   ├── config.yaml
│   │   └── decomposer.py
│   ├── manager_code_02/
│   ├── manager_code_03/
│   ├── manager_models_01/
│   ├── manager_data_01/
│   ├── manager_devsecops_01/
│   └── manager_docs_01/
├── tools/
│   ├── programmer_qwen_001/
│   │   ├── app.py
│   │   ├── config.yaml
│   │   └── executor.py
│   ├── programmer_qwen_002/
│   ├── programmer_qwen_003/
│   ├── programmer_qwen_004/
│   ├── programmer_qwen_005/
│   ├── programmer_claude_001/
│   ├── programmer_claude_002/
│   ├── programmer_gpt_001/
│   ├── programmer_deepseek_001/
│   └── programmer_deepseek_002/
└── common/
    └── programmer_pool.py

scripts/
├── start_all_managers.sh
└── start_all_programmers.sh

configs/pas/
├── manager_code_01.yaml
├── manager_code_02.yaml
├── manager_code_03.yaml
├── programmer_qwen_001.yaml
├── programmer_qwen_002.yaml
└── ...
```

---

## References

- `docs/SERVICE_PORTS.md` - Port allocation
- `MANAGER_TIER_COMPLETE.md` - Current Manager implementation
- `test_manager_e2e.py` - Existing E2E test
- `services/pas/director_code/app.py` - Director template
- `services/tools/aider_rpc/app.py` - Legacy Aider RPC
